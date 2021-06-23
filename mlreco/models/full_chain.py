from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.clustercnn_se import ClusterCNN, ClusteringLoss
from mlreco.models.graph_spice import GraphSPICE, GraphSPICELoss
from mlreco.models.grappa import GNN, GNNLoss

from mlreco.models.layers.cnn_encoder import ResidualEncoder

from mlreco.utils.deghosting import adapt_labels
from mlreco.utils.cluster.graph_spice import ClusterGraphConstructor
from mlreco.utils.cluster.fragmenter import *

from mlreco.utils.gnn.evaluation import (node_assignment_score, 
                                         primary_assignment)

from mlreco.utils.gnn.cluster import (form_clusters, 
                                      get_cluster_batch, 
                                      get_cluster_label, 
                                      cluster_direction)

class FullChain(torch.nn.Module):
    """
    Modular, End-to-end LArTPC Reconstruction Chain

    - Deghosting for 3D tomographic reconstruction artifiact removal
    - UResNet for voxel-wise semantic segmentation
    - PPN for point proposal
    - DBSCAN/PILOT/SPICE for dense particle clustering
    - GrapPA(s) for particle aggregation and identification
    - CNN for interaction classification

    Configuration goes under the ``modules`` section.
    The full chain-related sections (as opposed to each
    module-specific configuration) look like this:

    ..  code-block:: yaml

          modules:
            chain:
              enable_uresnet: True
              enable_ppn: True
              enable_cnn_clust: True
              enable_gnn_shower: True
              enable_gnn_track: True
              enable_gnn_particle: False
              enable_gnn_inter: True
              enable_gnn_kinematics: False
              enable_cosmic: False
              enable_ghost: True
              use_ppn_in_gnn: True
              verbose: True


            # full chain loss and weighting
            full_chain_loss:
              segmentation_weight: 1.
              clustering_weight: 1.
              ppn_weight: 1.
              particle_gnn_weight: 1.
              shower_gnn_weight: 1.
              track_gnn_weight: 1.
              inter_gnn_weight: 1.
              kinematics_weight: 1.
              kinematics_p_weight: 1.
              kinematics_type_weight: 1.
              flow_weight: 1.
              cosmic_weight: 1.

    The ``chain`` section enables or disables specific
    stages of the full chain. When a module is disabled
    through this section, it will not even be constructed.
    The section ``full_chain_loss`` allows
    to set different weights to the losses of different stages.
    The configuration blocks for each enabled module should
    also live under the `modules` section of the configuration.
    """
    MODULES = ['grappa_shower', 'grappa_track', 'grappa_inter',
               'grappa_shower_loss', 'grappa_track_loss', 'grappa_inter_loss',
               'full_chain_loss', 'spice', 'spice_loss',
               'fragment_clustering',  'chain', 'dbscan_frag',
               ('uresnet_ppn', ['uresnet_lonely', 'ppn'])]

    def __init__(self, cfg):
        super(FullChain, self).__init__()

        # Configure the chain first
        setup_chain_cfg(self, cfg)

        # Initialize the UResNet+PPN modules
        if self.enable_uresnet:
            self.uresnet_lonely = UResNet(cfg['uresnet_ppn'])
            self.input_features = cfg['uresnet_ppn']['uresnet_lonely'].get('features', 1)

        if self.enable_ppn:
            self.ppn            = PPN(cfg['uresnet_ppn'])

        # Initialize the CNN dense clustering module
        self.cluster_classes = []
        if self.enable_cnn_clust:
            self._enable_graph_spice = 'graph_spice' in cfg
            if self._enable_graph_spice:
                self.spatial_embeddings   = GraphSPICE(cfg)
                self.gs_manager           = ClusterGraphConstructor(cfg['graph_spice']['constructor_cfg'])
                self.gs_manager.training  = True # FIXME
                self._gspice_skip_classes = cfg['graph_spice']['skip_classes']
                self.fragment_manager     = GraphSPICEFragmentManager(cfg['fragment_manager'])
            else:
                self.spatial_embeddings = ClusterCNN(cfg['spice'])
                self.frag_cfg           = cfg['spice'].get('fragment_manager', {})
                self._s_thresholds      = self.frag_cfg.setdefault('s_thresholds', [0.0, 0.0, 0.0, 0.0])
                self._p_thresholds      = self.frag_cfg.setdefault('p_thresholds', [0.5, 0.5, 0.5, 0.5])
                self._spice_classes     = self.frag_cfg.setdefault('cluster_classes', [])
                self._spice_min_voxels  = self.frag_cfg.setdefault('min_voxels', 2)
                self.fragment_manager   = SPICEFragmentManager(self.frag_cfg)

        # Initialize the DBSCAN fragmenter module
        if self.enable_dbscan:
            self.frag_cfg = cfg['fragment_manager']
            self.fragment_manager = DBSCANFragmentManager(self.frag_cfg)

        # Initialize the particle aggregator modules
        if self.enable_gnn_shower:
            self.grappa_shower     = GNN(cfg, name='grappa_shower')
            self._shower_id        = cfg['grappa_shower']['base'].get('node_type', 0)

        if self.enable_gnn_track:
            self.grappa_track      = GNN(cfg, name='grappa_track')
            self._track_id         = cfg['grappa_track']['base'].get('node_type', 1)

        if self.enable_gnn_particle:
            self.grappa_particle   = GNN(cfg, name='grappa_particle')
            self._particle_ids     = cfg['grappa_particle']['base'].get('node_type', [0,1])

        if self.enable_gnn_inter:
            self.grappa_inter      = GNN(cfg, name='grappa_inter')
            self._inter_ids        = cfg['grappa_inter']['base'].get('node_type', [0,1,2,3])

        if self.enable_gnn_kinematics:
            self.grappa_kinematics = GNN(cfg, name='grappa_kinematics')
            self._kinematics_use_true_particles = cfg['grappa_kinematics'].get(
                'use_true_particles', False)

        # Initialize the interaction classifier module
        if self.enable_cosmic:
            self.cosmic_discriminator = ResidualEncoder(cfg['cosmic_discriminator'])
            self._cosmic_use_input_data = cfg['cosmic_discriminator'].get(
                'use_input_data', True)
            self._cosmic_use_true_interactions = cfg['cosmic_discriminator'].get(
                'use_true_interactions', False)


    def get_extra_gnn_features(self, fragments, frag_seg, classes, input, result,
                               use_ppn=False, use_supp=False):
        """
        Extracting extra features to feed into the GNN particle aggregators

        - PPN: Most likely PPN point for showers,
               end points for tracks (+ direction estimate)
        - Supplemental: Mean/RMS energy in the fragment + semantic class

        Parameters
        ==========
        fragments: np.ndarray
        frag_seg: np.ndarray
        classes: list
        input: list
        result: dictionary
        use_ppn: bool
        use_supp: bool

        Returns
        =======
        mask: np.ndarray
            Boolean mask to select fragments belonging to one
            of the requested classes.
        kwargs: dictionary
            Keys can include `points` (if `use_ppn` is `True`)
            and `extra_feats` (if `use_supp` is True).
        """
        # Build a mask for the requested classes
        mask = np.zeros(len(frag_seg), dtype=np.bool)
        for c in classes:
            mask |= (frag_seg == c)
        mask = np.where(mask)[0]

        # If requested, extract PPN-related features
        kwargs = {}
        if use_ppn:
            points_tensor = result['points'][0].detach().double()
            ppn_points = torch.empty((0,6), device=input[0].device,
                                            dtype=torch.double)
            for i, f in enumerate(fragments[mask]):
                if frag_seg[mask][i] == 1:
                    dist_mat = torch.cdist(input[0][f,:3], input[0][f,:3])
                    idx = torch.argmax(dist_mat)
                    idxs = int(idx)//len(f), int(idx)%len(f)
                    scores = torch.nn.functional.softmax(points_tensor[f, 3:5], dim=1)[:, 1]
                    correction0 = points_tensor[f][idxs[0], :3] + \
                                  0.5 if scores[idxs[0]] > 0.5 else 0.0
                    correction1 = points_tensor[f][idxs[1], :3] + \
                                  0.5 if scores[idxs[1]] > 0.5 else 0.0
                    end_points = torch.cat([input[0][f[idxs[0]],:3] + correction0,
                                            input[0][f[idxs[1]],:3] + correction1]).reshape(1,-1)
                    ppn_points = torch.cat((ppn_points, end_points), dim=0)
                else:
                    dmask  = torch.nonzero(torch.max(
                        torch.abs(points_tensor[f,:3]), dim=1).values < 1.,
                        as_tuple=True)[0]
                    scores = torch.softmax(points_tensor[f,3:5], dim=1)
                    argmax = dmask[torch.argmax(scores[dmask,-1])] \
                             if len(dmask) else torch.argmax(scores[:,-1])
                    start  = input[0][f][argmax,:3] + \
                             points_tensor[f][argmax,:3] + 0.5
                    ppn_points = torch.cat((ppn_points,
                        torch.cat([start, start]).reshape(1,-1)), dim=0)

            kwargs['points'] = ppn_points

        # If requested, add energy and semantic related features
        if use_supp:
            supp_feats = torch.empty((0,3), device=input[0].device,
                                            dtype=torch.float)
            for i, f in enumerate(fragments[mask]):
                values = torch.cat((input[0][f,4].mean().reshape(1),
                                    input[0][f,4].std().reshape(1))).float()
                if torch.isnan(values[1]): # Handle size-1 particles
                    values[1] = input[0][f,4] - input[0][f,4]
                sem_type = torch.tensor([frag_seg[mask][i]],
                                        dtype=torch.float,
                                        device=input[0].device)
                supp_feats = torch.cat((supp_feats,
                    torch.cat([values, sem_type.reshape(1)]).reshape(1,-1)), dim=0)

            kwargs['extra_feats'] = supp_feats

        return mask, kwargs


    def run_gnn(self, grappa, input, result, clusts, labels, kwargs={}):
        """
        Generic function to group in one place the common code to run a GNN model.

        Parameters
        ==========
        - grappa: GrapPA module to run
        - input: input data
        - result: dictionary
        - clusts: list of list of indices (indexing input data)
        - labels: dictionary of strings to label the final result
        - kwargs: extra arguments to pass to the gnn

        Returns
        =======
        None (modifies the result dict in place)
        """

        # Pass data through the GrapPA model
        gnn_output = grappa(input, clusts, **kwargs)

        # Update the result dictionary if the corresponding label exists
        for l, tag in labels.items():
            if l in gnn_output.keys():
                result.update({tag: gnn_output[l]})

        # Make group predictions based on the GNN output, if requested
        if 'group_pred' in labels:
            group_ids = []
            for b in range(len(gnn_output['clusts'][0])):
                if len(gnn_output['clusts'][0][b]) < 2:
                    group_ids.append(np.zeros(len(gnn_output['clusts'][0][b]),
                                     dtype=np.int64))
                else:
                    group_ids.append(node_assignment_score(
                        gnn_output['edge_index'][0][b],
                        gnn_output['edge_pred'][0][b].detach().cpu().numpy(),
                        len(gnn_output['clusts'][0][b])))

            result.update({labels['group_pred']: [group_ids]})


    def select_particle_in_group(self, result, counts, b, particles,
                                 part_primary_ids,
                                 node_pred,
                                 group_pred,
                                 fragments):
        """
        Merge fragments into particle instances, retain
        primary fragment id of each group
        """
        voxel_inds = counts[:b].sum().item()+np.arange(counts[b].item())
        primary_labels = None
        if node_pred in result:
            primary_labels = primary_assignment(
                result[node_pred][0][b].detach().cpu().numpy(),
                result[group_pred][0][b])

        for g in np.unique(result[group_pred][0][b]):
            group_mask = np.where(result[group_pred][0][b] == g)[0]
            particles.append(
                voxel_inds[np.concatenate(result[fragments][0][b][group_mask])])
            if node_pred in result:
                primary_id = group_mask[primary_labels[group_mask]][0]
                part_primary_ids.append(primary_id)
            else:
                part_primary_ids.append(g)


    def full_chain_cnn(self, input):
        '''
        Run the CNN portion of the full chain.

        Parameters
        ==========
        input:

        result:

        Returns
        =======
        result: dict
            dictionary of all network outputs from cnns.
        '''
        device = input[0].device

        label_seg, label_clustering = None, None
        if len(input) == 3:
            input, label_seg, label_clustering = input
            input = [input]
            label_seg = [label_seg]
            label_clustering = [label_clustering]
        if len(input) == 2:
            input, label_clustering = input
            input = [input]
            label_clustering = [label_clustering]

        result = {}
        if self.enable_uresnet:
            result = self.uresnet_lonely([input[0][:,:4+self.input_features]])
        if self.enable_ppn:
            ppn_input = {}
            ppn_input.update(result)
            ppn_input['ppn_feature_enc'] = ppn_input['ppn_feature_enc'][0]
            ppn_input['ppn_feature_dec'] = ppn_input['ppn_feature_dec'][0]
            if 'ghost' in ppn_input:
                ppn_input['ghost'] = ppn_input['ghost'][0]
            ppn_output = self.ppn(ppn_input)
            result.update(ppn_output)

        # The rest of the chain only needs 1 input feature
        if self.input_features > 1:
            input[0] = input[0][:, :-self.input_features+1]

        cnn_result = {}

        if self.enable_ghost:
            # Update input based on deghosting results
            deghost = result['ghost'][0].argmax(dim=1) == 0
            input = [input[0][deghost]]
            if label_seg is not None and label_clustering is not None:
                label_clustering = adapt_labels(result,
                                                label_seg,
                                                label_clustering)

            segmentation = result['segmentation'][0].clone()
            ppn_feature_dec = [x.features.clone() for x in result['ppn_feature_dec'][0]]
            if self.enable_ppn:
                points, mask_ppn2 = result['points'][0].clone(), \
                                    result['mask_ppn2'][0].clone()

            deghost_result = {}
            deghost_result.update(result)
            deghost_result.pop('ghost')
            deghost_result['segmentation'][0] = result['segmentation'][0][deghost]
            deghost_result['ppn_feature_dec'][0] = [result['ppn_feature_dec'][0][-1].features[deghost]]
            if self.enable_ppn:
                deghost_result['points'][0] = result['points'][0][deghost]
                deghost_result['mask_ppn2'][0] = result['mask_ppn2'][0][deghost]
            cnn_result.update(deghost_result)
            cnn_result['ghost'] = result['ghost']

        else:
            cnn_result.update(result)

        # ---
        # 1. Clustering w/ CNN or DBSCAN will produce
        # - fragments (list of list of integer indexing the input data)
        # - frag_batch_ids (list of batch ids for each fragment)
        # - frag_seg (list of integers, semantic label for each fragment)
        # ---
        cluster_result = {}
        semantic_labels = torch.argmax(cnn_result['segmentation'][0],
                                       dim=1).flatten().double()

        if self.enable_cnn_clust:
            if self._enable_graph_spice:
                if label_clustering is None:
                    raise Exception("Cluster labels from parse_cluster3d_clean_full are needed at this time.")

                filtered_semantic = ~(semantic_labels[..., None].cpu() == \
                                      torch.Tensor(self._gspice_skip_classes)).any(-1)

                # If there are voxels to process in the given semantic classes
                if torch.count_nonzero(filtered_semantic) > 0:
                    graph_spice_label = torch.cat((label_clustering[0][:, :-1],
                                                   semantic_labels.reshape(-1,1)), dim=1)
                    spatial_embeddings_output = self.spatial_embeddings((input[0][:,:5],
                                                                         graph_spice_label))
                    cnn_result.update(spatial_embeddings_output)

                    self.gs_manager.replace_state(spatial_embeddings_output['graph'][0],
                                                  spatial_embeddings_output['graph_info'][0])

                    self.gs_manager.fit_predict(gen_numpy_graph=True)
                    cluster_predictions = self.gs_manager._node_pred.x
                    filtered_input = torch.cat([input[0][filtered_semantic][:, :4],
                                                semantic_labels[filtered_semantic][:, None],
                                                cluster_predictions.to(device)[:, None]], dim=1)

                    if self.process_fragments:
                        cluster_result = self.fragment_manager(filtered_input)

            else:
                # Get fragment predictions from the CNN clustering algorithm
                spatial_embeddings_output = self.spatial_embeddings([input[0][:,:5]])
                cnn_result.update(spatial_embeddings_output)

                # Extract fragment predictions to input into the GNN
                if self.process_fragments:
                    cluster_result = self.fragment_manager(input[0],
                                                           cnn_result)

        if self.enable_dbscan and self.process_fragments:
            # Get the fragment predictions from the DBSCAN fragmenter
            cluster_result = self.fragment_manager(input[0], cnn_result)

        # Make np.array
        cnn_result.update(cluster_result)

        cnn_result.update({
            'label_clustering': [label_clustering],
            'semantic_labels': [semantic_labels],
        })

        def return_to_original(result):
            if self.enable_ghost:
                result['segmentation'][0] = segmentation
                result['ppn_feature_dec'][0] = ppn_feature_dec
                if self.enable_ppn:
                    result['points'][0] = points
                    result['mask_ppn2'][0] = mask_ppn2
            return result

        return cnn_result, input, return_to_original


    def full_chain_gnn(self, result, input):
        # ---
        # 2. GNN particle instance formation
        # ---
        device = input[0].device
        _, counts = torch.unique(input[0][:, 3], return_counts=True)
        fragments = result['frags'][0]
        frag_seg = result['frag_seg'][0]
        frag_batch_ids = result.pop('frag_batch_ids')[0]
        label_clustering = result.pop('label_clustering')[0]
        semantic_labels = result.pop('semantic_labels')[0]
        vids = result.pop('vids')[0]

        if self.enable_gnn_shower:
            # Run shower GrapPA: merges shower fragments into shower instances
            em_mask, kwargs = self.get_extra_gnn_features(fragments,
                                                          frag_seg,
                                                          [self._shower_id],
                                                          input,
                                                          result,
                                                          use_ppn=self.use_ppn_in_gnn,
                                                          use_supp=True)

            output_keys = {'clusts': 'shower_fragments',
                           'node_pred': 'shower_node_pred',
                           'edge_pred': 'shower_edge_pred',
                           'edge_index': 'shower_edge_index',
                           'group_pred': 'shower_group_pred'}

            self.run_gnn(self.grappa_shower,
                         input,
                         result,
                         fragments[em_mask],
                         output_keys,
                         kwargs)

        if self.enable_gnn_track:
            # Run track GrapPA: merges tracks fragments into track instances
            track_mask, kwargs = self.get_extra_gnn_features(fragments,
                                                             frag_seg,
                                                             [self._track_id],
                                                             input,
                                                             result,
                                                             use_ppn=self.use_ppn_in_gnn,
                                                             use_supp=True)

            output_keys = {'clusts': 'track_fragments',
                           'node_pred': 'track_node_pred',
                           'edge_pred': 'track_edge_pred',
                           'edge_index': 'track_edge_index',
                           'group_pred': 'track_group_pred'}

            self.run_gnn(self.grappa_track,
                         input,
                         result,
                         fragments[track_mask],
                         output_keys,
                         kwargs)

        if self.enable_gnn_particle:
            # Run particle GrapPA: merges particle fragments or labels in _partile_ids together into particle instances
            mask, kwargs = self.get_extra_gnn_features(fragments,
                                                       frag_seg,
                                                       self._particle_ids,
                                                       input,
                                                       result,
                                                       use_ppn=self.use_ppn_in_gnn,
                                                       use_supp=True)

            kwargs['groups'] = frag_seg[mask]

            output_keys = {'clusts': 'particle_fragments',
                           'node_pred': 'particle_node_pred',
                           'edge_pred': 'particle_edge_pred',
                           'edge_index': 'particle_edge_index',
                           'group_pred': 'particle_group_pred'}

            self.run_gnn(self.grappa_particle,
                         input,
                         result,
                         fragments[mask],
                         output_keys,
                         kwargs)

        # Merge fragments into particle instances, retain primary fragment id of showers
        particles, part_primary_ids = [], []
        for b in range(len(counts)):
            mask = (frag_batch_ids == b)
            # Append one particle per particle group
            if self.enable_gnn_particle:

                self.select_particle_in_group(result, counts, b, particles,
                                              part_primary_ids,
                                              'particle_node_pred',
                                              'particle_group_pred',
                                              'particle_fragments')

                for c in self._particle_ids:
                    mask &= (frag_seg != c)
            # Append one particle per shower group
            if self.enable_gnn_shower:

                self.select_particle_in_group(result, counts, b, particles,
                                              part_primary_ids,
                                              'shower_node_pred',
                                              'shower_group_pred',
                                              'shower_fragments')

                mask &= (frag_seg != self._shower_id)
            # Append one particle per track group
            if self.enable_gnn_track:

                self.select_particle_in_group(result, counts, b, particles,
                                              part_primary_ids,
                                              'track_node_pred',
                                              'track_group_pred',
                                              'track_fragments')

                mask &= (frag_seg != self._track_id)

            # Append one particle per fragment that is not already accounted for
            particles.extend(fragments[mask])
            part_primary_ids.extend(-np.ones(np.sum(mask)))

        same_length = np.all([len(p) == len(particles[0]) for p in particles])
        particles = np.array(particles,
                             dtype=object if not same_length else np.int64)
        part_batch_ids = get_cluster_batch(input[0], particles)
        part_primary_ids = np.array(part_primary_ids, dtype=np.int32)
        part_seg = np.empty(len(particles), dtype=np.int32)
        for i, p in enumerate(particles):
            vals, cnts = semantic_labels[p].unique(return_counts=True)
            assert len(vals) == 1
            part_seg[i] = vals[torch.argmax(cnts)].item()

        # Store in result the intermediate fragments
        bcids = [np.where(part_batch_ids == b)[0] for b in range(len(counts))]
        same_length = [np.all([len(c) == len(particles[b][0]) \
                       for c in particles[b]] ) for b in bcids]

        parts = [np.array([vids[c].astype(np.int64) for c in particles[b]],
                          dtype=np.object \
                          if not same_length[idx] \
                          else np.int64) for idx, b in enumerate(bcids)]

        parts_seg = [part_seg[b] for idx, b in enumerate(bcids)]

        result.update({
            'particles': [parts],
            'particles_seg': [parts_seg]
        })

        # ---
        # 3. GNN interaction clustering
        # ---

        if self.enable_gnn_inter:
            # For showers, select primary for extra feature extraction
            extra_feats_particles = []
            for i, p in enumerate(particles):
                if part_seg[i] == 0:

                    voxel_inds = counts[:part_batch_ids[i]].sum().item() + \
                                 np.arange(counts[part_batch_ids[i]].item())

                    p = voxel_inds[result['fragments'][0]\
                                  [part_batch_ids[i]][part_primary_ids[i]]]
                extra_feats_particles.append(p)
            same_length = np.all([len(p) == len(extra_feats_particles[0]) \
                                 for p in extra_feats_particles])

            extra_feats_particles = np.array(extra_feats_particles,
                                             dtype=object \
                                             if not same_length else np.int64)

            # Run interaction GrapPA: merges particle instances into interactions
            _, kwargs = self.get_extra_gnn_features(extra_feats_particles,
                                                    part_seg,
                                                    self._inter_ids,
                                                    input,
                                                    result,
                                                    use_ppn=self.use_ppn_in_gnn,
                                                    use_supp=True)

            output_keys = {'clusts': 'inter_particles',
                           'edge_pred': 'inter_edge_pred',
                           'edge_index': 'inter_edge_index',
                           'group_pred': 'inter_group_pred',
                           'node_pred': 'inter_node_pred',
                           'node_pred_type': 'node_pred_type',
                           'node_pred_p': 'node_pred_p',
                           'node_pred_vtx': 'node_pred_vtx'}

            self.run_gnn(self.grappa_inter,
                         input,
                         result,
                         particles,
                         output_keys,
                         kwargs)

        # ---
        # 4. GNN for particle flow & kinematics
        # ---

        if self.enable_gnn_kinematics:
            #print(len(particles))
            if self._kinematics_use_true_particles:
                if label_clustering is None:
                    raise Exception("The option to use true interactions requires label segmentation and clustering in the network input.")
                # Also exclude lowE
                kinematics_particles = form_clusters(label_clustering[0], column=6)
                kinematics_particles = [part.cpu().numpy() for part in kinematics_particles]
                kinematics_part_batch_ids = get_cluster_batch(input[0], kinematics_particles)
                kinematics_particles = np.array(kinematics_particles, dtype=object)
                kinematics_particles_seg = get_cluster_label(label_clustering[0], kinematics_particles, column=-1)
                kinematics_particles = kinematics_particles[kinematics_particles_seg<4]
            else:
                kinematics_particles = particles
                kinematics_part_batch_ids = part_batch_ids

            output_keys = {'clusts': 'kinematics_particles',
                           'edge_index': 'kinematics_edge_index',
                           'node_pred_p': 'node_pred_p',
                           'node_pred_type': 'node_pred_type',
                           'edge_pred': 'flow_edge_pred'}

            self.run_gnn(self.grappa_kinematics, input, result, kinematics_particles, output_keys)

        # ---
        # 5. CNN for interaction classification
        # ---

        if self.enable_cosmic:
            if not self.enable_gnn_inter and not self._cosmic_use_true_interactions:
                raise Exception("Need interaction clustering before cosmic discrimination.")

            _, counts = torch.unique(input[0][:,3], return_counts=True)
            interactions, inter_primary_ids = [], []
            # Note to self: inter_primary_ids is not used as of now

            if self._cosmic_use_true_interactions:
                if label_clustering is None:
                    raise Exception("The option to use true interactions requires label segmentation and clustering in the network input.")
                interactions = form_clusters(label_clustering[0], column=7)
                interactions = [inter.cpu().numpy() for inter in interactions]
            else:
                for b in range(len(counts)):

                    self.select_particle_in_group(result, counts, b, interactions, inter_primary_ids,
                                                  None, 'inter_group_pred', 'particles')

            inter_batch_ids = get_cluster_batch(input[0], interactions)
            inter_cosmic_pred = torch.empty((len(interactions), 2), dtype=torch.float)

            # Replace batch id column with a global "interaction id"
            # because ResidualEncoder uses the batch id column to shape its output
            feature_map = result['ppn_feature_dec'][0][-1]
            if not torch.is_tensor(feature_map):
                feature_map = feature_map.features

            inter_input_data = input[0].float() if self._cosmic_use_input_data \
                                                else torch.cat([input[0][:, :4].float(), feature_map], dim=1)

            inter_data = torch.empty((0, inter_input_data.size(1)), dtype=torch.float, device=device)
            for i, interaction in enumerate(interactions):
                inter_data = torch.cat([inter_data, inter_input_data[interaction]], dim=0)
                inter_data[-len(interaction):, 3] = i * torch.ones(len(interaction)).to(device)
            inter_cosmic_pred = self.cosmic_discriminator(inter_data)

            # Reorganize into batches before storing in result dictionary
            same_length = np.all([len(f) == len(interactions[0]) for f in interactions] )
            interactions = np.array(interactions, dtype=object if not same_length else np.int64)
            inter_batch_ids = np.array(inter_batch_ids)

            _, counts = torch.unique(input[0][:,3], return_counts=True)
            vids = np.concatenate([np.arange(n.item()) for n in counts])
            bcids = [np.where(inter_batch_ids == b)[0] for b in range(len(counts))]
            same_length = [np.all([len(c) == len(interactions[b][0]) for c in interactions[b]] ) for b in bcids]

            interactions_np = [np.array([vids[c].astype(np.int64) for c in interactions[b]],
                               dtype=np.object if not same_length[idx] else np.int64) \
                                   for idx, b in enumerate(bcids)]

            inter_cosmic_pred_np = [inter_cosmic_pred[b] for idx, b in enumerate(bcids)]

            result.update({
                'interactions': [interactions_np],
                'inter_cosmic_pred': [inter_cosmic_pred_np]
                })

        return result


    def forward(self, input):
        """
        Input can be either of the following:
        - input data only
        - input data, label clustering in this order
        - input data, label segmentation, label clustering in this order
        (when deghosting is enabled, label segmentation is needed to
        adapt label clustering properly)

        Parameters
        ==========
        input: list of np.ndarray
        """
        result, input, revert_func = self.full_chain_cnn(input)
        if self.process_fragments:
            result = self.full_chain_gnn(result, input)

        result = revert_func(result)

        return result


class FullChainLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(FullChainLoss, self).__init__()

        # Configure the chain first
        setup_chain_cfg(self, cfg)

        # Initialize loss components
        if self.enable_uresnet:
            self.uresnet_loss            = SegmentationLoss(cfg['uresnet_ppn'])
        if self.enable_ppn:
            self.ppn_loss                = PPNLoss(cfg['uresnet_ppn'])
        if self.enable_cnn_clust:
            self._enable_graph_spice = 'graph_spice' in cfg
            if not self._enable_graph_spice:
                self.spatial_embeddings_loss = ClusteringLoss(cfg)
            else:
                self.spatial_embeddings_loss = GraphSPICELoss(cfg)
                self._gspice_skip_classes = cfg['spice_loss']['skip_classes']
        if self.enable_gnn_shower:
            self.shower_gnn_loss         = GNNLoss(cfg, 'grappa_shower_loss')
        if self.enable_gnn_track:
            self.track_gnn_loss          = GNNLoss(cfg, 'grappa_track_loss')
        if self.enable_gnn_particle:
            self.particle_gnn_loss       = GNNLoss(cfg, 'grappa_particle_loss')
        if self.enable_gnn_inter:
            self.inter_gnn_loss          = GNNLoss(cfg, 'grappa_inter_loss')
        if self.enable_gnn_kinematics:
            self.kinematics_loss         = GNNLoss(cfg, 'grappa_kinematics_loss')
        if self.enable_cosmic:
            self.cosmic_loss             = GNNLoss(cfg, 'cosmic_loss')

        # Initialize the loss weights
        self.loss_config = cfg['full_chain_loss']

        self.segmentation_weight    = self.loss_config.get('segmentation_weight', 1.0)
        self.ppn_weight             = self.loss_config.get('ppn_weight', 0.0)
        self.cnn_clust_weight       = self.loss_config.get('cnn_clust_weight', 1.0)
        self.shower_gnn_weight      = self.loss_config.get('shower_gnn_weight', 0.0)
        self.track_gnn_weight       = self.loss_config.get('track_gnn_weight', 0.0)
        self.particle_gnn_weight    = self.loss_config.get('particle_gnn_weight', 0.0)
        self.inter_gnn_weight       = self.loss_config.get('inter_gnn_weight', 0.0)
        self.kinematics_weight      = self.loss_config.get('kinematics_weight', 0.0)
        self.flow_weight            = self.loss_config.get('flow_weight', 0.0)
        self.kinematics_p_weight    = self.loss_config.get('kinematics_p_weight', 0.0)
        self.kinematics_type_weight = self.loss_config.get('kinematics_type_weight', 0.0)
        self.cosmic_weight          = self.loss_config.get('cosmic_weight', 0.0)

    def forward(self, out, seg_label, ppn_label=None, cluster_label=None, kinematics_label=None, particle_graph=None):
        res = {}
        accuracy, loss = 0., 0.

        if self.enable_uresnet:
            res_seg = self.uresnet_loss(out, seg_label)
            res.update(res_seg)
            res['seg_accuracy'] = res_seg['accuracy']
            res['seg_loss'] = res_seg['loss']
            accuracy += res_seg['accuracy']
            loss += self.segmentation_weight*res_seg['loss']

        if self.enable_ppn:
            # Apply the PPN loss
            res_ppn = self.ppn_loss(out, seg_label, ppn_label)
            res.update(res_ppn)
            res['ppn_accuracy'] = res_ppn['ppn_acc']
            res['ppn_loss'] = res_ppn['ppn_loss']

            accuracy += res_ppn['ppn_acc']
            loss += self.ppn_weight*res_ppn['ppn_loss']

        if self.enable_ghost and (self.enable_cnn_clust or \
                                  self.enable_gnn_track or \
                                  self.enable_gnn_shower or \
                                  self.enable_gnn_inter or \
                                  self.enable_gnn_kinematics or \
                                  self.enable_cosmic):
            # Adapt to ghost points
            if cluster_label is not None:
                cluster_label = adapt_labels(out, seg_label, cluster_label)
            if kinematics_label is not None:
                kinematics_label = adapt_labels(out, seg_label, kinematics_label)

            deghost = out['ghost'][0].argmax(dim=1) == 0
            #print("cluster_label", torch.unique(cluster_label[0][:, 7]), torch.unique(cluster_label[0][:, 6]), torch.unique(cluster_label[0][:, 5]))
            #result = self.full_chain_loss(out, res_seg, res_ppn, seg_label[0][deghost][:, -1], cluster_label)
            segment_label = seg_label[0][deghost][:, -1]
            seg_label = seg_label[0][deghost]
        else:
            #result = self.full_chain_loss(out, res_seg, res_ppn, seg_label[0][:, -1], cluster_label)
            segment_label = seg_label[0][:, -1]
            seg_label = seg_label[0]

        if self.enable_cnn_clust:
            if self._enable_graph_spice:
                graph_spice_out = {
                    'graph': out['graph'],
                    'graph_info': out['graph_info'],
                    'spatial_embeddings': out['spatial_embeddings'],
                    'feature_embeddings': out['feature_embeddings'],
                    'covariance': out['covariance'],
                    'hypergraph_features': out['hypergraph_features'],
                    'features': out['features'],
                    'occupancy': out['occupancy'],
                    'coordinates': out['coordinates'],
                    'batch_indices': out['batch_indices'],
                    'segmentation': [out['segmentation'][0][deghost]] if self.enable_ghost else [out['segmentation'][0]]
                }
                select_classes = ~(torch.argmax(graph_spice_out['segmentation'][0], dim=1)[:, None].cpu() == torch.tensor(self._gspice_skip_classes)).any(-1)
                graph_spice_out['segmentation'] = [graph_spice_out['segmentation'][0][select_classes]]
                # print(graph_spice_out['segmentation'][0].size())
                # print('segment_label', segment_label.size(), out['segmentation'][0][deghost].size())
                # print(out['spatial_embeddings'][0].size())
                # print(out['segmentation'][0][deghost][torch.argmax(out['segmentation'][0][deghost], dim=1) == 1].size())
                # print(out['batch_indices'][0].size())
                # FIXME seg_label or segmentation predictions?
                # print(torch.argmax(out['segmentation'][0], dim=1)[:, None].size())
                # FIXME deghost not always true
                gs_seg_label = torch.cat([cluster_label[0][:, :4], torch.argmax(out['segmentation'][0][deghost], dim=1)[:, None]], dim=1)
                # if self.enable_ghost:
                #     gs_seg_label = gs_seg_label[deghost]
                # print(gs_seg_label.size())
                # print(gs_seg_label[gs_seg_label[:, -1] ==1].size())
                res_graph_spice = self.spatial_embeddings_loss(graph_spice_out, [gs_seg_label], cluster_label)
                #print(res_graph_spice.keys())
                accuracy += res_graph_spice['accuracy']
                loss += self.cnn_clust_weight * res_graph_spice['loss']
                res['graph_spice_loss'] = res_graph_spice['loss']
                res['graph_spice_accuracy'] = res_graph_spice['accuracy']
                #res['graph_spice_edge_loss'] = res_graph_spice['edge_loss']
                #res['graph_spice_edge_accuracy'] = res_graph_spice['edge_accuracy']
                res['graph_spice_occ_loss'] = res_graph_spice['occ_loss']
                res['graph_spice_cov_loss'] = res_graph_spice['cov_loss']
                res['graph_spice_sp_intra'] = res_graph_spice['sp_intra']
                res['graph_spice_sp_inter'] = res_graph_spice['sp_inter']
                res['graph_spice_ft_intra'] = res_graph_spice['ft_intra']
                res['graph_spice_ft_inter'] = res_graph_spice['ft_inter']
                res['graph_spice_ft_reg'] = res_graph_spice['ft_reg']
            else:
                # Apply the CNN dense clustering loss to HE voxels only
                he_mask = segment_label < 4
                # sem_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,-1].view(-1,1)), dim=1)]
                #clust_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,5].view(-1,1),cluster_label[0][he_mask,4].view(-1,1)), dim=1)]
                clust_label = [cluster_label[0][he_mask].clone()]
                cnn_clust_output = {'embeddings':[out['embeddings'][0][he_mask]], 'seediness':[out['seediness'][0][he_mask]], 'margins':[out['margins'][0][he_mask]]}
                #cluster_label[0] = cluster_label[0][he_mask]
                # FIXME does this suppose that clust_label has same ordering as embeddings?
                res_cnn_clust = self.spatial_embeddings_loss(cnn_clust_output, clust_label)
                res.update(res_cnn_clust)
                res['cnn_clust_accuracy'] = res_cnn_clust['accuracy']
                res['cnn_clust_loss'] = res_cnn_clust['loss']

                accuracy += res_cnn_clust['accuracy']
                loss += self.cnn_clust_weight*res_cnn_clust['loss']

        if self.enable_gnn_shower:
            # Apply the GNN shower clustering loss
            gnn_out = {}
            if 'shower_edge_pred' in out:
                gnn_out = {
                    'clusts':out['shower_fragments'],
                    'node_pred':out['shower_node_pred'],
                    'edge_pred':out['shower_edge_pred'],
                    'edge_index':out['shower_edge_index']
                }
            res_gnn_shower = self.shower_gnn_loss(gnn_out, cluster_label)
            res['shower_edge_loss'] = res_gnn_shower['edge_loss']
            res['shower_node_loss'] = res_gnn_shower['node_loss']
            res['shower_edge_accuracy'] = res_gnn_shower['edge_accuracy']
            res['shower_node_accuracy'] = res_gnn_shower['node_accuracy']

            accuracy += res_gnn_shower['accuracy']
            loss += self.shower_gnn_weight*res_gnn_shower['loss']

        if self.enable_gnn_track:
            # Apply the GNN track clustering loss
            gnn_out = {}
            if 'track_edge_pred' in out:
                gnn_out = {
                    'clusts':out['track_fragments'],
                    'edge_pred':out['track_edge_pred'],
                    'edge_index':out['track_edge_index']
                }
            res_gnn_track = self.track_gnn_loss(gnn_out, cluster_label)
            res['track_edge_loss'] = res_gnn_track['loss']
            res['track_edge_accuracy'] = res_gnn_track['accuracy']

            accuracy += res_gnn_track['accuracy']
            loss += self.track_gnn_weight*res_gnn_track['loss']

        if self.enable_gnn_particle:
            # Apply the GNN particle clustering loss
            gnn_out = {}
            if 'particle_edge_pred' in out:
                gnn_out = {
                    'clusts':out['particle_fragments'],
                    'node_pred':out['particle_node_pred'],
                    'edge_pred':out['particle_edge_pred'],
                    'edge_index':out['particle_edge_index']
                }
            res_gnn_part = self.particle_gnn_loss(gnn_out, cluster_label)
            res['particle_edge_loss'] = res_gnn_part['edge_loss']
            res['particle_node_loss'] = res_gnn_part['node_loss']
            res['particle_edge_accuracy'] = res_gnn_part['edge_accuracy']
            res['particle_node_accuracy'] = res_gnn_part['node_accuracy']

            accuracy += res_gnn_part['accuracy']
            loss += self.particle_gnn_weight*res_gnn_part['loss']

        if self.enable_gnn_inter:
            # Apply the GNN interaction grouping loss
            gnn_out = {}
            if 'inter_edge_pred' in out:
                gnn_out = {
                    'clusts':out['particles'],
                    'edge_pred':out['inter_edge_pred'],
                    'edge_index':out['inter_edge_index']
                }

            if 'inter_node_pred' in out: gnn_out.update({ 'node_pred': out['inter_node_pred'] })
            if 'node_pred_type' in out:  gnn_out.update({ 'node_pred_type': out['node_pred_type'] })
            if 'node_pred_p' in out:     gnn_out.update({ 'node_pred_p': out['node_pred_p'] })
            if 'node_pred_vtx' in out:   gnn_out.update({ 'node_pred_vtx': out['node_pred_vtx'] })
            res_gnn_inter = self.inter_gnn_loss(gnn_out, cluster_label, node_label=kinematics_label)
            res['inter_edge_loss'] = res_gnn_inter['loss']
            res['inter_edge_accuracy'] = res_gnn_inter['accuracy']
            if 'node_loss' in out:
                res['inter_node_loss'] = res_gnn_inter['node_loss']
                res['inter_node_accuracy'] = res_gnn_inter['node_accuracy']
            if 'type_loss' in res_gnn_inter:
                res['type_loss'] = res_gnn_inter['type_loss']
                res['type_accuracy'] = res_gnn_inter['type_accuracy']
            if 'p_loss' in res_gnn_inter:
                res['p_loss'] = res_gnn_inter['p_loss']
                res['p_accuracy'] = res_gnn_inter['p_accuracy']
            if 'vtx_position_loss' in res_gnn_inter:
                res['vtx_position_loss'] = res_gnn_inter['vtx_position_loss']
                res['vtx_score_loss'] = res_gnn_inter['vtx_score_loss']
                res['vtx_position_acc'] = res_gnn_inter['vtx_position_acc']
                res['vtx_score_acc'] = res_gnn_inter['vtx_score_acc']

            accuracy += res_gnn_inter['accuracy']
            loss += self.inter_gnn_weight*res_gnn_inter['loss']

        if self.enable_gnn_kinematics:
            # Loss on node predictions (type & momentum)
            gnn_out = {}
            if 'flow_edge_pred' in out:
                gnn_out = {
                    'clusts': out['kinematics_particles'],
                    'edge_pred': out['flow_edge_pred'],
                    'edge_index': out['kinematics_edge_index']
                }
            if 'node_pred_type' in out:
                gnn_out.update({ 'node_pred_type': out['node_pred_type'] })
            if 'node_pred_p' in out:
                gnn_out.update({ 'node_pred_p': out['node_pred_p'] })
            res_kinematics = self.kinematics_loss(gnn_out, kinematics_label, graph=particle_graph)

            #res['kinematics_loss'] = self.kinematics_p_weight * res_kinematics['p_loss'] + self.kinematics_type_weight * res_kinematics['type_loss'] #res_kinematics['loss']
            res['kinematics_loss'] = res_kinematics['node_loss']
            res['kinematics_accuracy'] = res_kinematics['accuracy']
            if 'type_loss' in res_kinematics:
                res['type_loss'] = res_kinematics['type_loss']
                res['type_accuracy'] = res_kinematics['type_accuracy']
            if 'p_loss' in res_kinematics:
                res['p_loss'] = res_kinematics['p_loss']
                res['p_accuracy'] = res_kinematics['p_accuracy']
            if 'type_loss' in res_kinematics or 'p_loss' in res_kinematics:
                res['kinematics_n_clusts_type'] = res_kinematics['n_clusts_type']
                res['kinematics_n_clusts_momentum'] = res_kinematics['n_clusts_momentum']
                res['kinematics_n_clusts_vtx'] = res_kinematics['n_clusts_vtx']
                res['kinematics_n_clusts_vtx_positives'] = res_kinematics['n_clusts_vtx_positives']

            accuracy += res_kinematics['node_accuracy']
            # Do not forget to take p_weight and type_weight into account (above)
            loss += self.kinematics_weight * res['kinematics_loss']

            # Loss on edge predictions (particle hierarchy)
            res['flow_loss'] = res_kinematics['edge_loss']
            res['flow_accuracy'] = res_kinematics['edge_accuracy']

            accuracy += res_kinematics['edge_accuracy']
            loss += self.flow_weight * res_kinematics['edge_loss']

        if self.enable_cosmic:
            gnn_out = {
                'clusts':out['interactions'],
                'node_pred':out['inter_cosmic_pred'],
            }

            res_cosmic = self.cosmic_loss(gnn_out, cluster_label)
            res['cosmic_loss'] = res_cosmic['loss']
            res['cosmic_accuracy'] = res_cosmic['accuracy']
            #res['cosmic_accuracy_cosmic'] = res_cosmic['cosmic_acc']
            #res['cosmic_accuracy_nu'] = res_cosmic['nu_acc']

            accuracy += res_cosmic['accuracy']
            loss += self.cosmic_weight * res_cosmic['loss']

        # Combine the results
        accuracy /= int(self.enable_uresnet) + int(self.enable_ppn) + int(self.enable_gnn_shower) \
                    + int(self.enable_gnn_inter) + int(self.enable_gnn_track) + int(self.enable_cnn_clust) \
                    + 2*int(self.enable_gnn_kinematics) + int(self.enable_cosmic) + int(self.enable_gnn_particle)

        res['loss'] = loss
        res['accuracy'] = accuracy

        if self.verbose:
            if self.enable_uresnet:
                print('Segmentation Accuracy: {:.4f}'.format(res_seg['accuracy']))
            if self.enable_ppn:
                print('PPN Accuracy: {:.4f}'.format(res_ppn['ppn_acc']))
            if self.enable_cnn_clust:
                if not self._enable_graph_spice:
                    print('Clustering Accuracy: {:.4f}'.format(res_cnn_clust['accuracy']))
                else:
                    print('Clustering Accuracy: {:.4f}'.format(res_graph_spice['accuracy']))
            if self.enable_gnn_shower:
                print('Shower fragment clustering accuracy: {:.4f}'.format(res_gnn_shower['edge_accuracy']))
                print('Shower primary prediction accuracy: {:.4f}'.format(res_gnn_shower['node_accuracy']))
            if self.enable_gnn_track:
                print('Track fragment clustering accuracy: {:.4f}'.format(res_gnn_track['edge_accuracy']))
            if self.enable_gnn_particle:
                print('Particle fragment clustering accuracy: {:.4f}'.format(res_gnn_part['edge_accuracy']))
                print('Particle primary prediction accuracy: {:.4f}'.format(res_gnn_part['node_accuracy']))
            if self.enable_gnn_inter:
                if 'node_accuracy' in res_gnn_inter: print('Particle ID accuracy: {:.4f}'.format(res_gnn_inter['node_accuracy']))
                print('Interaction grouping accuracy: {:.4f}'.format(res_gnn_inter['edge_accuracy']))
            if self.enable_gnn_kinematics:
                print('Flow accuracy: {:.4f}'.format(res_kinematics['edge_accuracy']))
            if 'node_pred_type' in out:
                print('Type accuracy: {:.4f}'.format(res['type_accuracy']))
            if 'node_pred_p' in out:
                print('Momentum accuracy: {:.4f}'.format(res['p_accuracy']))
            if 'node_pred_vtx' in out:
                #print('Vertex position accuracy: {:.4f}'.format(res['vtx_position_acc']))
                print('Primary score accuracy: {:.4f}'.format(res['vtx_score_acc']))
            if self.enable_cosmic:
                print('Cosmic discrimination accuracy: {:.4f}'.format(res_cosmic['accuracy']))
        return res

def setup_chain_cfg(self, cfg):
    """
    Prepare both FullChain and FullChainLoss

    Make sure config is logically sound with some basic checks
    """
    chain_cfg = cfg['chain']
    self.enable_ghost          = chain_cfg.get('enable_ghost', False)
    self.verbose               = chain_cfg.get('verbose', False)
    self.enable_uresnet        = chain_cfg.get('enable_uresnet', True)
    self.enable_ppn            = chain_cfg.get('enable_ppn', True)
    self.enable_dbscan         = chain_cfg.get('enable_dbscan', True)
    self.enable_cnn_clust      = chain_cfg.get('enable_cnn_clust', False)

    self.process_fragments     = chain_cfg.get('process_fragments', False)
    self.enable_gnn_shower     = chain_cfg.get('enable_gnn_shower', False)
    self.enable_gnn_track      = chain_cfg.get('enable_gnn_track', False)
    self.enable_gnn_particle   = chain_cfg.get('enable_gnn_particle', False)
    self.enable_gnn_inter      = chain_cfg.get('enable_gnn_inter', False)
    self.enable_gnn_kinematics = chain_cfg.get('enable_gnn_kinematics', False)
    self.enable_cosmic         = chain_cfg.get('enable_cosmic', False)

    if (self.enable_gnn_shower or \
        self.enable_gnn_track or \
        self.enable_gnn_particle or \
        self.enable_gnn_inter or \
        self.enable_gnn_kinematics or self.enable_cosmic):
        self.process_fragments = True

    if self.process_fragments:
        msg = """
        Fragment processing is turned ON. When training CNN models from
         scratch, we recommend turning fragment processing OFF as without
         reliable segmentation and/or cnn clustering outputs this could take
         prohibitively large training iterations.
        """
        print(msg)

    # If fragment processing is turned off, no inputs to GNN
    if not self.process_fragments:
        self.enable_gnn_shower     = False
        self.enable_gnn_track      = False
        self.enable_gnn_particle   = False
        self.enable_gnn_inter      = False
        self.enable_gnn_kinematics = False
        self.enable_cosmic         = False

    # Whether to use PPN information (GNN shower clustering step only)
    self.use_ppn_in_gnn    = chain_cfg.get('use_ppn_in_gnn', False)

    # Make sure the deghosting config is consistent
    if self.enable_ghost:
        assert cfg['uresnet_ppn']['uresnet_lonely']['ghost']
        if self.enable_ppn:
            assert cfg['uresnet_ppn']['ppn']['downsample_ghost']

    # Enforce basic logical order
    # 1. Need semantics for everything
    assert self.enable_uresnet
    # 2. If PPN is used in GNN, need PPN
    assert self.enable_ppn or (not self.use_ppn_in_gnn)
    # 3. Need at least one of two dense clusterer
    assert self.enable_dbscan or self.enable_cnn_clust
    # 4. Check that SPICE and DBSCAN are not redundant
    if self.enable_cnn_clust and self.enable_dbscan:
        if 'spice' in cfg:
            assert not (np.array(cfg['spice']['fragment_clustering']['cluster_classes']) == \
                        np.array(np.array(cfg['fragment_manager']['cluster_classes'])).reshape(-1)).any()
        else:
            assert 'graph_spice' in cfg
            assert set(cfg['fragment_manager']['cluster_classes']).issubset(
                set(cfg['graph_spice']['skip_classes']))

    if self.enable_gnn_particle: # If particle fragment GNN is used, make sure it is not redundant
        if self.enable_gnn_shower:
            assert cfg['grappa_shower']['base']['node_type'] \
                not in cfg['grappa_particle']['base']['node_type']

        if self.enable_gnn_track:
            assert cfg['grappa_track']['base']['node_type'] \
                not in cfg['grappa_particle']['base']['node_type']

    if self.enable_cosmic: assert self.enable_gnn_inter # Cosmic classification needs interaction clustering
