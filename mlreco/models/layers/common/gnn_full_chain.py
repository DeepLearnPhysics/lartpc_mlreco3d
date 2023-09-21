import torch
import numpy as np
from warnings import warn

from mlreco.models.grappa import GNN, GNNLoss
from mlreco.utils.globals import SHAPE_COL, TRACK_SHP
from mlreco.utils.unwrap import prefix_unwrapper_rules
from mlreco.utils.gnn.evaluation import (node_assignment_score,
                                         primary_assignment)
from mlreco.utils.gnn.cluster import (form_clusters,
                                      get_cluster_batch,
                                      get_cluster_label)

class FullChainGNN(torch.nn.Module):
    """
    GNN section of the full chain.

    See Also
    --------
    mlreco.models.full_chain.FullChain, FullChainLoss
    """
    MODULES = ['grappa_shower', 'grappa_track', 'grappa_inter',
               'grappa_shower_loss', 'grappa_track_loss', 'grappa_inter_loss',
               'full_chain_loss', 'spice', 'spice_loss',
               'fragment_clustering',  'chain', 'dbscan',
               ('uresnet_ppn', ['uresnet_lonely', 'ppn'])]

    def __init__(self, cfg):
        super(FullChainGNN, self).__init__()

        # Configure the chain first
        setup_chain_cfg(self, cfg)

        # Initialize the particle aggregator modules
        for stage in ['shower', 'track', 'particle', 'inter', 'kinematics']:
            if getattr(self, f'enable_gnn_{stage}'):
                # Initialize the GNN model
                name = f'grappa_{stage}'
                setattr(self, name, GNN(cfg, name=name, batch_col=self.batch_col, coords_col=self.coords_col))

                # Get the relevant attributes
                grappa_cfg = cfg.get(name, {})
                setattr(self, f'_{stage}_use_true_particles', grappa_cfg.get('use_true_particles', False))
                setattr(self, f'_{stage}_ids', getattr(self, name).node_type)
                setattr(self, f'_{stage}_add_value', getattr(getattr(self, name).node_encoder, 'add_value', False))
                setattr(self, f'_{stage}_add_shape', getattr(getattr(self, name).node_encoder, 'add_shape', False))
                if hasattr(getattr(self, name).node_encoder, 'add_value'):
                    setattr(getattr(self, name).node_encoder, 'add_value', False)
                    setattr(getattr(self, name).node_encoder, 'add_shape', False)
                if self.use_supp_in_gnn:
                    warn('`use_supp_in_gnn` is deprecated, '
                            'specify the extra features in the node encoder config')
                    setattr(self, f'_{stage}_add_value', True)
                    setattr(self, f'_{stage}_add_shape', True)

                # Interaction specific attributes
                if stage == 'inter':
                    self.inter_source_col = cfg.get('grappa_inter_loss', {}).get('edge_loss', {}).get('source_col', 6)
                    self._inter_use_shower_primary      = grappa_cfg.get('use_shower_primary', True)

                # Add unwrapping rules
                suffix = '_fragment' if stage not in ['inter','kinematics'] else ''
                tag    = f'{stage}{suffix}' if stage != 'inter' else 'particle'
                self.RETURNS.update(prefix_unwrapper_rules(getattr(self, name).RETURNS, tag))
                self.RETURNS[f'{tag}_clusts'][1][0] = 'input_data' if not self.enable_ghost else 'input_rescaled'


    def run_gnn(self, grappa, input, result, clusts, prefix, kwargs={}):
        """
        Generic function to group in one place the common code to run a GNN model.

        Parameters
        ==========
        - grappa: GrapPA module to run
        - input: input data
        - result: dictionary
        - clusts: list of list of indices (indexing input data)
        - prefix: prefix to append at the front of the output
        - kwargs: extra arguments to pass to the gnn

        Returns
        =======
        None (modifies the result dict in place)
        """
        # Figure out the expected output keys
        labels = {k:f'{prefix}_{k}' for k in grappa.RETURNS.keys()}
        labels['group_pred'] = f'{prefix}_group_pred'

        # Pass data through the GrapPA model
        gnn_output = grappa(input, clusts, batch_size=self.batch_size, **kwargs)

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


    def get_all_fragments(self, result, input):
        """
        Given geometric or CNN clustering results and (optional) true
        fragment labels, return true or predicted fragments
        """

        if self.use_true_fragments:
            label_clustering = result['cluster_label_adapted'][0]
            fragments = form_clusters(label_clustering[0].int().cpu().numpy(),
                                      column=5)

            fragments = np.array(fragments, dtype=object)
            frag_seg = get_cluster_label(label_clustering[0].int(),
                                         fragments,
                                         column=-1)
            semantic_labels = label_clustering[0].int()[:, -1]
            frag_batch_ids = get_cluster_batch(input[0][:, :5], fragments)
        else:
            fragments = result['frag_dict']['frags'][0]
            frag_seg = result['frag_dict']['frag_seg'][0]
            frag_batch_ids = result['frag_dict']['frag_batch_ids'][0]
            semantic_labels = result['segment_label_tmp'][0]

        frag_dict = {
            'frags': fragments,
            'frag_seg': frag_seg,
            'frag_batch_ids': frag_batch_ids,
            'segment_label_tmp': semantic_labels
        }

        # Since <vids> and <counts> depend on the batch column of the input
        # tensor, they are shared between the two settings.
        frag_dict['vids'] = result['frag_dict']['vids'][0]
        frag_dict['counts'] = result['frag_dict']['counts'][0]

        return frag_dict


    def run_fragment_gnns(self, result, input):
        """
        Run all fragment-level GNN models.

            1. Shower GNN
            2. Track GNN
            3. Particle GNN (optional?)
        """

        frag_dict = self.get_all_fragments(result, input)
        fragments = frag_dict['frags']
        frag_seg = frag_dict['frag_seg']

        if self.enable_gnn_shower:

            # Run shower GrapPA: merges shower fragments into shower instances
            em_mask, kwargs = self.get_extra_gnn_features(input, result,
                    fragments, frag_seg, self._shower_ids,
                    add_points=self.use_ppn_in_gnn,
                    add_value=self._shower_add_value,
                    add_shape=self._shower_add_shape)

            self.run_gnn(self.grappa_shower,
                         input,
                         result,
                         fragments[em_mask],
                         'shower_fragment',
                         kwargs)

        if self.enable_gnn_track:

            # Run track GrapPA: merges tracks fragments into track instances
            track_mask, kwargs = self.get_extra_gnn_features(input, result,
                    fragments, frag_seg, self._track_ids,
                    add_points=self.use_ppn_in_gnn,
                    add_value=self._track_add_value,
                    add_shape=self._track_add_shape)

            self.run_gnn(self.grappa_track,
                         input,
                         result,
                         fragments[track_mask],
                         'track_fragment',
                         kwargs)

        if self.enable_gnn_particle:
            # Run particle GrapPA: merges particle fragments or
            # labels in _partile_ids together into particle instances
            mask, kwargs = self.get_extra_gnn_features(input, result,
                    fragments, frag_seg, self._particle_ids,
                    add_points=self.use_ppn_in_gnn,
                    add_value=self._particle_add_value,
                    add_shape=self._particle_add_shape)

            kwargs['groups'] = frag_seg[mask]

            self.run_gnn(self.grappa_particle,
                         input,
                         result,
                         fragments[mask],
                         'particle_fragment',
                         kwargs)

        return frag_dict


    def get_all_particles(self, frag_result, result, input):

        fragments = frag_result['frags']
        frag_seg = frag_result['frag_seg']
        frag_batch_ids = frag_result['frag_batch_ids']
        semantic_labels = frag_result['segment_label_tmp']

        # for i, c in enumerate(fragments):
        #     print('format' , torch.unique(input[0][c, self.batch_col], return_counts=True))

        vids = frag_result['vids']
        counts = frag_result['counts']

        # Merge fragments into particle instances, retain primary fragment id of showers
        particles, part_primary_ids = [], []
        # It is possible that len(counts) > len(np.unique(frag_batch_ids))
        #assert len(counts) == len(np.unique(frag_batch_ids))
        # Can happen e.g. if an event has no shower fragments
        for b in range(len(counts)):
            mask = (frag_batch_ids == b)
            # Append one particle per particle group
            # To use true group predictions, change use_group_pred to True
            # in each grappa config.
            if self.enable_gnn_particle:
                self.select_particle_in_group(result, counts, b, particles,
                                            part_primary_ids,
                                            'particle_fragment_node_pred',
                                            'particle_fragment_group_pred',
                                            'particle_fragment_clusts')

                for c in self._particle_ids:
                    mask &= (frag_seg != c)
            # Append one particle per shower group
            if self.enable_gnn_shower:
                self.select_particle_in_group(result, counts, b, particles,
                                            part_primary_ids,
                                            'shower_fragment_node_pred',
                                            'shower_fragment_group_pred',
                                            'shower_fragment_clusts')

                for c in self._shower_ids:
                    mask &= (frag_seg != c)
            # Append one particle 'particle' track group
            if self.enable_gnn_track:
                self.select_particle_in_group(result, counts, b, particles,
                                            part_primary_ids,
                                            'track_fragment_node_pred',
                                            'track_fragment_group_pred',
                                            'track_fragment_clusts')

                for c in self._track_ids:
                    mask &= (frag_seg != c)

            # Append one particle per fragment that is not already accounted for
            particles.extend(fragments[mask])
            part_primary_ids.extend(-np.ones(np.sum(mask)).astype(int))

        particles_np    = np.empty(len(particles), dtype=object)
        particles_np[:] = particles

        part_batch_ids = get_cluster_batch(input[0], particles_np)
        part_primary_ids = np.array(part_primary_ids, dtype=np.int32)

        # Get the particle shape. If it's a shower, pick the shape of the primary
        part_seg = get_cluster_label(semantic_labels[:,None], particles, column=SHAPE_COL)
        for i, p in enumerate(particles):
            if part_seg[i] != TRACK_SHP and self._inter_use_shower_primary:
                voxel_inds = counts[:part_batch_ids[i]].sum().item() + \
                             np.arange(counts[part_batch_ids[i]].item())
                if len(voxel_inds) and len(result['shower_fragment_clusts'][0][part_batch_ids[i]]) > 0:
                    p = voxel_inds[result['shower_fragment_clusts'][0]\
                                  [part_batch_ids[i]][part_primary_ids[i]]]
                    part_seg[i] = get_cluster_label(semantic_labels[:,None], [p], column=SHAPE_COL)[0]

        # Store in result the intermediate fragments
        bcids = [np.where(part_batch_ids == b)[0] for b in range(len(counts))]
        parts = [np.empty(len(b), dtype=object) for b in bcids]
        for idx, b in enumerate(bcids):
            parts[idx][:] = [vids[c] for c in particles_np[b]]

        parts_seg = [part_seg[b] for idx, b in enumerate(bcids)]

        result.update({
            'particle_clusts': [parts],
            'particle_seg': [parts_seg],
            'particle_batch_ids': [part_batch_ids],
        })

        part_result = {
            'particles': particles_np,
            'part_seg': part_seg,
            'part_batch_ids': part_batch_ids,
            'part_primary_ids': part_primary_ids,
            'counts': counts
        }

        return part_result


    def run_particle_gnns(self, result, input, frag_result):

        part_result = self.get_all_particles(frag_result, result, input)

        particles = part_result['particles']
        part_seg = part_result['part_seg']
        part_batch_ids = part_result['part_batch_ids']
        part_primary_ids = part_result['part_primary_ids']
        counts = part_result['counts']

        label_clustering = result['cluster_label_adapted'][0] if 'cluster_label_adapted' in result else None
        if label_clustering is None and (self.use_true_fragments or (self.enable_cosmic and self._cosmic_use_true_interactions)):
            raise Exception('Need clustering labels to use true fragments or true interactions.')

        device = input[0].device

        if self.enable_gnn_inter:
            if self._inter_use_true_particles:
                #label_clustering = [label_clustering[0].cpu().numpy()]
                particles = form_clusters(label_clustering[0].int().cpu().numpy(), min_size=-1, column=self.inter_source_col, cluster_classes=self._inter_ids)
                particles = np.array(particles, dtype=object)
                part_seg = get_cluster_label(label_clustering[0].int(), particles, column=-1)
                part_batch_ids = get_cluster_batch(label_clustering[0], particles)
                _, counts = torch.unique(label_clustering[0][:, 0], return_counts=True)

            # For showers, select primary for extra feature extraction
            extra_feats_particles = []
            for i, p in enumerate(particles):
                if part_seg[i] != TRACK_SHP and not self._inter_use_true_particles and self._inter_use_shower_primary:
                    voxel_inds = counts[:part_batch_ids[i]].sum().item() + \
                                 np.arange(counts[part_batch_ids[i]].item())
                    if len(voxel_inds) and len(result['shower_fragment_clusts'][0][part_batch_ids[i]]) > 0:
                        try:
                            p = voxel_inds[result['shower_fragment_clusts'][0]\
                                          [part_batch_ids[i]][part_primary_ids[i]]]
                        except IndexError as e:
                            print(len(result['shower_fragment_clusts'][0]))
                            print([part_batch_ids[i]])
                            print(part_primary_ids[i])
                            print(len(voxel_inds))
                            print(result['shower_fragment_clusts'][0][part_batch_ids[i]][part_primary_ids[i]])
                            raise e

                extra_feats_particles.append(p)

            # result['extra_feats_particles'] = [extra_feats_particles]
            extra_feats_particles_np    = np.empty(len(extra_feats_particles), dtype=object)
            extra_feats_particles_np[:] = extra_feats_particles

            # Run interaction GrapPA: merges particle instances into interactions
            inter_mask, kwargs = self.get_extra_gnn_features(input, result,
                    extra_feats_particles_np, part_seg, self._inter_ids,
                    add_points=self.use_ppn_in_gnn,
                    add_value=self._inter_add_value,
                    add_shape=self._inter_add_shape)

            self.run_gnn(self.grappa_inter,
                         input,
                         result,
                         particles[inter_mask],
                         'particle',
                         kwargs)

        # ---
        # 4. GNN for particle flow & kinematics
        # ---

        if self.enable_gnn_kinematics:
            if not self.enable_gnn_inter:
                raise Exception("Need interaction clustering before kinematic GNN.")

            self.run_gnn(self.grappa_kinematics,
                         input,
                         result,
                         particles[inter_mask],
                         'kinematics')

        # ---
        # 5. CNN for interaction classification
        # ---

        if self.enable_cosmic:
            if not self.enable_gnn_inter and not self._cosmic_use_true_interactions:
                raise Exception("Need interaction clustering before cosmic discrimination.")

            _, counts = torch.unique(input[0][:, self.batch_col], return_counts=True)
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
                                                  None, 'particle_group_pred', 'particle_clusts')

            interactions_np    = np.empty(len(interations), dtype=object)
            interactions_np[:] = interactions

            inter_batch_ids = get_cluster_batch(input[0], interactions_np)
            inter_cosmic_pred = torch.empty((len(interactions_np), 2), dtype=torch.float)

            # Replace batch id column with a global "interaction id"
            # because ResidualEncoder uses the batch id column to shape its output
            if 'ppn_feature_dec' in result:
                feature_map = result['ppn_feature_dec'][0][-1]
            else:
                feature_map = result['ppn_layers'][0][-1]
            if not torch.is_tensor(feature_map):
                feature_map = feature_map.features

            inter_input_data = input[0].float() if self._cosmic_use_input_data \
                                                else torch.cat([input[0][:, :4].float(), feature_map], dim=1)

            inter_data = torch.empty((0, inter_input_data.size(1)), dtype=torch.float, device=device)
            for i, interaction in enumerate(interactions_np):
                inter_data = torch.cat([inter_data, inter_input_data[interaction]], dim=0)
                inter_data[-len(interaction):, self.batch_col] = i * torch.ones(len(interaction)).to(device)
            inter_cosmic_pred = self.cosmic_discriminator(inter_data)

            # Reorganize into batches before storing in result dictionary
            inter_batch_ids = np.array(inter_batch_ids)

            batches, counts = torch.unique(input[0][:, self.batch_col], return_counts=True)
            # In case one of the events is "missing" and len(counts) < batch_size
            if len(counts) < self.batch_size:
                new_counts = torch.zeros(self.batch_size, dtype=torch.int64, device=counts.device)
                new_counts[batches] = counts
                counts = new_counts

            vids = np.concatenate([np.arange(n.item()) for n in counts])
            bcids = [np.where(inter_batch_ids == b)[0] for b in range(len(counts))]

            inters = [np.empty(len(b), dtype=object) for b in enumerate(bcids)]
            for idx, b in enumeate(bcids):
                inters[idx][:] = [vids[c].astype(np.int64) for c in interactions_nb[b]]

            inter_cosmic_pred_np = [inter_cosmic_pred[b] for idx, b in enumerate(bcids)]

            result.update({
                'interactions': [inters],
                'inter_cosmic_pred': [inter_cosmic_pred_np]
                })


    def full_chain_gnn(self, result, input):

        frag_dict = self.run_fragment_gnns(result, input)
        self.run_particle_gnns(result, input, frag_dict)

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

        result, input = self.full_chain_cnn(input)
        if len(input[0]) and 'frag_dict' in result and self.process_fragments and (self.enable_gnn_track or self.enable_gnn_shower or self.enable_gnn_inter or self.enable_gnn_particle):
            result = self.full_chain_gnn(result, input)
        if 'frag_dict' in result:
            del result['frag_dict']

        return result


class FullChainLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain

    See Also
    --------
    mlreco.models.full_chain.FullChainLoss, FullChainGNN
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(FullChainLoss, self).__init__()

        # Configure the chain first
        setup_chain_cfg(self, cfg, False)

        if self.enable_gnn_shower:
            self.shower_gnn_loss         = GNNLoss(cfg, 'grappa_shower_loss', batch_col=self.batch_col, coords_col=self.coords_col)
        if self.enable_gnn_track:
            self.track_gnn_loss          = GNNLoss(cfg, 'grappa_track_loss', batch_col=self.batch_col, coords_col=self.coords_col)
        if self.enable_gnn_particle:
            self.particle_gnn_loss       = GNNLoss(cfg, 'grappa_particle_loss', batch_col=self.batch_col, coords_col=self.coords_col)
        if self.enable_gnn_inter:
            self.inter_gnn_loss          = GNNLoss(cfg, 'grappa_inter_loss', batch_col=self.batch_col, coords_col=self.coords_col)
        if self.enable_gnn_kinematics:
            self.kinematics_loss         = GNNLoss(cfg, 'grappa_kinematics_loss', batch_col=self.batch_col, coords_col=self.coords_col)
        if self.enable_cosmic:
            self.cosmic_loss             = GNNLoss(cfg, 'cosmic_loss', batch_col=self.batch_col, coords_col=self.coords_col)

        # Initialize the loss weights
        self.loss_config = cfg.get('full_chain_loss', {})

        self.deghost_weight         = self.loss_config.get('deghost_weight', 1.0)
        self.segmentation_weight    = self.loss_config.get('segmentation_weight', 1.0)
        self.ppn_weight             = self.loss_config.get('ppn_weight', 1.0)
        self.cnn_clust_weight       = self.loss_config.get('cnn_clust_weight', 1.0)
        self.shower_gnn_weight      = self.loss_config.get('shower_gnn_weight', 1.0)
        self.track_gnn_weight       = self.loss_config.get('track_gnn_weight', 1.0)
        self.particle_gnn_weight    = self.loss_config.get('particle_gnn_weight', 1.0)
        self.inter_gnn_weight       = self.loss_config.get('inter_gnn_weight', 1.0)
        self.kinematics_weight      = self.loss_config.get('kinematics_weight', 1.0)
        self.flow_weight            = self.loss_config.get('flow_weight', 1.0)
        self.kinematics_p_weight    = self.loss_config.get('kinematics_p_weight', 1.0)
        self.kinematics_type_weight = self.loss_config.get('kinematics_type_weight', 1.0)
        self.cosmic_weight          = self.loss_config.get('cosmic_weight', 1.0)

    def forward(self, out, seg_label, ppn_label=None, cluster_label=None, kinematics_label=None,
                particle_graph=None, iteration=None):
        res = {}
        accuracy, loss = 0., 0.
        if kinematics_label is not None:
            from warnings import warn
            warn('kinematics_label is no longer needed, remove it from the config', DeprecationWarning, stacklevel=2)

        if self.enable_charge_rescaling:
            ghost_label = torch.cat((seg_label[0][:,:4], (seg_label[0][:,-1] == 5).type(seg_label[0].dtype).reshape(-1,1)), dim=-1)
            res_deghost = self.deghost_loss({'segmentation':out['ghost']}, [ghost_label])
            for key in res_deghost:
                res['deghost_' + key] = res_deghost[key]
            accuracy += res_deghost['accuracy']
            loss += self.deghost_weight*res_deghost['loss']
            deghost = out['ghost'][0][:,0] > out['ghost'][0][:,1]

        if self.enable_uresnet and 'segmentation' in out:
            if not self.enable_charge_rescaling:
                res_seg = self.uresnet_loss(out, seg_label)
            else:
                true_deghost = seg_label[0][deghost,-1] < 5 # Do not apply loss on true ghosts classified as non-ghosts
                res_seg = self.uresnet_loss({'segmentation':[out['segmentation'][0][true_deghost]]}, [seg_label[0][deghost][true_deghost]])
            for key in res_seg:
                res['segmentation_' + key] = res_seg[key]
            accuracy += res_seg['accuracy']
            loss += self.segmentation_weight*res_seg['loss']
            #print('uresnet ', self.segmentation_weight, res_seg['loss'], loss)

        if self.enable_ppn and 'ppn_output_coords' in out:
            # Apply the PPN loss
            res_ppn = self.ppn_loss(out, seg_label, ppn_label)
            for key in res_ppn:
                res['ppn_' + key] = res_ppn[key]

            accuracy += res_ppn['accuracy']
            loss += self.ppn_weight*res_ppn['loss']

        # Fetch adapted labels
        if cluster_label is not None:
            cluster_label = out['cluster_label_adapted']

        if self.enable_ghost and 'ghost' in out \
                             and (self.enable_cnn_clust or \
                                  self.enable_gnn_track or \
                                  self.enable_gnn_shower or \
                                  self.enable_gnn_inter or \
                                  self.enable_gnn_kinematics or \
                                  self.enable_cosmic):

            deghost = out['ghost'][0].argmax(dim=1) == 0

            if self.cheat_ghost:
                true_mask = deghost
            else:
                true_mask = None

            segment_label = seg_label[0][deghost][:, -1]
            seg_label = seg_label[0][deghost]
        else:
            segment_label = seg_label[0][:, -1]
            seg_label = seg_label[0]

        if self.enable_cnn_clust:
            # If there is no track voxel, maybe GraphSpice didn't run
            if self._enable_graph_spice and 'graph_spice_graph_id' in out:
            # if self._enable_graph_spice and 'graph_spice_graph_info' in out:
                graph_spice_out = {k.split('graph_spice_')[-1]:v for k, v in out.items() if 'graph_spice_' in k}

                segmentation_pred = out['segmentation'][0]

                if self._gspice_use_true_labels:
                    gs_seg_label = torch.cat([cluster_label[0][:, :4], segment_label[:, None]], dim=1)
                else:
                    gs_seg_label = torch.cat([cluster_label[0][:, :4], torch.argmax(segmentation_pred, dim=1)[:, None]], dim=1)
                #gs_seg_label = torch.cat([cluster_label[0][:, :4], segment_label[:, None]], dim=1)

                # NOTE: We need to limit loss computation to voxels that are
                # in the intersection of truth and prediction.
                # Setting seg label to -1 does not work (embeddings already
                # have a shape based on predicted semantics). Instead we set
                # the cluster label to -1 and the GraphSPICEEmbeddingLoss
                # will remove voxels with true cluster label -1.
                gs_cluster_label = cluster_label[0]
                #if not self._gspice_use_true_labels:
                #    gs_cluster_label[(gs_cluster_label[:, -1] != torch.argmax(segmentation_pred, dim=1)), 5] = -1
                #res['gs_cluster_label'] = [gs_cluster_label]
                res_graph_spice = self.spatial_embeddings_loss(graph_spice_out, [gs_seg_label], [gs_cluster_label])
                if 'accuracy' in res_graph_spice:
                    accuracy += res_graph_spice['accuracy']
                loss += self.cnn_clust_weight * res_graph_spice['loss']
                for key in res_graph_spice:
                    res['graph_spice_' + key] = res_graph_spice[key]
            elif 'embeddings' in out:
                # Apply the CNN dense clustering loss to HE voxels only
                he_mask = segment_label < 4
                # sem_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,-1].view(-1,1)), dim=1)]
                #clust_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,5].view(-1,1),cluster_label[0][he_mask,4].view(-1,1)), dim=1)]
                clust_label = [cluster_label[0][he_mask].clone()]
                cnn_clust_output = {'embeddings':[out['embeddings'][0][he_mask]], 'seediness':[out['seediness'][0][he_mask]], 'margins':[out['margins'][0][he_mask]]}
                #cluster_label[0] = cluster_label[0][he_mask]
                # FIXME does this suppose that clust_label has same ordering as embeddings?
                res_cnn_clust = self.spatial_embeddings_loss(cnn_clust_output, clust_label)
                for key in res_cnn_clust:
                    res['cnn_clust_' + key] = res_cnn_clust[key]

                accuracy += res_cnn_clust['accuracy']
                loss += self.cnn_clust_weight*res_cnn_clust['loss']

        if self.enable_gnn_shower:
            # Apply the GNN shower clustering loss
            gnn_out = {}
            if 'shower_fragment_edge_pred' in out:
                gnn_out = {
                    'clusts':out['shower_fragment_clusts'],
                    'node_pred':out['shower_fragment_node_pred'],
                    'edge_pred':out['shower_fragment_edge_pred'],
                    'edge_index':out['shower_fragment_edge_index']
                }
            res_gnn_shower = self.shower_gnn_loss(gnn_out, cluster_label)
            for key in res_gnn_shower:
                res['grappa_shower_' + key] = res_gnn_shower[key]

            accuracy += res_gnn_shower['accuracy']
            loss += self.shower_gnn_weight*res_gnn_shower['loss']

        if self.enable_gnn_track:
            # Apply the GNN track clustering loss
            gnn_out = {}
            if 'track_fragment_edge_pred' in out:
                gnn_out = {
                    'clusts':out['track_fragment_clusts'],
                    'edge_pred':out['track_fragment_edge_pred'],
                    'edge_index':out['track_fragment_edge_index']
                }
            res_gnn_track = self.track_gnn_loss(gnn_out, cluster_label)
            for key in res_gnn_track:
                res['grappa_track_' + key] = res_gnn_track[key]
            accuracy += res_gnn_track['accuracy']
            loss += self.track_gnn_weight*res_gnn_track['loss']

        if self.enable_gnn_particle:
            # Apply the GNN particle clustering loss
            gnn_out = {}
            if 'particle_fragment_edge_pred' in out:
                gnn_out = {
                    'clusts':out['particle_fragment_clusts'],
                    'node_pred':out['particle_fragment_node_pred'],
                    'edge_pred':out['particle_fragment_edge_pred'],
                    'edge_index':out['particle_fragment_edge_index']
                }
            res_gnn_part = self.particle_gnn_loss(gnn_out, cluster_label)
            for key in res_gnn_particle:
                res['grappa_particle_' + key] = res_gnn_particle[key]

            accuracy += res_gnn_part['accuracy']
            loss += self.particle_gnn_weight*res_gnn_part['loss']

        if self.enable_gnn_inter:
            # Apply the GNN interaction grouping loss
            gnn_out = {}
            if 'particle_edge_pred' in out:
                gnn_out = {
                    'clusts':out['particle_clusts'],
                    'edge_pred':out['particle_edge_pred'],
                    'edge_index':out['particle_edge_index']
                }
            if 'particle_node_pred' in out: gnn_out.update({ 'node_pred': out['particle_node_pred'] })
            if 'particle_node_pred_type' in out:  gnn_out.update({ 'node_pred_type': out['particle_node_pred_type'] })
            if 'particle_node_pred_p' in out:     gnn_out.update({ 'node_pred_p': out['particle_node_pred_p'] })
            if 'particle_node_pred_vtx' in out:   gnn_out.update({ 'node_pred_vtx': out['particle_node_pred_vtx'] })
            if 'particle_node_features' in out:   gnn_out.update({ 'node_features': out['particle_node_features'] })
            if 'particle_edge_features' in out:   gnn_out.update({ 'edge_features': out['particle_edge_features'] })

            res_gnn_inter = self.inter_gnn_loss(gnn_out, cluster_label, node_label=cluster_label, graph=particle_graph, iteration=iteration)
            for key in res_gnn_inter:
                res['grappa_inter_' + key] = res_gnn_inter[key]

            accuracy += res_gnn_inter['accuracy']
            loss += self.inter_gnn_weight*res_gnn_inter['loss']

        if self.enable_gnn_kinematics:
            # Loss on node predictions (type & momentum)
            gnn_out = {}
            if 'kinematics_edge_pred' in out:
                gnn_out = {
                    'clusts': out['kinematics_particles'],
                    'edge_pred': out['kinematics_edge_pred'],
                    'edge_index': out['kinematics_edge_index']
                }
            if 'kinematics_node_pred_type' in out:
                gnn_out.update({ 'node_pred_type': out['kinematics_node_pred_type'] })
            if 'kinematics_node_pred_p' in out:
                gnn_out.update({ 'node_pred_p': out['kinematics_node_pred_p'] })
            res_kinematics = self.kinematics_loss(gnn_out, cluster_label, graph=particle_graph)
            for key in res_kinematics:
                res['grappa_kinematics_' + key] = res_kinematics[key]

            accuracy += res_kinematics['node_accuracy']
            # Do not forget to take p_weight and type_weight into account (above)
            loss += self.kinematics_weight * res['grappa_kinematics_loss']

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
            for key in res_cosmic:
                res['cosmic_' + key] = res_cosmic[key]

            accuracy += res_cosmic['accuracy']
            loss += self.cosmic_weight * res_cosmic['loss']

        # Combine the results
        accuracy /= int(self.enable_charge_rescaling) + int(self.enable_uresnet) + int(self.enable_ppn) + int(self.enable_gnn_shower) \
                    + int(self.enable_gnn_inter) + int(self.enable_gnn_track) + int(self.enable_cnn_clust) \
                    + 2*int(self.enable_gnn_kinematics) + int(self.enable_cosmic) + int(self.enable_gnn_particle)

        res['loss'] = loss
        res['accuracy'] = accuracy
        #print('Loss = ', res['loss'])

        if self.verbose:
            if self.enable_charge_rescaling:
                print('Deghosting Accuracy: {:.4f}'.format(res_deghost['accuracy']))
            if self.enable_uresnet and 'segmentation' in out:
                print('Segmentation Accuracy: {:.4f}'.format(res_seg['accuracy']))
            if self.enable_ppn and 'ppn_output_coords' in out:
                print('PPN Accuracy: {:.4f}'.format(res_ppn['accuracy']))
            # if self.enable_cnn_clust and ('graph_spice_graph_info' in out or 'embeddings' in out):
            if self.enable_cnn_clust and 'graph_spice_graph_id' in out:
                if not self._enable_graph_spice:
                    print('Clustering Embedding Accuracy: {:.4f}'.format(res_cnn_clust['accuracy']))
                else:
                    print('Clustering Accuracy: {:.4f}'.format(res_graph_spice['accuracy']))
                    if 'edge_accuracy' in res_graph_spice:
                        print('Clustering Edge Accuracy: {:.4f}'.format(res_graph_spice['edge_accuracy']))
            if self.enable_gnn_shower:
                print('Shower fragment clustering accuracy: {:.4f}'.format(res_gnn_shower['edge_accuracy']))
                print('Shower primary prediction accuracy: {:.4f}'.format(res_gnn_shower['node_accuracy']))
            if self.enable_gnn_track:
                print('Track fragment clustering accuracy: {:.4f}'.format(res_gnn_track['edge_accuracy']))
            if self.enable_gnn_particle:
                print('Particle fragment clustering accuracy: {:.4f}'.format(res_gnn_part['edge_accuracy']))
                print('Particle primary prediction accuracy: {:.4f}'.format(res_gnn_part['node_accuracy']))
            if self.enable_gnn_inter:
                #if 'node_accuracy' in res_gnn_inter: print('Particle ID accuracy: {:.4f}'.format(res_gnn_inter['node_accuracy']))
                print('Interaction grouping accuracy: {:.4f}'.format(res_gnn_inter['edge_accuracy']))
            if self.enable_gnn_kinematics:
                print('Flow accuracy: {:.4f}'.format(res_kinematics['edge_accuracy']))
            if 'particle_node_pred_type' in out:
                if 'grappa_inter_type_accuracy' in res:
                    print('Particle ID accuracy: {:.4f}'.format(res['grappa_inter_type_accuracy']))
                elif 'grappa_kinematics_type_accuracy' in res:
                    print('Particle ID accuracy: {:.4f}'.format(res['grappa_kinematics_type_accuracy']))
            if 'particle_node_pred_p' in out:
                if 'grappa_inter_p_accuracy' in res:
                    print('Momentum accuracy: {:.4f}'.format(res['grappa_inter_p_accuracy']))
                elif 'grappa_kinematics_p_accuracy' in res:
                    print('Momentum accuracy: {:.4f}'.format(res['grappa_kinematics_p_accuracy']))
            if 'particle_node_pred_vtx' in out:
                if 'grappa_inter_vtx_score_accuracy' in res:
                    print('Primary particle score accuracy: {:.4f}'.format(res['grappa_inter_vtx_score_accuracy']))
                elif 'grappa_kinematics_vtx_score_accuracy' in res:
                    print('Primary particle score accuracy: {:.4f}'.format(res['grappa_kinematics_vtx_score_accuracy']))
            if self.enable_cosmic:
                print('Cosmic discrimination accuracy: {:.4f}'.format(res_cosmic['accuracy']))

        return res


def setup_chain_cfg(self, cfg, print_info=True):
    """
    Prepare both FullChain and FullChainLoss

    Make sure config is logically sound with some basic checks

    See Also
    --------
    mlreco.models.full_chain.FullChain, FullChainGNN
    """
    chain_cfg = cfg.get('chain', {})

    self.use_me                = chain_cfg.get('use_mink', True)
    self.batch_col             = 0 if self.use_me else 3
    self.coords_col            = (1, 4) if self.use_me else (0, 3)
    self.batch_size            = None # To be set at forward time

    self.process_fragments     = chain_cfg.get('process_fragments', False)
    self.use_true_fragments    = chain_cfg.get('use_true_fragments', False)
    self.use_true_particles    = chain_cfg.get('use_true_particles', False)
    self._gspice_use_true_labels      = cfg.get('graph_spice', {}).get('use_true_labels', False)

    self.enable_charge_rescaling = chain_cfg.get('enable_charge_rescaling', False)
    self.collection_charge_only = chain_cfg.get('collection_charge_only', False)
    self.enable_ghost          = chain_cfg.get('enable_ghost', False)
    self.cheat_ghost           = chain_cfg.get('cheat_ghost', False)
    self.verbose               = chain_cfg.get('verbose', False)
    self.enable_uresnet        = chain_cfg.get('enable_uresnet', True)
    self.enable_ppn            = chain_cfg.get('enable_ppn', True)
    self.enable_dbscan         = chain_cfg.get('enable_dbscan', True)
    self.enable_cnn_clust      = chain_cfg.get('enable_cnn_clust', False)

    self.enable_gnn_shower     = chain_cfg.get('enable_gnn_shower', False)
    self.enable_gnn_track      = chain_cfg.get('enable_gnn_track', False)
    self.enable_gnn_particle   = chain_cfg.get('enable_gnn_particle', False)
    self.enable_gnn_inter      = chain_cfg.get('enable_gnn_inter', False)
    self.enable_gnn_kinematics = chain_cfg.get('enable_gnn_kinematics', False)
    self.enable_cosmic         = chain_cfg.get('enable_cosmic', False)

    if self.verbose and print_info:
        print("Shower GNN: {}".format(self.enable_gnn_shower))
        print("Track GNN: {}".format(self.enable_gnn_track))
        print("Particle GNN: {}".format(self.enable_gnn_particle))
        print("Interaction GNN: {}".format(self.enable_gnn_inter))
        print("Kinematics GNN: {}".format(self.enable_gnn_kinematics))
        print("Cosmic GNN: {}".format(self.enable_cosmic))

    if (self.enable_gnn_shower or \
        self.enable_gnn_track or \
        self.enable_gnn_particle or \
        self.enable_gnn_inter or \
        self.enable_gnn_kinematics or self.enable_cosmic):
        if self.verbose and print_info:
            msg = """
            Since one of the GNNs are turned on, process_fragments is turned ON.
            """
            print(msg)
        self.process_fragments = True

    if self.process_fragments and self.verbose and print_info:
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
    self.use_ppn_in_gnn  = chain_cfg.get('use_ppn_in_gnn', False)
    self.use_supp_in_gnn = chain_cfg.get('use_supp_in_gnn', False)

    # Make sure the deghosting config is consistent
    if self.enable_ghost and not self.enable_charge_rescaling:
        assert cfg['uresnet_ppn']['uresnet_lonely']['ghost']
        if self.enable_ppn:
            assert cfg['uresnet_ppn']['ppn']['ghost']

    # Enforce basic logical order
    # 1. Need semantics for everything
    assert self.enable_uresnet
    # 2. If PPN is used in GNN, need PPN
    if self.enable_gnn_shower or self.enable_gnn_track:
        assert self.enable_ppn or (not self.use_ppn_in_gnn)
    # 3. Need at least one of two dense clusterer
    # assert self.enable_dbscan or self.enable_cnn_clust
    # 4. Check that SPICE and DBSCAN are not redundant
    if self.enable_cnn_clust and self.enable_dbscan:
        if 'spice' in cfg:
            assert not (np.array(cfg['spice']['spice_fragment_manager']['cluster_classes']) == \
                        np.array(np.array(cfg['dbscan']['dbscan_fragment_manager']['cluster_classes'])).reshape(-1)).any()
        else:
            assert 'graph_spice' in cfg
            assert set(cfg['dbscan']['dbscan_fragment_manager']['cluster_classes']).issubset(
                set(cfg['graph_spice']['skip_classes']))

    if self.enable_gnn_particle: # If particle fragment GNN is used, make sure it is not redundant
        if self.enable_gnn_shower:
            assert cfg['grappa_shower']['base']['node_type'] \
                not in cfg['grappa_particle']['base']['node_type']

        if self.enable_gnn_track:
            assert cfg['grappa_track']['base']['node_type'] \
                not in cfg['grappa_particle']['base']['node_type']

    if self.enable_cosmic: assert self.enable_gnn_inter # Cosmic classification needs interaction clustering
