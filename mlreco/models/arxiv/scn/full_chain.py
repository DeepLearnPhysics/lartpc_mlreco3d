import torch
import numpy as np

from mlreco.models.scn.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.models.scn.layers.ppn import PPN, PPNLoss
from mlreco.models.scn.clustercnn_se import ClusterCNN, ClusteringLoss
from mlreco.models.scn.graph_spice import GraphSPICE, GraphSPICELoss
from mlreco.models.scn.layers.cnn_encoder import ResidualEncoder
from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.cluster.fragmenter import DBSCANFragmentManager, GraphSPICEFragmentManager, SPICEFragmentManager, format_fragments
from mlreco.models.gnn_full_chain import FullChainGNN, FullChainLoss
from mlreco.utils.deghosting import adapt_labels


class SCNFullChain(FullChainGNN):
    def __init__(self, cfg):
        super(SCNFullChain, self).__init__(cfg)

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
                self.spatial_embeddings       = GraphSPICE(cfg)
                self.gs_manager               = ClusterGraphConstructor(cfg['graph_spice']['constructor_cfg'])
                self.gs_manager.training      = True # FIXME
                self._gspice_skip_classes     = cfg['graph_spice']['skip_classes']
                self.gspice_fragment_manager  = GraphSPICEFragmentManager(cfg['graph_spice']['gspice_fragment_manager'])
            else:
                self.spatial_embeddings     = ClusterCNN(cfg['spice'])
                self.frag_cfg               = cfg['spice'].get('spice_fragment_manager', {})
                self._s_thresholds          = self.frag_cfg.setdefault('s_thresholds', [0.0, 0.0, 0.0, 0.0])
                self._p_thresholds          = self.frag_cfg.setdefault('p_thresholds', [0.5, 0.5, 0.5, 0.5])
                self._spice_classes         = self.frag_cfg.setdefault('cluster_classes', [])
                self._spice_min_voxels      = self.frag_cfg.setdefault('min_voxels', 2)
                self.spice_fragment_manager = SPICEFragmentManager(self.frag_cfg)

        # Initialize the DBSCAN fragmenter module
        if self.enable_dbscan:
            self.frag_cfg = cfg['dbscan']['dbscan_fragment_manager']
            self.dbscan_fragment_manager = DBSCANFragmentManager(self.frag_cfg)

        # Initialize the interaction classifier module
        if self.enable_cosmic:
            self.cosmic_discriminator = ResidualEncoder(cfg['cosmic_discriminator'])
            self._cosmic_use_input_data = cfg['cosmic_discriminator'].get(
                'use_input_data', True)
            self._cosmic_use_true_interactions = cfg['cosmic_discriminator'].get('use_true_interactions', False)

        print('Total Number of Trainable Parameters (scn full_chain)= {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

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
            segmentation = result['segmentation'][0].clone()
            true_mask = None

            if self.cheat_ghost:
                deghost = label_seg[0][:, self.uresnet_lonely._ghost_label] != self.uresnet_lonely._num_classes
                true_mask = deghost
            else:
                deghost = result['ghost'][0].argmax(dim=1) == 0

            result['ghost_label'] = [deghost]
            input = [input[0][deghost]]
            #print(label_seg[0].shape, label_clustering[0].shape)

            if self.enable_cnn_clust and label_seg is not None and label_clustering is not None:
                label_clustering = adapt_labels(result,
                                                label_seg,
                                                label_clustering,
                                                batch_column=self.batch_col,
                                                true_mask=true_mask)

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
        cluster_result = {
            'fragments': [],
            'frag_batch_ids': [],
            'frag_seg': []
        }
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
                        fragment_data = self.gspice_fragment_manager(filtered_input, input[0], filtered_semantic)
                        cluster_result['fragments'].extend(fragment_data[0])
                        cluster_result['frag_batch_ids'].extend(fragment_data[1])
                        cluster_result['frag_seg'].extend(fragment_data[2])


            else:
                # Get fragment predictions from the CNN clustering algorithm
                spatial_embeddings_output = self.spatial_embeddings([input[0][:,:5]])
                cnn_result.update(spatial_embeddings_output)

                # Extract fragment predictions to input into the GNN
                if self.process_fragments:
                    fragment_data = self.spice_fragment_manager(input[0],
                                                                cnn_result)
                    cluster_result['fragments'].extend(fragment_data[0])
                    cluster_result['frag_batch_ids'].extend(fragment_data[1])
                    cluster_result['frag_seg'].extend(fragment_data[2])

        if self.enable_dbscan and self.process_fragments:
            # Get the fragment predictions from the DBSCAN fragmenter
            fragment_data = self.dbscan_fragment_manager(input[0], cnn_result)
            cluster_result['fragments'].extend(fragment_data[0])
            cluster_result['frag_batch_ids'].extend(fragment_data[1])
            cluster_result['frag_seg'].extend(fragment_data[2])

        # Format Fragments
        fragments_result = format_fragments(cluster_result['fragments'],
                                            cluster_result['frag_batch_ids'],
                                            cluster_result['frag_seg'],
                                            input[0][:, self.batch_col])

        cnn_result.update(fragments_result)

        if self.enable_cnn_clust and label_clustering is not None:
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


class SCNFullChainLoss(FullChainLoss):
    def __init__(self, cfg):
        super(SCNFullChainLoss, self).__init__(cfg)

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
                self._gspice_skip_classes = cfg['graph_spice_loss']['skip_classes']
