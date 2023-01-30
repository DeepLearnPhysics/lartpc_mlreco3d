import torch
import MinkowskiEngine as ME
import numpy as np

from mlreco.models.layers.common.gnn_full_chain import FullChainGNN, FullChainLoss
from mlreco.models.layers.common.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.uresnet import UResNet_Chain, SegmentationLoss
from mlreco.models.graph_spice import MinkGraphSPICE, GraphSPICELoss

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.deghosting import adapt_labels_knn as adapt_labels
from mlreco.utils.deghosting import compute_rescaled_charge
from mlreco.utils.cluster.fragmenter import (DBSCANFragmentManager,
                                             GraphSPICEFragmentManager,
                                             format_fragments)
from mlreco.utils.ppn import get_track_endpoints_geo
from mlreco.utils.gnn.data import _get_extra_gnn_features
from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder


class FullChain(FullChainGNN):
    '''
    Full Chain with MinkowskiEngine implementations for CNNs.

    Modular, End-to-end LArTPC Reconstruction Chain

    - Deghosting for 3D tomographic reconstruction artifiact removal
    - UResNet for voxel-wise semantic segmentation
    - PPN for point proposal
    - DBSCAN/GraphSPICE for dense particle clustering
    - GrapPA(s) for particle/interaction aggregation and identification

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

    The ``chain`` section enables or disables specific
    stages of the full chain. When a module is disabled
    through this section, it will not even be constructed.
    The configuration blocks for each enabled module should
    also live under the `modules` section of the configuration.

    To see an example of full chain configuration, head over to
    https://github.com/DeepLearnPhysics/lartpc_mlreco3d_tutorials/blob/master/book/data/inference.cfg

    See Also
    --------
    mlreco.models.layers.common.gnn_full_chain.FullChainGNN, FullChainLoss
    '''
    MODULES = ['grappa_shower', 'grappa_track', 'grappa_inter',
               'grappa_shower_loss', 'grappa_track_loss', 'grappa_inter_loss',
               'full_chain_loss', 'mink_graph_spice', 'graph_spice_loss',
               'fragment_clustering',  'chain', 'dbscan_frag',
               ('mink_uresnet_ppn', ['mink_uresnet', 'mink_ppn'])]

    def __init__(self, cfg):
        super(FullChain, self).__init__(cfg)

        # Initialize the charge rescaling module
        if self.enable_charge_rescaling:
            self.uresnet_deghost = UResNet_Chain(cfg.get('uresnet_deghost', {}),
                                                 name='uresnet_lonely')
            self.deghost_input_features = self.uresnet_deghost.net.num_input

        # Initialize the UResNet+PPN modules
        self.input_features = 1
        if self.enable_uresnet:
            self.uresnet_lonely = UResNet_Chain(cfg.get('uresnet_ppn', {}),
                                                name='uresnet_lonely')
            self.input_features = self.uresnet_lonely.net.num_input

        if self.enable_ppn:
            self.ppn            = PPN(cfg.get('uresnet_ppn', {}))

        # Initialize the CNN dense clustering module
        # We will only use GraphSPICE for CNN based clustering, as it is
        # superior to SPICE.
        self.cluster_classes = []
        if self.enable_cnn_clust:
            self._enable_graph_spice       = 'graph_spice' in cfg
            self.graph_spice               = MinkGraphSPICE(cfg)
            self.gs_manager                = ClusterGraphConstructor(cfg.get('graph_spice', {}).get('constructor_cfg', {}),
                                                                    batch_col=self.batch_col,
                                                                    training=False) # for downstream, need to run prediction in inference mode
            # edge cut threshold is usually 0. (unspecified) during training, but 0.1 at inference
            self.gs_manager.ths = cfg.get('graph_spice', {}).get('constructor_cfg', {}).get('edge_cut_threshold', 0.1)

            self._gspice_skip_classes         = cfg.get('graph_spice', {}).get('skip_classes', [])
            self._gspice_invert               = cfg.get('graph_spice_loss', {}).get('invert', True)
            self._gspice_fragment_manager     = GraphSPICEFragmentManager(cfg.get('graph_spice', {}).get('gspice_fragment_manager', {}), batch_col=self.batch_col)
            self._gspice_min_points           = cfg.get('graph_spice', {}).get('min_points', 1)

        if self.enable_dbscan:
            self.frag_cfg = cfg.get('dbscan', {}).get('dbscan_fragment_manager', {})
            self.dbscan_fragment_manager = DBSCANFragmentManager(self.frag_cfg,
                                                                 mode='mink')

        # Initialize the interaction classifier module
        if self.enable_cosmic:
            cosmic_cfg = cfg.get('cosmic_discriminator', {})
            self.cosmic_discriminator = SparseResidualEncoder(cosmic_cfg)
            self._cosmic_use_input_data = cosmic_cfg.get('use_input_data', True)
            self._cosmic_use_true_interactions = cosmic_cfg.get('use_true_interactions', False)

        # print('Total Number of Trainable Parameters (mink_full_chain)= {}'.format(
        #             sum(p.numel() for p in self.parameters() if p.requires_grad)))

    @staticmethod
    def get_extra_gnn_features(fragments,
                               frag_seg,
                               classes,
                               input,
                               result,
                               use_ppn=False,
                               use_supp=False):
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
        return _get_extra_gnn_features(fragments,
                                       frag_seg,
                                       classes,
                                       input,
                                       result,
                                       use_ppn=use_ppn,
                                       use_supp=use_supp)


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

        label_seg, label_clustering, coords = None, None, None
        if len(input) == 3:
            input, label_seg, label_clustering = input
            input = [input]
            label_seg = [label_seg]
            label_clustering = [label_clustering]
        elif len(input) == 2:
            input, label_clustering = input
            input = [input]
            label_clustering = [label_clustering]

        # Store batch size for GNN formatting
        batches = torch.unique(input[0][:, self.batch_col])
        assert len(batches) == batches.max().int().item() + 1
        self.batch_size = len(batches)

        result = {}

        if self.enable_charge_rescaling:
            # Pass through the deghosting
            assert self.enable_ghost
            last_index = 4 + self.deghost_input_features
            result.update(self.uresnet_deghost([input[0][:,:last_index]]))
            result['ghost'] = result['segmentation']
            deghost = result['ghost'][0].argmax(dim=1) == 0
            del result['segmentation']

            # Rescale the charge column, store it
            charges = compute_rescaled_charge(input[0], deghost, last_index=last_index)
            input[0][deghost, 4] = charges
            result.update({'input_rescaled':[input[0][deghost,:5]]})

        if self.enable_uresnet:
            if not self.enable_charge_rescaling:
                result.update(self.uresnet_lonely([input[0][:, :4+self.input_features]]))
            else:
                full_seg = torch.zeros((input[0][:,:5].shape[0], 5), device=input[0].device, dtype=input[0].dtype)
                if torch.sum(deghost):
                    result.update(self.uresnet_lonely([input[0][deghost, :4+self.input_features]]))
                    seg = result['segmentation'][0]
                    full_seg[deghost] = seg
                    result['segmentation'][0] = full_seg
                else:
                    result['segmentation'] = [full_seg]
                    return result, input, lambda x: x

        if self.enable_ppn:
            ppn_input = {}
            ppn_input.update(result)
            if 'ghost' in ppn_input and not self.enable_charge_rescaling:
                ppn_input['ghost'] = ppn_input['ghost'][0]
                ppn_output = self.ppn(ppn_input['finalTensor'][0],
                                      ppn_input['decoderTensors'][0],
                                      ppn_input['ghost_sptensor'][0])
            else:
                ppn_output = self.ppn(ppn_input['finalTensor'][0],
                                      ppn_input['decoderTensors'][0])
            result.update(ppn_output)

        # The rest of the chain only needs 1 input feature
        if self.input_features > 1:
            input[0] = input[0][:, :-self.input_features+1]

        cnn_result = {}

        if self.enable_ghost:

            # Update input based on deghosting results
            # if self.cheat_ghost:
            #     assert label_seg is not None
            #     deghost = label_seg[0][:, self.uresnet_lonely.ghost_label] == \
            #               self.uresnet_lonely.num_classes
            #     print(deghost, deghost.shape)
            # else:
            deghost = result['ghost'][0].argmax(dim=1) == 0

            result['ghost_label'] = [deghost]
            input = [input[0][deghost]]

            if label_seg is not None and label_clustering is not None:
                # ME uses 0 for batch column, so need to compensate
                label_clustering = adapt_labels(result,
                                                label_seg,
                                                label_clustering,
                                                batch_column=0,
                                                coords_column_range=(1,4))

            segmentation = result['segmentation'][0].clone()

            deghost_result = {}
            deghost_result.update(result)
            deghost_result.pop('ghost')
            deghost_result['segmentation'][0] = result['segmentation'][0][deghost]
            if self.enable_ppn and not self.enable_charge_rescaling:
                deghost_result['points']            = [result['points'][0][deghost]]
                if 'classify_endpoints' in deghost_result:
                    deghost_result['classify_endpoints'] = [result['classify_endpoints'][0][deghost]]
                deghost_result['mask_ppn'][0][-1]   = result['mask_ppn'][0][-1][deghost]
                #print(len(result['ppn_score']))
                #deghost_result['ppn_score'][0][-1]   = result['ppn_score'][0][-1][deghost]
                deghost_result['ppn_coords'][0][-1] = result['ppn_coords'][0][-1][deghost]
                deghost_result['ppn_layers'][0][-1] = result['ppn_layers'][0][-1][deghost]
            cnn_result.update(deghost_result)
            cnn_result['ghost'] = result['ghost']
            # cnn_result['segmentation'][0] = segmentation

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
        if self._gspice_use_true_labels:
            semantic_labels = label_seg[0][:, -1]
        else:
            semantic_labels = torch.argmax(cnn_result['segmentation'][0], dim=1).flatten()

        if self.enable_cnn_clust:
            if label_clustering is None and self.training:
                raise Exception("Cluster labels from parse_cluster3d_clean_full are needed at this time for training.")

            filtered_semantic = ~(semantic_labels[..., None] == \
                                    torch.tensor(self._gspice_skip_classes, device=device)).any(-1)

            # If there are voxels to process in the given semantic classes
            if torch.count_nonzero(filtered_semantic) > 0:
                if label_clustering is not None:
                    # If labels are present, compute loss and accuracy
                    graph_spice_label = torch.cat((label_clustering[0][:, :-1],
                                                    semantic_labels.reshape(-1,1)), dim=1)
                else:
                #     # Otherwise run in data inference mode (will not compute loss and accuracy)
                    graph_spice_label = torch.cat((input[0][:, :4],
                                                    semantic_labels.reshape(-1, 1)), dim=1)
                cnn_result['graph_spice_label'] = [graph_spice_label]
                spatial_embeddings_output = self.graph_spice((input[0][:,:5],
                                                              graph_spice_label))
                cnn_result.update(spatial_embeddings_output)


                if self.process_fragments:
                    self.gs_manager.replace_state(spatial_embeddings_output['graph'][0],
                                                  spatial_embeddings_output['graph_info'][0])

                    self.gs_manager.fit_predict(invert=self._gspice_invert, min_points=self._gspice_min_points)
                    cluster_predictions = self.gs_manager._node_pred.x
                    filtered_input = torch.cat([input[0][filtered_semantic][:, :4],
                                                semantic_labels[filtered_semantic][:, None],
                                                cluster_predictions.to(device)[:, None]], dim=1)
                    # For the record - (self.gs_manager._node_pred.pos == input[0][filtered_semantic][:, 1:4]).all()
                    # ie ordering of voxels is the same in node predictions and (filtered) input data
                    # with np.printoptions(precision=3, suppress=True):
                    #     print('filtered input', filtered_input.shape, filtered_input[:, 0].sum(), filtered_input[:, 1].sum(), filtered_input[:, 2].sum(), filtered_input[:, 3].sum(), filtered_input[:, 4].sum(), filtered_input[:, 5].sum())
                    #     print(torch.unique( filtered_input[:, 5], return_counts=True))
                    fragment_data = self._gspice_fragment_manager(filtered_input, input[0], filtered_semantic)
                    cluster_result['fragments'].extend(fragment_data[0])
                    cluster_result['frag_batch_ids'].extend(fragment_data[1])
                    cluster_result['frag_seg'].extend(fragment_data[2])

        if self.enable_dbscan and self.process_fragments:
            # Get the fragment predictions from the DBSCAN fragmenter
            # print('Input = ', input[0].shape)
            # print('points = ', cnn_result['points'][0].shape)
            fragment_data = self.dbscan_fragment_manager(input[0], cnn_result)
            cluster_result['fragments'].extend(fragment_data[0])
            cluster_result['frag_batch_ids'].extend(fragment_data[1])
            cluster_result['frag_seg'].extend(fragment_data[2])

        # Format Fragments
        # for i, c in enumerate(cluster_result['fragments']):
        #     print('format' , torch.unique(input[0][c, self.batch_column_id], return_counts=True))
        fragments_result = format_fragments(cluster_result['fragments'],
                                            cluster_result['frag_batch_ids'],
                                            cluster_result['frag_seg'],
                                            input[0][:, self.batch_col],
                                            batch_size=self.batch_size)

        cnn_result.update(fragments_result)

        if self.enable_cnn_clust or self.enable_dbscan:
            cnn_result.update({ 'semantic_labels': [semantic_labels] })
            if label_clustering is not None:
                cnn_result.update({ 'label_clustering': label_clustering })

        # if self.use_true_fragments and coords is not None:
        #     print('adding true points info')
        #     cnn_result['true_points'] = coords

        def return_to_original(result):
            if self.enable_ghost:
                result['segmentation'][0] = segmentation
            return result

        return cnn_result, input, return_to_original


class FullChainLoss(FullChainLoss):
    """
    Loss function for the full chain.

    See Also
    --------
    FullChain, mlreco.models.layers.common.gnn_full_chain.FullChainLoss
    """

    def __init__(self, cfg):
        super(FullChainLoss, self).__init__(cfg)

        # Initialize loss components
        if self.enable_charge_rescaling:
            self.deghost_loss            = SegmentationLoss(cfg.get('uresnet_deghost', {}), batch_col=self.batch_col)
        if self.enable_uresnet:
            self.uresnet_loss            = SegmentationLoss(cfg.get('uresnet_ppn', {}), batch_col=self.batch_col)
        if self.enable_ppn:
            self.ppn_loss                = PPNLonelyLoss(cfg.get('uresnet_ppn', {}), name='ppn')
        if self.enable_cnn_clust:
            # As ME is an updated model, ME backend full chain will not support old SPICE
            # for CNN Clustering.
            # assert self._enable_graph_spice
            self._enable_graph_spice = True
            self.spatial_embeddings_loss = GraphSPICELoss(cfg, name='graph_spice_loss')
            self._gspice_skip_classes = cfg.get('graph_spice_loss', {}).get('skip_classes', [])
