import torch
import MinkowskiEngine as ME
import numpy as np

from mlreco.models.layers.common.gnn_full_chain import FullChainGNN, FullChainLoss
from mlreco.models.layers.common.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder
from mlreco.models.uresnet import UResNet_Chain, SegmentationLoss
from mlreco.models.graph_spice import GraphSPICE, GraphSPICELoss

from mlreco.utils.globals import *
from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import get_particle_points
from mlreco.utils.ghost import compute_rescaled_charge, adapt_labels
from mlreco.utils.cluster.fragmenter import (DBSCANFragmentManager,
                                             GraphSPICEFragmentManager,
                                             format_fragments)
from mlreco.utils.gnn.cluster import get_cluster_features_extended
from mlreco.utils.unwrap import prefix_unwrapper_rules



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

    RETURNS = { # TODO
        'fragment_clusts': ['index_list', ['input_data', 'fragment_batch_ids'], True],
        'fragment_seg' : ['tensor', 'fragment_batch_ids', True],
        'fragment_batch_ids' : ['tensor'],
        'particle_seg': ['tensor', 'particle_batch_ids', True],
        'segment_label_tmp': ['tensor', 'input_data'], # Will get rid of this
        'cluster_label_adapted': ['tensor', 'cluster_label_adapted', False, True]
    }

    def __init__(self, cfg):
        super(FullChain, self).__init__(cfg)

        # Initialize the charge rescaling module
        if self.enable_charge_rescaling:
            self.uresnet_deghost = UResNet_Chain(cfg.get('uresnet_deghost', {}),
                                                 name='uresnet_lonely')
            self.deghost_input_features = self.uresnet_deghost.net.num_input
            self.RETURNS.update(self.uresnet_deghost.RETURNS)
            self.RETURNS['input_rescaled'] = ['tensor', 'input_rescaled', False, True]
            self.RETURNS['input_rescaled_coll'] = ['tensor', 'input_rescaled', False, True]
            self.RETURNS['input_rescaled_source'] = ['tensor', 'input_rescaled']
            self.RETURNS['segmentation'][1] = 'input_rescaled'
            self.RETURNS['segment_label_tmp'][1] = 'input_rescaled'
            self.RETURNS['fragment_clusts'][1][0] = 'input_rescaled'

        # Initialize the UResNet+PPN modules
        self.input_features = 1
        if self.enable_uresnet:
            self.uresnet_lonely = UResNet_Chain(cfg.get('uresnet_ppn', {}),
                                                name='uresnet_lonely')
            self.input_features = self.uresnet_lonely.net.num_input
            self.RETURNS.update(self.uresnet_lonely.RETURNS)

        if self.enable_ppn:
            self.ppn            = PPN(cfg.get('uresnet_ppn', {}))
            self.RETURNS.update(self.ppn.RETURNS)

        # Initialize the CNN dense clustering module
        # We will only use GraphSPICE for CNN based clustering, as it is
        # superior to SPICE.
        self.cluster_classes = []
        if self.enable_cnn_clust:
            self._enable_graph_spice       = 'graph_spice' in cfg
            self.graph_spice               = GraphSPICE(cfg)
            self.gs_manager                = ClusterGraphConstructor(cfg.get('graph_spice', {}).get('constructor_cfg', {}),
                                                                    # batch_col=self.batch_col,
                                                                     training=False) # for downstream, need to run prediction in inference mode
            # edge cut threshold is usually 0. (unspecified) during training, but 0.1 at inference
            self.gs_manager.ths = cfg.get('graph_spice', {}).get('constructor_cfg', {}).get('edge_cut_threshold', 0.1)

            self._gspice_skip_classes         = cfg.get('graph_spice', {}).get('skip_classes', [])
            self._gspice_invert               = cfg.get('graph_spice_loss', {}).get('invert', True)
            self._gspice_fragment_manager     = GraphSPICEFragmentManager(cfg.get('graph_spice', {}).get('gspice_fragment_manager', {}), batch_col=self.batch_col)
            self._gspice_min_points           = cfg.get('graph_spice', {}).get('min_points', 1)

            self.RETURNS.update(prefix_unwrapper_rules(self.graph_spice.RETURNS, 'graph_spice'))
            self.RETURNS['graph_spice_label'] = ['tensor', 'graph_spice_label', False, True]


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
    def get_extra_gnn_features(data, result, clusts, clusts_seg, classes,
            add_points=True, add_value=True, add_shape=True):
        '''
        Extracting extra features to feed into the GNN particle aggregators

        Parameters
        ==========
        data : torch.Tensor
            Tensor of input voxels to the particle aggregator
        result : dict
            Dictionary of output of the CNN stages
        clusts : List[numpy.ndarray]
            List of clusters representing the fragment or particle objects
        clusts_seg : numpy.ndarray
            Array of cluster semantic types
        classes : List, optional
            List of semantic classes to include in the output set of particles
        add_points : bool, default True
            If `True`, add particle points as node features
        add_value : bool, default True
            If `True`, add mean and std voxel values as node features
        add_shape : bool, default True
            If `True`, add cluster semantic shape as a node feature

        Returns
        =======
        index : np.ndarray
            Index to select fragments belonging to one of the requested classes
        kwargs : dict
            Keys can include `points` (if `add_points` is `True`)
            and `extra_feats` (if `add_value` or `add_shape` is True).
        '''
        # If needed, build a particle mask based on semantic classes
        if classes is not None:
            mask = np.zeros(len(clusts_seg), dtype=bool)
            for c in classes:
                mask |= (clusts_seg == c)
            index = np.where(mask)[0]
        else:
            index = np.arange(len(clusts))

        # Get the particle end points, if requested
        kwargs = {}
        if add_points:
            coords     = data[0][:, COORD_COLS].detach().cpu().numpy()
            ppn_points = result['ppn_points'][0].detach().cpu().numpy()
            points     = get_particle_points(coords, clusts[index],
                    clusts_seg[index], ppn_points)

            kwargs['points'] = torch.tensor(points,
                    dtype=torch.float, device=data[0].device)

        # Get the supplemental information, if requested
        if add_value or add_shape:
            extra_feats = torch.empty((len(index), 2*add_value + add_shape),
                    dtype=torch.float, device=data[0].device)
            if add_value:
                extra_feats[:,:2] = get_cluster_features_extended(data[0],
                        clusts[index], add_value=True, add_shape=False)
            if add_shape:
                extra_feats[:,-1] = torch.tensor(clusts_seg[index],
                        dtype=torch.float, device=data[0].device)

            kwargs['extra_feats'] = torch.tensor(extra_feats,
                    dtype=torch.float, device=data[0].device)

        return index, kwargs


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
        if not len(input[0]):
            # TODO: move empty case handling elsewhere
            return {}, input

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

        # If not availabel, store batch size for GNN formatting
        if not hasattr(self, 'batch_size'):
            batches = torch.unique(input[0][:, self.batch_col])
            assert len(batches) == batches.max().int().item() + 1
            self.batch_size = len(batches)

        result = {}

        deghost = None
        if self.enable_charge_rescaling:
            # Pass through the deghosting
            assert self.enable_ghost
            last_index = 4 + self.deghost_input_features
            result.update(self.uresnet_deghost([input[0][:,:last_index]]))
            result['ghost'] = result['segmentation']
            deghost = result['ghost'][0][:, 0] > result['ghost'][0][:,1]
            del result['segmentation']

            # Rescale the charge column, store it
            charges = compute_rescaled_charge(input[0], deghost, last_index=last_index)
            charges_coll = compute_rescaled_charge(input[0], deghost, last_index=last_index, collection_only=True)
            input[0][deghost, VALUE_COL] = charges if not self.collection_charge_only else charges_coll

            input_rescaled = input[0][deghost,:5].clone()
            input_rescaled[:, VALUE_COL] = charges
            input_rescaled_coll = input[0][deghost,:5].clone()
            input_rescaled_coll[:, VALUE_COL] = charges_coll

            result.update({'input_rescaled':[input_rescaled]})
            result.update({'input_rescaled_coll':[input_rescaled_coll]})
            if input[0].shape[1] == (last_index + 6 + 2):
                result.update({'input_rescaled_source':[input[0][deghost,-2:]]})

        if self.enable_uresnet:
            if not self.enable_charge_rescaling:
                result.update(self.uresnet_lonely([input[0][:, :4+self.input_features]]))
            else:
                if torch.sum(deghost):
                    result.update(self.uresnet_lonely([input[0][deghost, :4+self.input_features]]))
                else:
                    # TODO: move empty case handling elsewhere
                    seg = torch.zeros((input[0][deghost,:5].shape[0], 5), device=input[0].device, dtype=input[0].dtype) # DUMB
                    result['segmentation'] = [seg]
                    return result, input

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

        if label_seg is not None and label_clustering is not None:
            label_clustering = [adapt_labels(label_clustering[0],
                                             label_seg[0],
                                             result['segmentation'][0],
                                             deghost)]

        if self.enable_ghost:
            # Update input based on deghosting results
            # if self.cheat_ghost:
            #     assert label_seg is not None
            #     deghost = label_seg[0][:, self.uresnet_lonely.ghost_label] == \
            #               self.uresnet_lonely.num_classes
            #     print(deghost, deghost.shape)
            # else:
            deghost = result['ghost'][0][:,0] > result['ghost'][0][:,1]

            input = [input[0][deghost]]

            deghost_result = {}
            deghost_result.update(result)
            deghost_result.pop('ghost')
            if self.enable_ppn and not self.enable_charge_rescaling:
                deghost_result['ppn_points'] = [result['ppn_points'][0][deghost]]
                deghost_result['ppn_masks'][0][-1]  = result['ppn_masks'][0][-1][deghost]
                deghost_result['ppn_coords'][0][-1] = result['ppn_coords'][0][-1][deghost]
                deghost_result['ppn_layers'][0][-1] = result['ppn_layers'][0][-1][deghost]
                if 'ppn_classify_endpoints' in deghost_result:
                    deghost_result['ppn_classify_endpoints'] = [result['ppn_classify_endpoints'][0][deghost]]
            cnn_result.update(deghost_result)
            cnn_result['ghost'] = result['ghost']

        else:
            cnn_result.update(result)


        # ---
        # 1. Clustering w/ CNN or DBSCAN will produce
        # - fragments (list of list of integer indexing the input data)
        # - fragments_batch_ids (list of batch ids for each fragment)
        # - fragments_seg (list of integers, semantic label for each fragment)
        # ---

        cluster_result = {
            'fragment_clusts': [],
            'fragment_batch_ids': [],
            'fragment_seg': []
        }
        if self._gspice_use_true_labels:
            semantic_labels = label_seg[0][:, -1]
        else:
            semantic_labels = torch.argmax(cnn_result['segmentation'][0], dim=1).flatten()
            if not self.enable_charge_rescaling and 'ghost' in cnn_result:
                deghost = result['ghost'][0].argmax(dim=1) == 0
                semantic_labels = semantic_labels[deghost]

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
                cnn_result.update({f'graph_spice_{k}':v for k, v in spatial_embeddings_output.items()})

                if self.process_fragments:

                    self.gs_manager.load_state(spatial_embeddings_output)   
                    
                    graphs = self.gs_manager.fit_predict(min_points=self._gspice_min_points)
                    
                    perm = torch.argsort(graphs.voxel_id)
                    cluster_predictions = graphs.node_pred[perm]

                    filtered_input = torch.cat([input[0][filtered_semantic][:, :4],
                                                semantic_labels[filtered_semantic].view(-1, 1),
                                                cluster_predictions.view(-1, 1)], dim=1)
                    
                    # For the record - (self.gs_manager._node_pred.pos == input[0][filtered_semantic][:, 1:4]).all()
                    # ie ordering of voxels is NOT the same in node predictions and (filtered) input data
                    # It is likely that input data is lexsorted while node predictions 
                    # (and anything that are concatenated through Batch.from_data_list) are not. 

                    fragment_data = self._gspice_fragment_manager(filtered_input, input[0], filtered_semantic)
                    cluster_result['fragment_clusts'].extend(fragment_data[0])
                    cluster_result['fragment_batch_ids'].extend(fragment_data[1])
                    cluster_result['fragment_seg'].extend(fragment_data[2])

        if self.enable_dbscan and self.process_fragments:
            # Get the fragment predictions from the DBSCAN fragmenter
            fragment_data = self.dbscan_fragment_manager(input[0], cnn_result)
            cluster_result['fragment_clusts'].extend(fragment_data[0])
            cluster_result['fragment_batch_ids'].extend(fragment_data[1])
            cluster_result['fragment_seg'].extend(fragment_data[2])

        # Format Fragments
        fragments_result = format_fragments(cluster_result['fragment_clusts'],
                                            cluster_result['fragment_batch_ids'],
                                            cluster_result['fragment_seg'],
                                            input[0][:, self.batch_col],
                                            batch_size=self.batch_size)

        cnn_result.update({'frag_dict':fragments_result})

        cnn_result.update({
            'fragment_clusts': fragments_result['fragment_clusts'],
            'fragment_seg': fragments_result['fragment_seg'],
            'fragment_batch_ids': fragments_result['fragment_batch_ids']
        })

        if self.enable_cnn_clust or self.enable_dbscan:
            cnn_result.update({'segment_label_tmp': [semantic_labels] })
            if label_clustering is not None:
                if 'input_rescaled' in cnn_result:
                    label_clustering[0][:, VALUE_COL] = input[0][:, VALUE_COL]
                cnn_result.update({'cluster_label_adapted': label_clustering })

        # if self.use_true_fragments and coords is not None:
        #     print('adding true points info')
        #     cnn_result['true_points'] = coords

        return cnn_result, input


class FullChainLoss(FullChainLoss):
    '''
    Loss function for the full chain.

    See Also
    --------
    FullChain, mlreco.models.layers.common.gnn_full_chain.FullChainLoss
    '''

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
