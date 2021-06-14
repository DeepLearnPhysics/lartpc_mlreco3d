from mlreco.models.layers.dbscan import MinkDBSCANFragmenter
import torch 
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

from mlreco.models.full_chain import FullChain, FullChainLoss
from mlreco.mink.layers.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.mink_uresnet import UResNet_Chain
from mlreco.models.uresnet_lonely import SegmentationLoss
from mlreco.models.mink_graph_spice import MinkGraphSPICE
from mlreco.models.graph_spice import GraphSPICELoss

from mlreco.utils.cluster.graph_spice import ClusterGraphConstructor
from mlreco.utils.deghosting import adapt_labels


from mlreco.utils.gnn.evaluation import (node_assignment_score, 
                                         primary_assignment)
                                         
from mlreco.utils.gnn.cluster import (form_clusters, 
                                      get_cluster_batch, 
                                      get_cluster_label, 
                                      cluster_direction)


class MinkFullChain(FullChain):
    '''
    Full Chain with MinkowskiEngine implementations for CNNs.

    See FullChain class for general description.
    '''
    MODULES = ['grappa_shower', 'grappa_track', 'grappa_inter',
               'grappa_shower_loss', 'grappa_track_loss', 'grappa_inter_loss',
               'full_chain_loss', 'mink_graph_spice', 'graph_spice_loss',
               'fragment_clustering',  'chain', 'dbscan_frag',
               ('mink_uresnet_ppn', ['mink_uresnet', 'mink_ppn'])]

    def __init__(self, cfg):
        super(MinkFullChain, self).__init__(cfg)

        # Initialize the UResNet+PPN modules
        if self.enable_uresnet:
            self.uresnet_lonely = UResNet_Chain(cfg['uresnet_ppn'], 
                                                name='uresnet_lonely')
            self.input_features = cfg['uresnet_ppn']['uresnet_lonely'].get(
                'features', 1)

        if self.enable_ppn:
            self.ppn            = PPN(cfg['uresnet_ppn'])

        # Initialize the CNN dense clustering module
        self.cluster_classes = []
        if self.enable_cnn_clust:
            self._enable_graph_spice = 'graph_spice' in cfg
            self.spatial_embeddings = MinkGraphSPICE(cfg)
            self.gs_manager = ClusterGraphConstructor(
                cfg['graph_spice']['constructor_cfg'])
            self.gs_manager.training = True # FIXME
            self._graph_spice_skip_classes = cfg['graph_spice']['skip_classes']
            self.frag_cfg = cfg['spice'].get('fragment_clustering', {})

        if self.enable_dbscan:
            self.dbscan_frag = MinkDBSCANFragmenter(cfg)

        print(self)
        assert False


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
            if 'ghost' in ppn_input:
                ppn_input['ghost'] = ppn_input['ghost'][0]
                ppn_output = self.ppn(ppn_input['finalTensor'][0], 
                                      ppn_input['encoderTensors'][0],
                                      ppn_input['ghost'])
            else:
                ppn_output = self.ppn(ppn_input['finalTensor'][0], 
                                      ppn_input['encoderTensors'][0])
            result.update(ppn_output)

        # The rest of the chain only needs 1 input feature
        if self.input_features > 1:
            input[0] = input[0][:, :-self.input_features+1]
        
        cnn_result = {}

        if self.enable_ghost:
            # Update input based on deghosting results
            deghost = result['ghost'][0].argmax(dim=1) == 0
            new_input = [input[0][deghost]]
            if label_seg is not None and label_clustering is not None:

                # ME uses 0 for batch column, so need to compensate
                label_clustering = adapt_labels(result, 
                                                label_seg, 
                                                label_clustering,
                                                batch_column=0,
                                                coords_column_range=(1,4))

            segmentation = result['segmentation'][0].clone()

            if self.enable_ppn:

                points          = result['points'][0].clone()
                ppn_coords_last = result['ppn_coords'][0][-1].clone()
                ppn_layers_last = result['ppn_layers'][0][-1].clone()
                mask_ppn_last   = result['mask_ppn'][0][-1].clone()

            deghost_result = {}
            deghost_result.update(result)
            deghost_result.pop('ghost')
            deghost_result['segmentation'][0] = result['segmentation'][0][deghost]
            if self.enable_ppn:
                deghost_result['points']            = [result['points'][0][deghost]]
                deghost_result['mask_ppn'][0][-1]   = result['mask_ppn'][0][-1][deghost]
                deghost_result['ppn_coords'][0][-1] = result['ppn_coords'][0][-1][deghost]
                deghost_result['ppn_layers'][0][-1] = result['ppn_layers'][0][-1][deghost]
            cnn_result.update(deghost_result)
            cnn_result['ghost'] = result['ghost']
            cnn_result['segmentation'][0] = segmentation

        else:
            cnn_result.update(result)

        # ---
        # 1. Clustering w/ CNN or DBSCAN will produce
        # - fragments (list of list of integer indexing the input data)
        # - frag_batch_ids (list of batch ids for each fragment)
        # - frag_seg (list of integers, semantic label for each fragment)
        # ---

        semantic_labels = torch.argmax(
            cnn_result['segmentation'][0], dim=1).flatten().double()
        semantic_data = torch.cat((input[0][:,:4], 
                                   semantic_labels.reshape(-1,1)), dim=1)
        fragments, frag_batch_ids, frag_seg = [], [], []

        if self.enable_cnn_clust:
            if self._enable_graph_spice:
                if label_clustering is None:
                    raise Exception("Cluster labels from parse_cluster3d_clean_full are needed at this time.")

                filtered_semantic = ~(semantic_labels[..., None].cpu() == \
                                      torch.Tensor(self._graph_spice_skip_classes)).any(-1)

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

                    fragments_spice = form_clusters(filtered_input, column=-1)
                    fragments_batch_ids_spice = get_cluster_batch(filtered_input, fragments_spice)
                    fragments_seg_spice = get_cluster_label(filtered_input, fragments_spice, column=4)
                    # Current indices in fragments_spice refer to input[0][filtered_semantic]
                    # but we want them to refer to input[0]
                    fragments_spice = [np.arange(len(input[0]))[filtered_semantic][clust.cpu().numpy()] \
                                       for clust in fragments_spice]

                    fragments.extend(fragments_spice)
                    frag_batch_ids.extend(fragments_batch_ids_spice)
                    frag_seg.extend(fragments_seg_spice)

            else:
                # Get fragment predictions from the CNN clustering algorithm
                spatial_embeddings_output = self.spatial_embeddings([input[0][:,:5]])
                cnn_result.update(spatial_embeddings_output)

                # Extract fragment predictions to input into the GNN
                fragments_cnn, frag_batch_ids_cnn, frag_seg_cnn = self.extract_fragment(input, cnn_result)
                fragments.extend(fragments_cnn)
                frag_batch_ids.extend(frag_batch_ids_cnn)
                frag_seg.extend(frag_seg_cnn)

        if self.enable_dbscan:
            # Get the fragment predictions from the DBSCAN fragmenter
            fragments_dbscan = self.dbscan_frag(semantic_data, cnn_result)
            frag_batch_ids_dbscan = get_cluster_batch(input[0], fragments_dbscan, batch_index=0)
            frag_seg_dbscan = np.empty(len(fragments_dbscan), dtype=np.int32)
            for i, f in enumerate(fragments_dbscan):
                vals, cnts = semantic_labels[f].unique(return_counts=True)
                assert len(vals) == 1
                frag_seg_dbscan[i] = vals[torch.argmax(cnts)].item()
            fragments.extend(fragments_dbscan)
            frag_batch_ids.extend(frag_batch_ids_dbscan)
            frag_seg.extend(frag_seg_dbscan)

        # Make np.array
        same_length = np.all([len(f) == len(fragments[0]) for f in fragments] )
        fragments = np.array(fragments, dtype=object if not same_length else np.int64)
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.array(frag_seg)

        # Store in result the intermediate fragments
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        bcids = [np.where(frag_batch_ids == b)[0] for b in range(len(counts))]
        same_length = [np.all([len(c) == len(fragments[b][0]) for c in fragments[b]] ) for b in bcids]

        frags = [np.array([vids[c].astype(np.int64) for c in fragments[b]], 
                          dtype=np.object if not same_length[idx] else np.int64) \
                              for idx, b in enumerate(bcids)]

        frags_seg = [frag_seg[b] for idx, b in enumerate(bcids)]

        cnn_result.update({
            'frags': [fragments],   # TODO: name change frags -> fragments and
            'fragments': [frags],   # fragments -> frags (dict key should be consistent with variable name)
            'fragments_seg': [frags_seg],
            'frag_batch_ids': [frag_batch_ids],
            'label_clustering': [label_clustering],
            'semantic_labels': [semantic_labels],
            'vids': [vids]
        })
        
        return cnn_result


class MinkFullChainLoss(FullChainLoss):

    def __init__(self, cfg):
        super(MinkFullChainLoss, self).__init__(cfg)

        # Initialize loss components
        if self.enable_uresnet:
            self.uresnet_loss            = SegmentationLoss(cfg['uresnet_ppn'])
        if self.enable_ppn:
            self.ppn_loss                = PPNLonelyLoss(cfg['uresnet_ppn'], name='ppn')
        if self.enable_cnn_clust:
            assert self._enable_graph_spice
            self.spatial_embeddings_loss = GraphSPICELoss(cfg)
            self._graph_spice_skip_classes = cfg['graph_spice_loss']['skip_classes']