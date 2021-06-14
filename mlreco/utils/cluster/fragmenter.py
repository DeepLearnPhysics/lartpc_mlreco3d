from abc import abstractmethod
from lartpc_mlreco3d.mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label

import torch
import torch.nn as nn
import numpy as np

from mlreco.utils.cluster.dense_cluster import fit_predict, gaussian_kernel_cuda
from mlreco.models.layers.dbscan import DBSCANFragmenter


class FragmentManager(nn.Module):
    """
    Base class for fragmenters

    Fragmenters take the input data tensor + outputs feature maps of CNNs
    in <result> to give fragment labels. 
    """
    def __init__(self, frag_cfg : dict):
        super(FragmentManager, self).__init__()
        self._batch_column                = frag_cfg.get('batch_column', 3)
        self._semantic_column             = frag_cfg.get('semantic_column', -1)
        self._use_segmentation_prediction = frag_cfg.get('use_segmentation_prediction', True)

    @staticmethod
    def make_np_array(fragments, frag_batch_ids, frag_seg):
        """
        TODO: Add Docstring
        """
        same_length = np.all([len(f) == len(fragments[0]) for f in fragments])
        fragments_np = np.array(fragments, 
                             dtype=object if not same_length else np.int64)
        frag_batch_ids_np = np.array(frag_batch_ids)
        frag_seg_np = np.array(frag_seg)
        return fragments_np, frag_batch_ids_np, frag_seg_np

    @staticmethod
    def unwrap_fragments(batch_column, fragments, frag_batch_ids, frag_seg):
        """
        TODO: Add Docstring
        """
        _, counts = torch.unique(batch_column, return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        bcids = [np.where(frag_batch_ids == b)[0] for b in range(len(counts))]
        same_length = [np.all([len(c) == len(fragments[b][0]) \
                       for c in fragments[b]] ) for b in bcids]

        frags = [np.array([vids[c].astype(np.int64) for c in fragments[b]], 
                          dtype=np.object if not same_length[idx] else np.int64) \
                          for idx, b in enumerate(bcids)]

        frags_seg = [frag_seg[b] for idx, b in enumerate(bcids)]

        return frags, frags_seg


    @abstractmethod
    def extract_fragments(self, input, cnn_result):
        raise NotImplementedError


    def forward(self, input, cnn_result, semantic_labels=None):
        """
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - points
                - mask_ppn2

        Returns:
            - frag:
            - fragments:
            - frag_batch_ids:
            - frag_seg:
        """

        if self._use_segmentation_prediction:
            assert semantic_labels is None
            semantic_labels = torch.argmax(cnn_result['segmentation'][0], 
                                           dim=1).flatten().double()

        fragment_data = self.extract_fragments(input, 
                                               cnn_result, 
                                               semantic_labels=semantic_labels)

        fragments, frag_batch_ids, frag_seg = self.make_np_array(*fragment_data)

        frags, frags_seg = self.unwrap_fragments(input[:, self._batch_column], 
                                                 fragments, 
                                                 frag_batch_ids, 
                                                 frag_seg)

        out = {
            'frags'         : [frags],
            'fragments'     : [fragments],
            'fragments_seg' : [frags_seg],
            'frag_batch_ids': [frag_batch_ids]
        }

        return out


class DBSCANFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for DBSCAN Clustering
    '''
    def __init__(self, frag_cfg: dict):
        super(DBSCANFragmentManager, self).__init__(frag_cfg)
        dbscan_frag = {'dbscan_frag': frag_cfg}
        self.dbscan_fragmenter = DBSCANFragmenter(dbscan_frag)


    def extract_fragments(self, input, cnn_result, semantic_labels):
        '''
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - points
                - mask_ppn2

        Returns:
            - fragments
            - frag_batch_ids
            - frag_seg

        '''

        semantic_data = torch.cat([input[:, :4], 
                                   semantic_labels.reshape(-1, 1)], dim=1)
        fragments, frags = [], []
        fragments_dbscan = self.dbscan_fragmenter(semantic_data, 
                                                  cnn_result)
        frag_batch_ids = get_cluster_batch(input[:, :5], fragments_dbscan,
                                           batch_index=self._batch_column)
        frag_seg = np.empty(len(fragments_dbscan), dtype=np.int32)
        for i, f in enumerate(fragments_dbscan):
            vals, counts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(counts)].item()
        fragments.extend(fragments_dbscan)

        return fragments, frag_batch_ids, frag_seg


class SPICEFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for SPICE Clustering
    '''
    def __init__(self, frag_cfg : dict):
        super(SPICEFragmentManager, self).__init__(frag_cfg)
        self._s_thresholds     = frag_cfg.get('s_thresholds'   , [0.0, 0.0, 0.0, 0.0])
        self._p_thresholds     = frag_cfg.get('p_thresholds'   , [0.5, 0.5, 0.5, 0.5])
        self._spice_classes    = frag_cfg.get('cluster_classes', []                  )
        self._spice_min_voxels = frag_cfg.get('min_voxels'     , 2                   )

    def extract_fragments(self, input, cnn_result, semantic_labels):
        '''
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - embeddings
                - seediness
                - margins

        Returns:
            - fragments
            - frag_batch_ids
            - frag_seg

        '''
        batch_labels = input[:, self._batch_column]
        fragments, frag_batch_ids = [], []
        for batch_id in batch_labels.unique():
            for s in self._spice_classes:
                mask = torch.nonzero((batch_labels == batch_id) &
                                     (semantic_labels == s), as_tuple=True)[0]
                if len(cnn_result['embeddings'][0][mask]) < self._spice_min_voxels:
                    continue

                pred_labels = fit_predict(embeddings  = cnn_result['embeddings'][0][mask],
                                          seediness   = cnn_result['seediness'][0][mask],
                                          margins     = cnn_result['margins'][0][mask],
                                          fitfunc     = gaussian_kernel_cuda,
                                          s_threshold = self._s_thresholds[s],
                                          p_threshold = self._p_thresholds[s])

                for c in pred_labels.unique():
                    if c < 0:
                        continue
                    fragments.append(mask[pred_labels == c])
                    frag_batch_ids.append(int(batch_id))

        same_length = np.all([len(f) == len(fragments[0]) for f in fragments])
        fragments = np.array([f.detach().cpu().numpy() for f in fragments if len(f)],
                             dtype=object if not same_length else np.int64)
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(cnts)].item()

        return fragments, frag_batch_ids, frag_seg


class GraphSPICEFragmenter(FragmentManager):
    '''
    Full chain model fragment mananger for GraphSPICE Clustering
    '''
    def __init__(self, frag_cfg : dict):
        super(GraphSPICEFragmenter, self).__init__(frag_cfg)
        

    def extract_fragments(self, input, 
                          cnn_result, 
                          semantic_labels, 
                          gs_manager):
        '''
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
                for GraphSPICE, we skip clustering for some labels
                (namely michel, delta, and low E)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - graph
                - graph_info
            - semantic_labels:
            - gs_manager: ClusterGraphManager object for GraphSPICE handling

        Returns:
            - fragments
            - frag_batch_ids
            - frag_seg

        '''
        device = input.device
        # If there are voxels to process in the given semantic classes
        gs_manager.replace_state(cnn_result['graph'][0], 
                                      cnn_result['graph_info'][0])

        gs_manager.fit_predict(gen_numpy_graph=True)
        cluster_predictions = gs_manager._node_pred.x
        filtered_input = torch.cat([input[0][:, :4], 
                                    semantic_labels[:, None], 
                                    cluster_predictions.to(device)[:, None]], dim=1)

        fragments = form_clusters(filtered_input, column=-1)
        fragments_batch_ids = get_cluster_batch(filtered_input, fragments)
        fragments_seg = get_cluster_label(filtered_input, fragments, column=4)
        # Current indices in fragments_spice refer to input[0][filtered_semantic]
        # but we want them to refer to input[0]
        fragments = [np.arange(input.shape[0])[clust.cpu().numpy()] \
                     for clust in fragments]

        return fragments, fragments_batch_ids, fragments_seg

        