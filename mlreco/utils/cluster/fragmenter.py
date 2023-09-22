from abc import abstractmethod
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label

import torch
import torch.nn as nn
import numpy as np

from mlreco.utils.cluster.dense_cluster import fit_predict, gaussian_kernel_cuda
# from mlreco.models.layers.common.dbscan import DBSCANFragmenter
from mlreco.models.layers.common.dbscan import DBSCANFragmenter


def format_fragments(fragments, frag_batch_ids, frag_seg, batch_column, batch_size=None):
    """
    INPUTS:
        - fragments
        - frag_batch_ids
        - frag_seg
        - batch_column
    """
    fragments_np      = np.empty(len(fragments), dtype=object)
    fragments_np[:]   = fragments
    frag_batch_ids_np = np.array(frag_batch_ids)
    frag_seg_np       = np.array(frag_seg)

    batches, counts = torch.unique(batch_column, return_counts=True)
    # In case one of the events is "missing" and len(counts) < batch_size
    if batch_size is not None:
        new_counts = torch.zeros(batch_size, dtype=torch.int64, device=counts.device)
        new_counts[batches.long()] = counts
        counts = new_counts
        # `batches` does not matter after this

    vids = np.concatenate([np.arange(n.item()) for n in counts])
    bcids = [np.where(frag_batch_ids_np == b)[0] for b in range(len(counts))]
    frags = [np.empty(len(b), dtype=object) for b in bcids]
    for idx, b in enumerate(bcids):
        frags[idx][:] = [vids[c].astype(np.int64) for c in fragments_np[b]]

    frags_seg = [frag_seg_np[b].astype(np.int32) for idx, b in enumerate(bcids)]

    out = {
        'frags'             : [fragments_np],
        'frag_seg'          : [frag_seg_np],
        'frag_batch_ids'    : [frag_batch_ids_np],
        'fragment_clusts'   : [frags],
        'fragment_seg'      : [frags_seg],
        'fragment_batch_ids': [frag_batch_ids_np],
        'vids'              : [vids],
        'counts'            : [counts]
    }

    return out


class FragmentManager(nn.Module):
    """
    Base class for fragmenters

    Fragmenters take the input data tensor + outputs feature maps of CNNs
    in <result> to give fragment labels.
    """
    def __init__(self, frag_cfg : dict, batch_col : int = 0):
        super(FragmentManager, self).__init__()
        self._batch_column                = batch_col
        self._semantic_column             = frag_cfg.get('semantic_column', -1)
        self._use_segmentation_prediction = frag_cfg.get('use_segmentation_prediction', True)


    @abstractmethod
    def forward(self, input, cnn_result, semantic_labels=None):
        """
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - ppn_points
                - ppn_masks

        Returns:
            - fragment_data
        """
        raise NotImplementedError


class DBSCANFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for DBSCAN Clustering
    '''
    def __init__(self, frag_cfg: dict, mode='mink', **kwargs):
        super(DBSCANFragmentManager, self).__init__(frag_cfg, **kwargs)
        dbscan_frag = {'dbscan_frag': frag_cfg}
        if mode == 'mink':
            self._batch_column = 0
            self.dbscan_fragmenter = DBSCANFragmenter(dbscan_frag)
        # elif mode == 'scn':
        #     self._batch_column = 3
        #     self.dbscan_fragmenter = DBSCANFragmenter(dbscan_frag)
        else:
            raise Exception('Invalid sparse CNN backend name {}'.format(mode))


    def forward(self, input, cnn_result, semantic_labels=None):
        '''
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - ppn_points
                - ppn_masks

        Returns:
            - fragments
            - frag_batch_ids
            - frag_seg

        '''
        if self._use_segmentation_prediction:
            assert semantic_labels is None
            semantic_labels = torch.argmax(cnn_result['segmentation'][0],
                                           dim=1).flatten()

        semantic_data = torch.cat([input[:, :4],
                                   semantic_labels.reshape(-1, 1)], dim=1)

        fragments = self.dbscan_fragmenter(semantic_data,
                                           cnn_result)

        frag_batch_ids = get_cluster_batch(input[:, :5], fragments)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, counts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(counts)].item()

        return fragments, frag_batch_ids, frag_seg


class SPICEFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for SPICE Clustering
    '''
    def __init__(self, frag_cfg : dict, **kwargs):
        super(SPICEFragmentManager, self).__init__(frag_cfg, **kwargs)
        self._s_thresholds     = frag_cfg.get('s_thresholds'   , [0.0, 0.0, 0.0, 0.0])
        self._p_thresholds     = frag_cfg.get('p_thresholds'   , [0.5, 0.5, 0.5, 0.5])
        self._spice_classes    = frag_cfg.get('cluster_classes', []                  )
        self._spice_min_voxels = frag_cfg.get('min_voxels'     , 2                   )

    def forward(self, input, cnn_result, semantic_labels=None):
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
        if self._use_segmentation_prediction:
            assert semantic_labels is None
            semantic_labels = torch.argmax(cnn_result['segmentation'][0],
                                           dim=1).flatten()

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

        fragments_np    = np.empty(len(fragments), dtype=object)
        fragments_np[:] = fragments
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(cnts)].item()

        return fragemnts_np, frag_batch_ids, frag_seg
            

class GraphSPICEFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for GraphSPICE Clustering
    '''
    def __init__(self, frag_cfg : dict, **kwargs):
        super(GraphSPICEFragmentManager, self).__init__(frag_cfg, **kwargs)


    def process(self, filtered_input, n, filtered_semantic, offset=0):
        
        fragments = form_clusters(filtered_input, column=-1)
        fragments = [f.int().detach().cpu().numpy() for f in fragments]

        if len(fragments) > 0:
            frag_batch_ids = get_cluster_batch(filtered_input.detach().cpu().numpy(),\
                                            fragments)
            fragments_seg = get_cluster_label(filtered_input, fragments, column=-2)
            fragments_id = get_cluster_label(filtered_input, fragments, column=-1)
        else:
            frag_batch_ids = np.empty((0,))
            fragments_seg = np.empty((0,))
            fragments_id = np.empty((0,))
        
        fragments = [np.arange(n)[filtered_semantic.detach().cpu().numpy()][clust]+offset \
                     for clust in fragments]

        return fragments, frag_batch_ids, fragments_seg, fragments_id

    def forward(self, filtered_input, original_input, filtered_semantic):
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
        all_fragments, all_frag_batch_ids, all_fragments_seg = [], [], []
        all_fragments_id = []
        for b in filtered_input[:, self._batch_column].unique():
            mask = filtered_input[:, self._batch_column] == b
            original_mask = original_input[:, self._batch_column] == b
        
            # How many voxels belong to that batch
            n = torch.count_nonzero(original_mask)
            # The index start of the batch in original data
            # - note: we cannot simply accumulate the values
            # of n, as this will fail if a batch is missing
            # from the original data (eg no track in that batch).
            offset = torch.nonzero(original_mask).min().item()
            
            fragments, frag_batch_ids, fragments_seg, fragments_id = self.process(filtered_input[mask], 
                                                                    n.item(), 
                                                                    filtered_semantic[original_mask].cpu(),
                                                                    offset=offset)
            
            all_fragments.extend(fragments)
            all_frag_batch_ids.extend(frag_batch_ids)
            all_fragments_seg.extend(fragments_seg)
            all_fragments_id.extend(fragments_id)
        return all_fragments, all_frag_batch_ids, all_fragments_seg, all_fragments_id
