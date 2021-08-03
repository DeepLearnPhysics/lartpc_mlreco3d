from abc import abstractmethod
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label

import torch
import torch.nn as nn
import numpy as np

from mlreco.utils.cluster.dense_cluster import fit_predict, gaussian_kernel_cuda
from mlreco.models.layers.dbscan import DBSCANFragmenter, MinkDBSCANFragmenter


def format_fragments(fragments, frag_batch_ids, frag_seg, batch_column):
    """
    INPUTS:
        - fragments
        - frag_batch_ids
        - frag_seg
        - batch_column
    """
    same_length = np.all([len(f) == len(fragments[0]) for f in fragments])
    fragments_np = np.array(fragments,
                            dtype=object if not same_length else np.int64)
    frag_batch_ids_np = np.array(frag_batch_ids)
    frag_seg_np = np.array(frag_seg)

    _, counts = torch.unique(batch_column, return_counts=True)
    vids = np.concatenate([np.arange(n.item()) for n in counts])
    bcids = [np.where(frag_batch_ids_np == b)[0] for b in range(len(counts))]
    same_length = [np.all([len(c) == len(fragments_np[b][0]) \
                    for c in fragments_np[b]] ) for b in bcids]

    frags = [np.array([vids[c].astype(np.int64) for c in fragments_np[b]],
                        dtype=np.object if not same_length[idx] else np.int64) \
                        for idx, b in enumerate(bcids)]

    frags_seg = [frag_seg_np[b] for idx, b in enumerate(bcids)]

    out = {
        'frags'         : [fragments_np],
        'frag_seg'      : [frag_seg_np],
        'fragments'     : [frags],
        'fragments_seg' : [frags_seg],
        'frag_batch_ids': [frag_batch_ids_np],
        'vids'          : [vids],
        'counts'        : [counts]
    }

    return out


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


    @abstractmethod
    def forward(self, input, cnn_result, semantic_labels=None):
        """
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - points
                - mask_ppn2

        Returns:
            - fragment_data
        """
        raise NotImplementedError


class DBSCANFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for DBSCAN Clustering
    '''
    def __init__(self, frag_cfg: dict, mode='scn'):
        super(DBSCANFragmentManager, self).__init__(frag_cfg)
        dbscan_frag = {'dbscan_frag': frag_cfg}
        if mode == 'mink':
            self.dbscan_fragmenter = MinkDBSCANFragmenter(dbscan_frag)
        elif mode == 'scn':
            self.dbscan_fragmenter = DBSCANFragmenter(dbscan_frag)
        else:
            raise Exception('Invalid sparse CNN backend name {}'.format(mode))


    def forward(self, input, cnn_result, semantic_labels=None):
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
        if self._use_segmentation_prediction:
            assert semantic_labels is None
            semantic_labels = torch.argmax(cnn_result['segmentation'][0],
                                           dim=1).flatten().double()

        semantic_data = torch.cat([input[:, :4],
                                   semantic_labels.reshape(-1, 1)], dim=1)

        fragments = self.dbscan_fragmenter(semantic_data,
                                           cnn_result)

        frag_batch_ids = get_cluster_batch(input[:, :5], fragments,
                                           batch_index=self._batch_column)
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
    def __init__(self, frag_cfg : dict):
        super(SPICEFragmentManager, self).__init__(frag_cfg)
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
                                           dim=1).flatten().double()

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


class GraphSPICEFragmentManager(FragmentManager):
    '''
    Full chain model fragment mananger for GraphSPICE Clustering
    '''
    def __init__(self, frag_cfg : dict):
        super(GraphSPICEFragmentManager, self).__init__(frag_cfg)


    def process(self, filtered_input, n, filtered_semantic, offset=0):
        fragments = form_clusters(filtered_input, column=-1)
        fragments = [f.int().detach().cpu().numpy() for f in fragments]
        # for i, f in enumerate(fragments):
        #     print(torch.unique(filtered_input[f, self._batch_column], return_counts=True))

        #print(torch.unique(filtered_input[:, self._batch_column]))
        frag_batch_ids = get_cluster_batch(filtered_input.detach().cpu().numpy(), \
                                        fragments, batch_index=self._batch_column)
        fragments_seg = get_cluster_label(filtered_input, fragments, column=4)
        # fragments = [np.arange(filtered_input.shape[0])[clust] \
        #              for clust in fragments]
        # We want the indices to refer to the unfiltered, original input
        #filtered_semantic = filtered_semantic.detach().cpu().numpy()
        fragments = [np.arange(n)[filtered_semantic][clust]+offset \
                     for clust in fragments]
        return fragments, frag_batch_ids, fragments_seg

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
        offset = 0
        all_fragments, all_frag_batch_ids, all_fragments_seg = [], [], []
        for b in filtered_input[:, self._batch_column].unique():
            mask = filtered_input[:, self._batch_column] == b
            original_mask = original_input[:, self._batch_column] == b
            n = torch.count_nonzero(original_mask)
            fragments, frag_batch_ids, fragments_seg = self.process(filtered_input[mask], n.item(), filtered_semantic[original_mask], offset=offset)
            offset += n.item()
            all_fragments.extend(fragments)
            all_frag_batch_ids.extend(frag_batch_ids)
            all_fragments_seg.extend(fragments_seg)
        return all_fragments, all_frag_batch_ids, all_fragments_seg
