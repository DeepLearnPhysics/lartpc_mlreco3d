import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pprint import pprint

import networkx as nx
from torch_cluster import knn_graph, radius_graph
from typing import Union

from mlreco.utils.metrics import *
from mlreco.utils.graph_batch import GraphBatch
from torch_geometric.data import Data as GraphData

# class OccuSegPredictor:

#     def __init__(self, cfg):
#         mode = cfg.get('mode', 'knn')
#         if mode == 'knn':
#             self.init_graph = knn_graph
#         elif mode == 'radius':
#             self.init_graph = radius_graph
#         else:
#             raise ValueError('Mode {} is not supported for initial graph construction!'.format(mode))

#         self.ths = cfg.get('cut_threshold', '0.5')
#         self.kwargs = cfg.get('cluster_kwargs', dict(k=5))
#         self.eps = cfg.get('eps', 0.001)

#     @staticmethod
#     def get_edge_weight(sp_emb: torch.Tensor,
#                         ft_emb: torch.Tensor,
#                         cov: torch.Tensor,
#                         edge_indices: torch.Tensor,
#                         occ=None,
#                         eps=0.001):

#         device = sp_emb.device
#         ui, vi = edge_indices[0, :], edge_indices[1, :]

#         sp_cov_i = cov[:, 0][ui]
#         sp_cov_j = cov[:, 0][vi]
#         sp_i = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov_i**2 + eps)
#         sp_j = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov_j**2 + eps)

#         ft_cov_i = cov[:, 1][ui]
#         ft_cov_j = cov[:, 1][vi]
#         ft_i = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov_i**2 + eps)
#         ft_j = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov_j**2 + eps)

#         p_ij = torch.exp(-sp_i-ft_i)
#         p_ji = torch.exp(-sp_j-ft_j)

#         pvec = torch.clamp(p_ij + p_ji - p_ij * p_ji, min=0, max=1)


#         # # Compute spatial term
#         # sp_cov = (cov[:, 0][ui] + cov[:, 0][vi]) / 2
#         # sp = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov**2 + eps)

#         # # Compute feature term
#         # ft_cov = (cov[:, 1][ui] + cov[:, 1][vi]) / 2
#         # ft = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov**2 + eps)

#         # pvec = torch.exp(- sp - ft)
#         if occ is not None:
#             r1 = occ[edge_indices[0, :]]
#             r2 = occ[edge_indices[1, :]]
#             r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
#             pvec = pvec / r
#         return pvec

#     @staticmethod
#     def get_edge_truth(edge_indices: torch.Tensor, labels: torch.Tensor):
#         '''

#             - edge_indices: 2 x E
#             - labels: N
#         '''
#         u = labels[edge_indices[0, :]]
#         v = labels[edge_indices[1, :]]
#         return (u == v).long()


#     def fit_predict(self, coords: torch.Tensor,
#                           sp_emb: torch.Tensor,
#                           ft_emb: torch.Tensor,
#                           cov: torch.Tensor,
#                           occ=None, cluster_all=True):

#         edge_indices = self.init_graph(coords, **self.kwargs)
#         w = self.get_edge_weight(sp_emb, ft_emb, cov, edge_indices, occ=occ.squeeze(), eps=self.eps)
#         edge_indices = edge_indices.T[w > self.ths]
#         edges = [(e[0], e[1], w[i].item()) \
#             for i, e in enumerate(edge_indices.cpu().numpy())]
#         w = w[w > self.ths]
#         G = nx.Graph()
#         G.add_nodes_from(np.arange(coords.shape[0]))
#         G.add_weighted_edges_from(edges)
#         pred = -np.ones(coords.shape[0], dtype=np.int32)
#         for i, comp in enumerate(nx.connected_components(G)):
#             x = np.asarray(list(comp))
#             pred[x] = i
#         return pred, edge_indices, w


# class GraphDataConstructor:

#     def __init__(self, predictor, cfg):
#         self.predictor = predictor
#         self.seg_col = cfg.get('seg_col', -1)
#         self.cluster_col = cfg.get('cluster_col', 5)

#     def construct_graph(self, coords: torch.Tensor,
#                               edge_weights: torch.Tensor,
#                               edge_index: torch.Tensor,
#                               feats: torch.Tensor):

#         graph_data = GraphData(x=feats, edge_index=edge_index, edge_attr=edge_weights, pos=coords)
#         return graph_data

#     def construct_batched_graphs(self, res):

#         data_list = []

#         coordinates = res['coordinates'][0]
#         segmentation = res['segmentation'][0]
#         features = res['features'][0]
#         sp_embeddings = res['spatial_embeddings'][0]
#         ft_embeddings = res['feature_embeddings'][0]
#         covariance = res['covariance'][0]
#         batch_indices = res['batch_indices'][0]
#         occupancy = res['occupancy'][0]


#         for i, bidx in enumerate(torch.unique(batch_indices)):
#             mask = batch_indices == bidx
#             cov_batch = covariance[mask]
#             seg_batch = segmentation[mask]
#             occ_batch = occupancy[mask]
#             sp_batch = sp_embeddings[mask]
#             ft_batch = ft_embeddings[mask]
#             coords_batch = coordinates[mask]
#             features_batch = features[mask]

#             pred_seg = torch.argmax(seg_batch, dim=1).int()

#             for c in (torch.unique(pred_seg).int()):
#                 if int(c) == 4:
#                     continue
#                 class_mask = pred_seg == c
#                 seg_class = seg_batch[class_mask]
#                 cov_class = cov_batch[class_mask]
#                 occ_class = occ_batch[class_mask]
#                 sp_class = sp_batch[class_mask]
#                 ft_class = ft_batch[class_mask]
#                 coords_class = coords_batch[class_mask]
#                 features_class = features_batch[class_mask]

#                 pred, edge_index, w = self.predictor.fit_predict(
#                     coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())

#                 data = self.construct_graph(coords_class, w, edge_index.T, features_class)
#                 data_list.append(data)

#         # for x in data_list:
#         #     print(x)
#         graph_batch = GraphBatch().from_data_list(data_list)
#         return graph_batch


#     def construct_batched_graphs_with_labels(self, res, labels: torch.Tensor):
#         data_list = []

#         coordinates = res['coordinates'][0]
#         segmentation = res['segmentation'][0]
#         features = res['features'][0]
#         sp_embeddings = res['spatial_embeddings'][0]
#         ft_embeddings = res['feature_embeddings'][0]
#         covariance = res['covariance'][0]
#         batch_indices = res['batch_indices'][0]
#         occupancy = res['occupancy'][0]

#         for i, bidx in enumerate(torch.unique(batch_indices)):
#             mask = batch_indices == bidx
#             cov_batch = covariance[mask]
#             seg_batch = segmentation[mask]
#             occ_batch = occupancy[mask]
#             sp_batch = sp_embeddings[mask]
#             ft_batch = ft_embeddings[mask]
#             coords_batch = coordinates[mask]
#             features_batch = features[mask]
#             labels_batch = labels[mask].int()

#             for c in torch.unique(labels_batch[:, self.seg_col]):
#                 if int(c) == 4:
#                     continue
#                 class_mask = labels_batch[:, self.seg_col] == c
#                 seg_class = seg_batch[class_mask]
#                 cov_class = cov_batch[class_mask]
#                 occ_class = occ_batch[class_mask]
#                 sp_class = sp_batch[class_mask]
#                 ft_class = ft_batch[class_mask]
#                 coords_class = coords_batch[class_mask]
#                 features_class = features_batch[class_mask]
#                 frag_labels = labels_batch[class_mask][:, self.cluster_col]

#                 pred, edge_index, w = self.predictor.fit_predict(
#                     coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())

#                 data = self.construct_graph(coords_class, w, edge_index, features_class)
#                 truth = self.predictor.get_edge_truth(edge_index, frag_labels)
#                 data.edge_truth = truth
#                 data_list.append(data)
#         graph_batch = GraphBatch.from_data_list(data_list)
#         return graph_batch


class ParametricGDC(nn.Module):
    '''
    Parametric Graph-SPICE clustering

    Parametric GDC includes a bilinear layer to predict edge weights, 
    given pairs of node features. 
    '''
    def __init__(self, cfg):
        super(ParametricGDC, self).__init__()

        # Input Data/Label conventions
        self.seg_col = cfg.get('seg_col', -1)
        self.cluster_col = cfg.get('cluster_col', 5)

        # Initial Neighbor Graph Construction Mode
        mode = cfg.get('mode', 'knn')
        if mode == 'knn':
            self._init_graph = knn_graph
        elif mode == 'radius':
            self._init_graph = radius_graph
        else:
            raise ValueError('''Mode {} is not supported for initial 
                graph construction!'''.format(mode))

        # Clustering Algorithm Parameters
        self.ths = cfg.get('edge_cut_threshold', '0.5')
        self.kwargs = cfg.get('cluster_kwargs', dict(k=5))
        self.hyp_dim = cfg['hyper_dimension']

        # Parametrized Model (Edge Attribute Constructor)
        self.bilinear = nn.Bilinear(self.hyp_dim, self.hyp_dim, 1)

        # GraphBatch containing graphs per semantic class. 
        self._graph_batch = GraphBatch()

    @property
    def graph_batch(self):
        if self._graph_batch is None:
            raise('The GraphBatch data has not been initialized yet!')
        return self._graph_batch

    @property
    def entry_mapping(self):
        return self._entry_mapping


    def get_graph(self, batch_id, semantic_id):
        '''
        Retrieve single graph from GraphBatch object by batch and semantic id.

        INPUTS:
            - event_id: Event ID (Index)
            - semantic_id: Semantic Class (0-4)

        RETURNS:
            - Subgraph corresponding to class [semantic_id] and event [event_id]
        '''
        df = self._entry_mapping.query(
            'BatchID == {} and SemanticID == {}'.format(batch_id, semantic_id))
        assert df.shape[0] < 2
        if df.shape[0] == 0:
            raise ValueError('''Event ID: {} and Class Label: {} does not 
                exist in current batch'''.format(batch_id, semantic_id))
            return None
        else:
            entry_num = df['GraphID'].item()
            return self._graph_batch.get_example(entry_num)


    def fit_predict_one(self, batch_id, semantic_id, 
                        gen_numpy_graph=False, 
                        min_points=0,
                        cluster_all=True,
                        remainder_alg='knn'):
        '''
        Generate predicted fragment cluster labels for single subgraph.
        '''
        subgraph = self.get_graph(batch_id, semantic_id)
        G = nx.Graph()
        G.add_nodes_from(np.arange(subgraph.num_nodes))

        # Drop edges with low predicted probability score
        edges = subgraph.edge_index.T.cpu().numpy()
        edge_weights = subgraph.edge_attr.cpu().numpy()
        pos_edges = edges[edge_weights > self.ths]

        edges = [(e[0], e[1], w) for e, w in zip(pos_edges, edge_weights)]
        G.add_weighted_edges_from(edges)
        pred = -np.ones(coords.shape[0], dtype=np.int32)
        for i, comp in enumerate(nx.connected_components(G)):
            x = np.asarray(list(comp))
            pred[x] = i

        if gen_numpy_graph:
            G.x = subgraph.x.cpu().numpy()
            G.edge_index = subgraph.edge_index.cpu().numpy()
            G.edge_attr = subgraph.edge_index.cpu().numpy()
            G.pos = subgraph.pos.cpu().numpy()
        return pred, G


    def fit_predict(self):
        '''
        Iterate over all subgraphs and assign predicted labels. 
        '''
        pass


    @staticmethod
    def get_edge_truth(edge_indices : torch.Tensor, 
                       fragment_labels : torch.Tensor):
        '''
        Given edge indices and ground truth fragment labels, 
        get true labels for binary edge classification. 

        INPUTS:
            - edge_indices : 2 x E
            - labels : (N, ) Fragment label tensor

        RETURNS:
            - Edge labels : (N,) Tensor, where 0 indicate edges between 
            different fragment voxels and 1 otherwise. 
        '''
        u = fragment_labels[edge_indices[0, :]]
        v = fragment_labels[edge_indices[1, :]]
        return (u == v).long()


    @staticmethod
    def construct_graph(coords : torch.Tensor,
                        edge_index : torch.Tensor,
                        feats : torch.Tensor):
        '''
        Given spatial coordinates, edge indices of knn-graph, and node
        features, construct graph data object (pytorch_geometric).
        '''
        graph_data = GraphData(x=feats, edge_index=edge_index, pos=coords)
        return graph_data
    

    def initialize_graph(self, res : dict, 
                               labels: Union[torch.Tensor, None]):
        '''
        From GraphSPICE Embedder Output, initialize GraphBatch object
        with edge truth labels. 

        Inputs:
            - res (dict): result dictionary output of GraphSPICE Embedder
            - labels ( N x F Tensor) : 

        Transforms point cloud embeddings to collection of graphs 
        (one per unique image id and segmentation class), and stores graph
        collection as attribute.
        '''
        features = res['hypergraph_features'][0]
        batch_indices = res['batch_indices'][0]
        coordinates = res['coordinates'][0]
        data_list = []

        graph_id = 0

        self._entry_mapping = []

        for i, bidx in enumerate(torch.unique(batch_indices)):
            mask = batch_indices == bidx
            coords_batch = coordinates[mask]
            features_batch = features[mask]
            if labels is not None:
                labels_batch = labels[mask].int()

            for c in torch.unique(labels_batch[:, self.seg_col]):
                if int(c) == 4:
                    continue
                class_mask = labels_batch[:, self.seg_col] == c
                coords_class = coords_batch[class_mask]
                features_class = features_batch[class_mask]

                edge_indices = self._init_graph(coords_class, **self.kwargs)
                data = self.construct_graph(coords_class, 
                                            edge_indices, 
                                            features_class)
                graph_id_key = dict(Index=0, 
                                    BatchID=int(bidx), 
                                    SemanticID=int(c), 
                                    GraphID=graph_id)
                graph_id += 1
                self._entry_mapping.append(graph_id_key)

                if labels is not None:
                    if self.training:
                        frag_labels = labels_batch[class_mask][:, self.cluster_col]
                        truth = self.get_edge_truth(edge_indices, frag_labels)
                        data.edge_truth = truth
                data_list.append(data)

        self._entry_mapping = pd.DataFrame(self._entry_mapping)
        self.data_list = data_list
        self._graph_batch = self._graph_batch.from_data_list(data_list)
        print(self._graph_batch)

    def _set_edge_attributes(self):
        '''
        Constructs edge attributes from node feature tensors, and saves 
        edge attributes to current GraphBatch. 
        '''
        if self._graph_batch is None:
            raise ValueError('The graph data has not been initialized yet!')
        elif isinstance(self._graph_batch.edge_attr, torch.Tensor):
            raise ValueError('Edge attributes are already set: {}'\
                .format(self._graph_batch.edge_attr))
        else:
            edge_attr = self.bilinear(
                self._graph_batch.x[self._graph_batch.edge_index[0, :]],
                self._graph_batch.x[self._graph_batch.edge_index[1, :]])
            self._graph_batch.edge_attr = edge_attr.squeeze()
            self._graph_batch.__slices__['edge_attr'] = \
                self._graph_batch.__slices__['edge_index']
            self._graph_batch.__cat_dims__['edge_attr'] = \
                self._graph_batch.__cat_dims__['x']
            self._graph_batch.__cumsum__['edge_attr'] = \
                self._graph_batch.__cumsum__['x']


    def forward(self, res : dict, 
                      labels: Union[torch.Tensor, None]):
        '''
        Train time labels include cluster column (default: 5) and segmentation column (default: -1)
        Test time labels only include segment column (default: -1)
        '''
        self.initialize_graph(res, labels)
        self._set_edge_attributes()
        return self._graph_batch