import numpy as np
import pandas as pd
import torch

# from mlreco.main_funcs import process_config, train, inference
from mlreco.utils.metrics import *
# from mlreco.trainval import trainval
# from mlreco.main_funcs import process_config
# from mlreco.iotools.factories import loader_factory
# from mlreco.main_funcs import cycle

from pprint import pprint

import networkx as nx
import time
<<<<<<< HEAD
=======
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import cdist


def get_edge_truth(edge_indices, labels):
    '''

        - edge_indices: 2 x E
        - labels: N
    '''
    u = labels[edge_indices[0, :]]
    v = labels[edge_indices[1, :]]
    return (u == v).astype(bool)


def get_edge_weight(sp_emb: torch.Tensor,
                    ft_emb: torch.Tensor,
                    cov: torch.Tensor,
                    edge_indices: torch.Tensor,
                    occ=None,
                    eps=0.001):

    device = sp_emb.device
    if edge_indices.shape[1] == 0:
        return torch.Tensor([0]).to(device)
    ui, vi = edge_indices[0, :], edge_indices[1, :]
    # Compute spatial term
    sp_cov = (cov[:, 0][ui] + cov[:, 0][vi]) / 2
    sp = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov**2 + eps)

    # Compute feature term
    ft_cov = (cov[:, 1][ui] + cov[:, 1][vi]) / 2
    ft = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov**2 + eps)

    pvec = torch.exp(- sp - ft)
    if occ is not None:
        r1 = occ[edge_indices[0, :]]
        r2 = occ[edge_indices[1, :]]
        r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
        pvec = pvec / r
    return pvec


def fit_graph(coords, edge_index, edge_pred, features, min_cluster=10):
    edges = edge_index[edge_pred]
    edge_indices = edges
    edges = [(e[0], e[1]) for i, e in enumerate(edges.cpu().numpy())]
    G = nx.Graph()
    G.add_nodes_from(np.arange(coords.shape[0]))
    G.add_edges_from(edges)
    pred = -np.ones(coords.shape[0], dtype=np.int32)
    hypernode_features = []
    singletons = []
    ccs = []
    labels = []
    temp_pred = np.zeros(coords.shape[0], dtype=np.int32)
    for i, comp in enumerate(nx.connected_components(G)):
        if len(comp) < min_cluster:
            singletons.extend(list(comp))
            x = np.asarray(list(comp))
            temp_pred[x] = i
        else:
            x = np.asarray(list(comp))
            pred[x] = i
            ccs.extend(list(comp))
    if len(singletons) > 0:
        # Handle Singletons
        if len(ccs) == 0:   # Current class only contains singletons
            pred = temp_pred
        else:
            singletons = np.asarray(singletons)
            ccs = np.asarray(ccs)
            u = features[ccs]
            v = features[singletons]
            dist = cdist(v, u)
            nearest = np.argmin(dist, axis=1)
            new_labels = pred[ccs][nearest]
            pred[singletons] = new_labels
    pred, _ = unique_label(pred)
    return pred, edge_indices
>>>>>>> 36637149f2385b7f3a5a16faf6ae497c0d87be21


class OccuSegPredictor:

    def __init__(self, cfg):
        from torch_cluster import knn_graph, radius_graph

        mode = cfg.get('mode', 'knn')
        if mode == 'knn':
            self.graph_constructor = knn_graph
        elif mode == 'radius':
            self.graph_constructor = radius_graph
        else:
            raise ValueError('Mode {} is not supported for initial graph construction!'.format(mode))

        self.ths = cfg.get('cut_threshold', '0.5')
        self.kwargs = cfg.get('cluster_kwargs', dict(k=5))
        self.eps = cfg.get('eps', 0.001)

    @staticmethod
    def get_edge_weight(sp_emb: torch.Tensor,
                        ft_emb: torch.Tensor,
                        cov: torch.Tensor,
                        edge_indices: torch.Tensor,
                        occ=None,
                        eps=0.001):

        device = sp_emb.device
        if edge_indices.shape[1] == 0:
            return torch.Tensor([0]).to(device)
        ui, vi = edge_indices[0, :], edge_indices[1, :]
        # Compute spatial term
        sp_cov = (cov[:, 0][ui] + cov[:, 0][vi]) / 2
        sp = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov**2 + eps)

        # Compute feature term
        ft_cov = (cov[:, 1][ui] + cov[:, 1][vi]) / 2
        ft = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov**2 + eps)

        pvec = torch.exp(- sp - ft)
        if occ is not None:
            r1 = occ[edge_indices[0, :]]
            r2 = occ[edge_indices[1, :]]
            r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
            pvec = pvec / r
        return pvec

    @staticmethod
    def get_edge_attr(sp_emb: torch.Tensor,
                      ft_emb: torch.Tensor,
                      cov: torch.Tensor,
                      edge_indices: torch.Tensor,
                      occ=None,
                      eps=0.001):

        device = sp_emb.device
        if edge_indices.shape[1] == 0:
            return torch.Tensor([0]).to(device)
        ui, vi = edge_indices[0, :], edge_indices[1, :]
        # Compute spatial term
        f = torch.cat([sp_emb, ft_emb, cov, occ], dim=1)
        dist = torch.abs(f[ui] - f[vi])
        return dist

    @staticmethod
    def get_edge_truth(edge_indices: torch.Tensor, labels: torch.Tensor):
        '''

            - edge_indices: 2 x E
            - labels: N
        '''
        u = labels[edge_indices[0, :]]
        v = labels[edge_indices[1, :]]
        return (u == v).long()

    
    def get_edge_and_attr(self, coords: torch.Tensor,
                          sp_emb: torch.Tensor,
                          ft_emb: torch.Tensor,
                          cov: torch.Tensor,
                          occ=None, cluster_all=True):
        edge_indices = self.graph_constructor(coords, **self.kwargs)
        edge_attr = self.get_edge_attr(
            sp_emb, ft_emb, cov, edge_indices, occ, eps=self.eps)
        return edge_indices, edge_attr            


    def fit_predict(self, coords: torch.Tensor,
                          sp_emb: torch.Tensor,
                          ft_emb: torch.Tensor,
                          cov: torch.Tensor,
                          occ=None, cluster_all=True):

        pred = -np.ones(coords.shape[0], dtype=np.int32)
        edge_indices = self.graph_constructor(coords, **self.kwargs)
        if edge_indices.shape[1] < 1:
            pred[0] = 0
            return pred, None, None
        w = self.get_edge_weight(sp_emb, ft_emb, cov, edge_indices, occ=occ.squeeze(), eps=self.eps)
        edge_indices = edge_indices.T[w > self.ths].T
        edges = [(e[0], e[1], w[i].item()) \
            for i, e in enumerate(edge_indices.cpu().numpy())]
        w = w[w > self.ths]
        G = nx.Graph()
        G.add_nodes_from(np.arange(coords.shape[0]))
        G.add_weighted_edges_from(edges)
        for i, comp in enumerate(nx.connected_components(G)):
            x = np.asarray(list(comp))
            pred[x] = i
        return pred, edge_indices, w


class GraphDataConstructor:

    def __init__(self, predictor, cfg):
        self.predictor = predictor
        self.seg_col = cfg.get('seg_col', -1)
        self.cluster_col = cfg.get('cluster_col', 5)
        self.edge_mode = cfg.get('edge_mode', 'probability')
        self.edge_feats = cfg.get('edge_feats', 14)
        self.node_feats = cfg.get('node_feats', 32)

    def construct_graph(self, coords: torch.Tensor,
                              edge_weights: torch.Tensor,
                              edge_index: torch.Tensor,
                              feats: torch.Tensor):
<<<<<<< HEAD
        from torch_geometric.data import Data
        graph_data = Data(x=feats, edge_index=edge_index, edge_attr=edge_weights, pos=coords)
=======
        graph_data = Data(x=feats.view(-1, self.node_feats), 
                          edge_index=edge_index.view(2, -1), 
                          edge_attr=edge_weights.view(-1, self.edge_feats), 
                          pos=coords.view(-1, 3))
>>>>>>> 36637149f2385b7f3a5a16faf6ae497c0d87be21
        return graph_data

    def construct_batched_graphs(self, res):
        from torch_geometric.data import Batch
        data_list = []

        coordinates = res['coordinates'][0]
        segmentation = res['segmentation'][0]
        features = res['features'][0]
        sp_embeddings = res['spatial_embeddings'][0]
        ft_embeddings = res['feature_embeddings'][0]
        covariance = res['covariance'][0]
        batch_indices = res['batch_indices'][0]
        occupancy = res['occupancy'][0]


        for i, bidx in enumerate(torch.unique(batch_indices)):
            mask = batch_indices == bidx
            cov_batch = covariance[mask]
            seg_batch = segmentation[mask]
            occ_batch = occupancy[mask]
            sp_batch = sp_embeddings[mask]
            ft_batch = ft_embeddings[mask]
            coords_batch = coordinates[mask]
            features_batch = features[mask]

            pred_seg = torch.argmax(seg_batch, dim=1).int()

            for c in (torch.unique(pred_seg).int()):
                if int(c) == 4:
                    continue
                class_mask = pred_seg == c
                seg_class = seg_batch[class_mask]
                cov_class = cov_batch[class_mask]
                occ_class = occ_batch[class_mask]
                sp_class = sp_batch[class_mask]
                ft_class = ft_batch[class_mask]
                coords_class = coords_batch[class_mask]
                features_class = features_batch[class_mask]

                if self.edge_mode == 'probability':
                    _, edge_index, w = self.predictor.fit_predict(
                        coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())
                elif self.edge_mode == 'attributes':
                    edge_index, w = self.predictor.get_edge_and_attr(
                        coords_class, sp_class, ft_class, cov_class, occ=occ_class)
                else:
                    raise NotImplementedError
                # print("size = ", edge_index.size())
                if (edge_index is None) or (edge_index.shape[1] == 0):
                    continue

                data = self.construct_graph(coords_class, w, edge_index, features_class)
                data.index = (int(bidx), int(c))
                data_list.append(data)
        graph_batch = Batch().from_data_list(data_list)
        return graph_batch


    def construct_batched_graphs_with_labels(self, res, labels: torch.Tensor):
        from torch_geometric.data import Batch

        data_list = []

        coordinates = res['coordinates'][0]
        segmentation = res['segmentation'][0]
        features = res['features'][0]
        sp_embeddings = res['spatial_embeddings'][0]
        ft_embeddings = res['feature_embeddings'][0]
        covariance = res['covariance'][0]
        batch_indices = res['batch_indices'][0]
        occupancy = res['occupancy'][0]

        for i, bidx in enumerate(torch.unique(batch_indices)):
            cov_batch = covariance[mask]
            seg_batch = segmentation[mask]
            occ_batch = occupancy[mask]
            sp_batch = sp_embeddings[mask]
            coords_batch = coordinates[mask]
            features_batch = features[mask]

            for c in torch.unique(labels_batch[:, self.seg_col]):
                if int(c) == 4:
                    continue
                class_mask = labels_batch[:, self.seg_col] == c
                seg_class = seg_batch[class_mask]
                cov_class = cov_batch[class_mask]
                occ_class = occ_batch[class_mask]
                sp_class = sp_batch[class_mask]
                ft_class = ft_batch[class_mask]
                coords_class = coords_batch[class_mask]
                features_class = features_batch[class_mask]
                frag_labels = labels_batch[class_mask][:, self.cluster_col]

                if self.edge_mode == 'probability':
                    _, edge_index, w = self.predictor.fit_predict(
                        coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())
                elif self.edge_mode == 'attributes':
                    edge_index, w = self.predictor.get_edge_and_attr(
                        coords_class, sp_class, ft_class, cov_class, occ=occ_class)
                else:
                    raise NotImplementedError

                data = self.construct_graph(coords_class, w, edge_index, features_class)
                truth = self.predictor.get_edge_truth(edge_index, frag_labels)
                data.edge_truth = truth
                data.index = (int(bidx), int(c))
                data_list.append(data)
        graph_batch = Batch().from_data_list(data_list)
        return graph_batch
