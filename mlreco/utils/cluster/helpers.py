from typing import Union, Callable, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mlreco.utils.metrics import *

from sklearn.neighbors import (KNeighborsClassifier, 
                               RadiusNeighborsClassifier, 
                               kneighbors_graph)
import scipy.sparse as sp

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
# -------------------------- Helper Functions--------------------------------

def knn_sklearn(coords, k=5):
    if isinstance(coords, torch.Tensor):
        device = coords.device
        G = kneighbors_graph(coords.cpu().numpy(), n_neighbors=k).tocoo()
        out = np.vstack([G.row, G.col])
        return torch.Tensor(out).long().to(device=device)
    elif isinstance(coords, np.ndarray):
        G = kneighbors_graph(coords, n_neighbors=k).tocoo()
        out = np.vstack([G.row, G.col])
        return out


def get_edge_weight(sp_emb: torch.Tensor,
                    ft_emb: torch.Tensor,
                    cov: torch.Tensor,
                    edge_indices: torch.Tensor,
                    occ=None,
                    eps=0.001):
    '''
    INPUTS:
        - sp_emb (N x D)
        - ft_emb (N x F)
        - cov (N x 2)
        - occ (N x 0)
        - edge_indices (2 x E)

    RETURNS:
        - pvec (N x 0)
    '''
    # print(edge_indices.shape)
    device = sp_emb.device
    ui, vi = edge_indices[0, :], edge_indices[1, :]

    sp_cov_i = cov[:, 0][ui]
    sp_cov_j = cov[:, 0][vi]
    sp_i = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov_i**2 + eps)
    sp_j = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov_j**2 + eps)

    ft_cov_i = cov[:, 1][ui]
    ft_cov_j = cov[:, 1][vi]
    ft_i = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov_i**2 + eps)
    ft_j = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov_j**2 + eps)

    p_ij = torch.exp(-sp_i-ft_i)
    p_ji = torch.exp(-sp_j-ft_j)

    pvec = torch.clamp(p_ij + p_ji - p_ij * p_ji, min=0, max=1).squeeze()

    if occ is not None:
        r1 = occ[edge_indices[0, :]]
        r2 = occ[edge_indices[1, :]]
        # print(r1.shape, r2.shape)
        r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
        pvec = pvec / r
        # print(pvec.shape)
    pvec = torch.clamp(pvec, min=1e-5, max=1-1e-5)
    return pvec


class StrayAssigner(ABC):
    '''
    Abstract Class for orphan assigning functors.
    '''
    def __init__(self, X, Y, metric_fn : Callable = None, **kwargs):
        self.clustered = X
        self.d = metric_fn
        self.partial_labels = Y
        super().__init__()

    @abstractmethod
    def assign_orphans(self):
        pass

class NearestNeighborsAssigner(StrayAssigner):
    '''
    Assigns orphans to the k-nearest cluster using simple kNN Classifier.
    '''
    def __init__(self, X, Y, **kwargs):
        '''
            X: Points to run Nearest-Neighbor Classifier (N x F)
            Y: Labels of Points (N, )
        '''
        super(NearestNeighborsAssigner, self).__init__(X, Y, **kwargs)
        self._neigh = KNeighborsClassifier(**kwargs)
        self._neigh.fit(X, Y)

    def assign_orphans(self, orphans, get_proba=False):
        pred = self._neigh.predict(orphans)
        self._pred = pred
        if get_proba:
            self._proba = self._neigh.predict_proba(orphans)
            self._max_proba = np.max(self._proba, axis=1)
        return pred


class RadiusNeighborsAssigner(StrayAssigner):
    '''
    Assigns orphans to the k-nearest cluster using simple kNN Classifier.
    '''
    def __init__(self, X, Y, **kwargs):
        '''
            X: Points to run Nearest-Neighbor Classifier (N x F)
            Y: Labels of Points (N, )
        '''
        super(RadiusNeighborsAssigner, self).__init__(X, Y, **kwargs)
        self._neigh = RadiusNeighborsClassifier(**kwargs)
        self._neigh.fit(X, Y)

    def assign_orphans(self, orphans, get_proba=False):
        pred = self._neigh.predict(orphans)
        self._pred = pred
        if get_proba:
            self._proba = self._neigh.predict_proba(orphans)
            self._max_proba = np.max(self._proba, axis=1)
        return pred
    
    
class ConnectedComponents(BaseTransform):
    def __init__(self, connection: str = 'weak'):
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        self.connection = connection

    def fit_predict_one(self, data: Data, edge_mode='edge_index') -> Data:
        
        device = data.x.device
        edge_index = getattr(data, edge_mode)
        if edge_mode == 'edge_index':
            adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes)
        else:
            assert edge_index.shape[1] == 2
            adj = to_scipy_sparse_matrix(edge_index.T, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(
            adj, connection=self.connection)

        data.node_pred = torch.tensor(component).long().to(device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.connection})'
    
    def forward(self, batch: Batch, edge_mode='edge_pred') -> Batch:
        
        data_list = batch.to_data_list()
        for graph_id, subgraph in enumerate(data_list):
            self.fit_predict_one(subgraph)
        out = Batch.from_data_list(data_list)
        return out