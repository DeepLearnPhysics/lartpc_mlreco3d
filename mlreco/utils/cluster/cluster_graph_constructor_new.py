from typing import Union, Callable, Tuple, List

# Silencing this warning about unclassified (-1) voxels
# /usr/local/lib/python3.8/dist-packages/sklearn/neighbors/_classification.py:601: 
# UserWarning: Outlier label -1 is not in training classes. 
# All class probabilities of outliers will be assigned with 0.
# warnings.warn('Outlier label {} is not in training '
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn

from pprint import pprint
from torch_cluster import knn_graph, radius_graph

from mlreco.utils.metrics import *
from mlreco.utils.globals import *
from torch_geometric.data import Data, Batch

from .helpers import ConnectedComponents

class ClusterGraphConstructor:
    """Manager class for handling per-batch, per-semantic type predictions
    in GraphSPICE clustering.
    """
    
    def __init__(self, 
                 constructor_cfg : dict,
                 training=False):
        """Initialize the ClusterGraphConstructor.

        Parameters
        ----------
        graph_type : str, optional
            Type of graph to construct, by default 'knn'
        edge_cut_threshold : float, optional
            Thresholding value for edge prediction, by default 0.1
        training : bool, optional
            Whether we are in training mode or not, by default False

        Raises
        ------
        ValueError
            If the graph type is not supported.
        """
        self.constructor_cfg     = constructor_cfg
        self._graph_type         = constructor_cfg.get('mode', 'knn')
        self._edge_cut_threshold = constructor_cfg.get('edge_cut_threshold', 0.0) 
        self._graph_params       = constructor_cfg.get('cluster_kwargs', dict(k=5))
        self._skip_classes       = constructor_cfg.get('skip_classes', [])
        self.training = training
        
        # Graph Constructors
        if self._graph_type == 'knn':
            self._init_graph = knn_graph
        elif self._graph_type == 'radius':
            self._init_graph = radius_graph
        else:
            msg = f"Graph type {self._graph_type} is not supported for "\
                "GraphSPICE initialzation!"
            raise ValueError(msg)
        
        # Clustering Algorithm Parameters
        self.ths = self._edge_cut_threshold # Prob values 0-1
        if self.training:
            self.ths = 0.0

        # Radius within which orphans get assigned to neighbor cluster
        self._orphans_radius      = constructor_cfg.get('orphans_radius', 1.9)
        self._orphans_iterate     = constructor_cfg.get('orphans_iterate', True)
        self._orphans_cluster_all = constructor_cfg.get('orphans_cluster_all', True)
        self.use_cluster_labels   = constructor_cfg.get('use_cluster_labels', True)
        
        self._data = None
        self._key_to_graph = {}
        self._graph_to_key = {}
        self._cc_predictor = ConnectedComponents()
        
    
    def clear_data(self):
        """Clear the data stored in the constructor.
        """
        self._data = None
        self._key_to_graph = {}
        self._graph_to_key = {}
        
        
    def initialize_graph(self, res : dict,
                               labels: Union[torch.Tensor, list],
                               kernel_fn: Callable,
                               unwrapped=False,
                               invert=False):
        """Initialize the graph for a given batch.

        Parameters
        ----------
        res : dict
            Results dictionary from the network.
        labels : Union[torch.Tensor, list]
            Labels for the batch.
        kernel_fn : Callable
            Kernel function for computing edge weights.
        unwrapped : bool, optional
            Indicates whether the results are unwrapped or not, by default False
        invert : bool, optional
            Indicates whether the model is trained to predict disconnected edges.

        Returns
        -------
        None
            Operates in place.
        """
        self.clear_data()
        
        if unwrapped:
            return self._initialize_graph_unwrapped(res, labels, kernel_fn,
                                                    invert=invert)

        features = res['hypergraph_features'][0]
        batch_indices = res['coordinates'][0][:, BATCH_COL].int()
        coordinates = res['coordinates'][0][:, COORD_COLS]
        
        data_list = []
        graph_id  = 0

        for i, bidx in enumerate(torch.unique(batch_indices)):
            mask = batch_indices == bidx
            coords_batch = coordinates[mask]
            features_batch = features[mask]
            labels_batch = labels[mask].int()

            for c in torch.unique(labels_batch[:, SHAPE_COL]):
                data = self._construct_graph_data(int(bidx),
                                    int(c),
                                    graph_id,
                                    coords_batch,
                                    features_batch,
                                    labels_batch,
                                    kernel_fn=kernel_fn,
                                    invert=invert)
                data_list.append(data)
                graph_id += 1
                
        self._data = Batch.from_data_list(data_list)

        
    def initialize_graph_unwrapped(self, res, labels, kernel_fn,
                                   invert=False):
        """Same as initialize_graph, but for unwrapped results.
        """
        self.clear_data()
        
        features    = res['hypergraph_features']
        batch_size  = len(res['coordinates'])
        coordinates = res['coordinates'][:, COORD_COLS]
        
        assert batch_size == len(features)
        assert batch_size == len(coordinates)
        
        data_list = []
        graph_id  = 0
        
        for bidx in range(batch_size):
            coords_batch   = coordinates[bidx]
            features_batch = features[bidx]
            labels_batch   = labels[bidx]
            
            for c in torch.unique(labels_batch[:, SHAPE_COL]):
                data = self._construct_graph_data(int(bidx),
                                                  int(c),
                                                  graph_id,
                                                  coords_batch,
                                                  features_batch,
                                                  labels_batch,
                                                  kernel_fn=kernel_fn,
                                                  invert=invert)
                data_list.append(data)
                graph_id += 1
        
        self._data = Batch.from_data_list(data_list)
        

    def _construct_graph_data(self, 
                              batch_id,
                              semantic_type,
                              graph_id,
                              coords_batch, 
                              features_batch, 
                              labels_batch,
                              kernel_fn,
                              invert=False,
                              build_graph=True,
                              edge_index=None) -> Data:
        """Construct a single graph for a given batch, semantic type pair.
        """
        
        class_mask = labels_batch[:, SHAPE_COL] == semantic_type
        coords_class = coords_batch[class_mask]
        features_class = features_batch[class_mask]

        if build_graph:
            edge_index = self._init_graph(coords_class, **self._graph_params)
        else:
            assert edge_index.shape[0] == 2
        edge_attr  = kernel_fn(
            features_class[edge_index[0, :]],
            features_class[edge_index[1, :]])
        
        data = Data(x=features_class,
                    pos=coords_class,
                    edge_index=edge_index,
                    edge_attr=edge_attr)
        
        # Mappings from GraphID to (BatchID, SemanticID)
        data.graph_id = int(graph_id)
        data.graph_key = (int(batch_id), int(semantic_type))
        self._key_to_graph[(int(batch_id), int(semantic_type))] = graph_id
        self._graph_to_key[graph_id] = (int(batch_id), int(semantic_type))
        self._predict_edges(data, invert=invert)

        if self.use_cluster_labels:
            frag_labels = labels_batch[class_mask][:, CLUST_COL]
            truth = self.get_edge_truth(edge_index, frag_labels)
            data.node_truth = frag_labels
            data.edge_label = truth
            data.edge_truth = data.edge_index[:, truth == 1].T
        return data

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
        
    def get_graph_at(self, batch_id, semantic_id):
        """Retrieve the graph at a given batch, semantic type pair.

        Parameters
        ----------
        batch_id : int
            Batch ID.
        semantic_id : int
            Semantic type ID.

        Returns
        -------
        subgraph: torch_geometric.data.Data
            One subgraph corresponding to a unique (batch, semantic type) pair.
        """
        graph_id = self._key_to_graph((batch_id, semantic_id))
        return self._data.get_example(graph_id)
    
    def get_example(self, graph_id, verbose=False):
        if verbose:
            key = self._graph_to_key[graph_id]
            print(f"Graph at BatchID={key[0]} and SemanticID={key[1]}")
        return self._data.get_example(graph_id)
    
    def __getitem__(self, graph_id):
        return self.get_example(graph_id)
    
    def _predict_edges(self, data, invert=False):
        
        device = data.edge_attr.device
        edge_index  = data.edge_index.T
        edge_logits = data.edge_attr
        edge_probs  = torch.sigmoid(edge_logits)
        
        if invert:
            # If invert=True, model is trained to predict disconnected edges
            # as positives. Hence connected edges are those with scores less
            # than the thresholding value. (Higher the value, the more likely
            # it is to be disconnected). 
            mask = (edge_probs < self.ths).squeeze()
            edge_pred  = edge_index[mask]
        else:
            mask = (edge_probs >= self.ths).squeeze()
            edge_pred  = edge_index[mask]
            
        data.edge_pred = torch.tensor(edge_pred).to(device)
        
    def fit_predict(self, 
                    skip=[-1, 4], 
                    edge_mode='edge_pred', 
                    **kwargs):
        """Run GraphSPICE clustering on all graphs in the batch.

        Parameters
        ----------
        skip : list, optional
            Semantic types to skip, by default [-1, 4]
        edge_mode : str, optional
            Edge mode to use for clustering, by default 'edge_pred'

        Returns
        -------
        node_pred : torch.Tensor
            Predicted cluster labels for each voxel.
        """
        skip = set(skip)
        graphs = self._cc_predictor.forward(self._data, edge_mode=edge_mode)
        self._data = graphs
        return graphs.node_pred
    
    def save_state(self):
        
        attr_names = [
            'edge_index',
            'edge_attr',
            'edge_label',
            'edge_truth',
            'edge_pred',
            'x',
            'pos',
            'node_pred',
            'node_truth',
        ]
        
        state_dict = defaultdict(list)
        
        data_list = self._data.to_data_list()
        for graph_id, subgraph in enumerate(data_list):
            for attr_name in attr_names:
                if hasattr(subgraph, attr_name):
                    state_dict[attr_name].append(getattr(subgraph, attr_name))
            state_dict['graph_id'].append(int(subgraph.graph_id))
            state_dict['graph_key'].append(subgraph.graph_key)
            
        return state_dict
    
    
    # def save_state(self, unwrapped=False):
        
    #     if unwrapped:
    #         return self.save_state_unwrapped()
    #     else:
    #         attr_names = [
    #             'edge_index',
    #             'edge_attr',
    #             'edge_label',
    #             'edge_truth',
    #             'edge_pred',
    #             'x',
    #             'pos',
    #             'node_pred',
    #             'node_truth',
    #             'batch'
    #         ]
            
    #         state_dict = defaultdict(list)
    #         for attr_name in attr_names:
    #             if hasattr(self._data, attr_name):
    #                 state_dict[attr_name] = [getattr(self._data, attr_name)]
    #         return state_dict
    
    
    def load_state(self, state_dict):
        
        self.clear_data()
        data_list = []
        optionals  = ['node_pred', 'node_truth', 
                      'edge_label', 'edge_truth', 'edge_pred']
        num_graphs = len(state_dict['x'])
        for i in range(num_graphs):
            subgraph = Data(x=state_dict['x'][i],
                            pos=state_dict['pos'][i],
                            edge_index=state_dict['edge_index'][i],
                            edge_attr=state_dict['edge_attr'][i],
                            edge_pred=state_dict['edge_pred'][i])
            for name in optionals:
                if name in state_dict:
                    setattr(subgraph, name, state_dict[name][i])
                    
            subgraph.graph_id = int(state_dict['graph_id'][i])
            subgraph.graph_key = tuple(state_dict['graph_key'][i])
            self._key_to_graph[state_dict['graph_key'][i]] = state_dict['graph_id'][i]
            self._graph_to_key[state_dict['graph_id'][i]] = state_dict['graph_key'][i]
            data_list.append(subgraph)
        
        self._data = Batch.from_data_list(data_list)
        
    # def load_state(self, state_dict, unwrapped=False):
        
    #     if unwrapped:
    #         self.load_state_unwrapped(state_dict)
    #     else:
    #         self.clear_data()
    #         optionals  = ['node_pred', 'node_truth', 
    #                     'edge_label', 'edge_truth', 'edge_pred']
    #         num_graphs = len(state_dict['batch'][0].unique())
    #         for graph_id in range(num_graphs):
    #             data = Data(x=state_dict['x'][0][state_dict['batch'][0] == graph_id],
    #                         pos=state_dict['pos'][0][state_dict['batch'][0] == graph_id],
    #                         edge)
        
    
    def __call__(self, res: dict,
                 kernel_fn: Callable,
                 labels: Union[torch.Tensor, list],
                 unwrapped=False,
                 state_dict=None,
                 invert=False):
        if state_dict is None:
            self.initialize_graph(res, labels, kernel_fn, unwrapped=unwrapped, invert=invert)
            node_pred = self.fit_predict(self._skip_classes)
        else:
            self.load_state(state_dict)
            if 'node_pred' not in state_dict:
                node_pred = self.fit_predict(self._skip_classes)
        return self._data
    
    def __repr__(self):
        msg = f"""
        ClusterGraphConstructor(
            constructor_cfg={self.constructor_cfg},
            training={self.training},
            cc_predictor={self._cc_predictor.__repr__()},
            data={self._data.__repr__()})
        """
        return msg