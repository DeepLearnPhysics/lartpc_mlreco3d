from typing import Union, Callable, Tuple, List

# Silencing this warning about unclassified (-1) voxels
# /usr/local/lib/python3.8/dist-packages/sklearn/neighbors/_classification.py:601: 
# UserWarning: Outlier label -1 is not in training classes. 
# All class probabilities of outliers will be assigned with 0.
# warnings.warn('Outlier label {} is not in training '
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch

import networkx as nx
from torch_cluster import knn_graph, radius_graph

from mlreco.utils.metrics import *
from mlreco.utils.cluster.graph_batch import GraphBatch
from torch_geometric.data import Data as GraphData
from scipy.special import expit
from .helpers import *


class ClusterGraphConstructor:
    '''
    Parametric Graph-SPICE clustering

    Parametric GDC includes a bilinear layer to predict edge weights,
    given pairs of node features.
    '''
    def __init__(self, constructor_cfg : dict,
                       graph_batch : GraphBatch = None,
                       graph_info : pd.DataFrame = None,
                       batch_col : int = 0,
                       training : bool = False):

        # Input Data/Label conventions
        self.seg_col = constructor_cfg.get('seg_col', -1)
        self.cluster_col = constructor_cfg.get('cluster_col', 5)
        self.batch_col = batch_col
        self.training = training # Default mode is evaluation, this is set otherwise when we initialize

        self.constructor_cfg = constructor_cfg

        # Initial Neighbor Graph Construction Mode
        mode = constructor_cfg.get('mode', 'knn')
        if mode == 'knn':
            self._init_graph = knn_graph
        elif mode == 'radius':
            self._init_graph = radius_graph
        elif mode == 'knn_sklearn':
            self._init_graph = knn_sklearn
        else:
            raise ValueError('''Mode {} is not supported for initial
                graph construction!'''.format(mode))

        # Clustering Algorithm Parameters
        self.ths = constructor_cfg.get('edge_cut_threshold', 0.0) # Prob values 0-1
        if self.training:
            self.ths = 0.0
        # print("Edge Threshold Probability Score = ", self.ths)
        self.kwargs = constructor_cfg.get('cluster_kwargs', dict(k=5))
        # Radius within which orphans get assigned to neighbor cluster
        self._orphans_radius = constructor_cfg.get('orphans_radius', 1.9)
        self._orphans_iterate = constructor_cfg.get('orphans_iterate', True)
        self._orphans_cluster_all = constructor_cfg.get('orphans_cluster_all', True)
        self.use_cluster_labels = constructor_cfg.get('use_cluster_labels', True)

        # GraphBatch containing graphs per semantic class.
        if graph_batch is None:
            self._graph_batch = GraphBatch()
        else:
            assert graph_info is not None
            self._info = graph_info
            self._graph_batch = graph_batch
            self._num_total_nodes = self._graph_batch.x.shape[0]
            self._node_dim = self._graph_batch.x.shape[1]
            self._num_total_edges = self._graph_batch.edge_index.shape[1]

        self._edge_pred = None
        self._node_pred = None


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


    def _initialize_graph_unwrapped(self, res: dict,
                                          labels: list):
        '''
        CGC initializer for unwrapped tensors.
        (see initialize_graph for functionality)
        '''
        features = res['hypergraph_features']
        batch_indices = res['coordinates'][:,0].int()
        coordinates = res['coordinates'][:,1:4]
        assert len(features) != len(labels)
        assert len(features) != torch.unique(batch_indices).shpae[0]
        data_list = []
        graph_id = 0
        index = 0

        for i, bidx in enumerate(torch.unique(batch_indices)):
            coords_batch = coordinates[i]
            features_batch = features[i]
            labels_batch = labels[i]

            for c in torch.unique(labels_batch[:, self.seg_col]):
                class_mask = labels_batch[:, self.seg_col] == c
                coords_class = coords_batch[class_mask]
                features_class = features_batch[class_mask]

                edge_indices = self._init_graph(coords_class, **self.kwargs)
                data = GraphData(x=features_class,
                                 pos=coords_class,
                                 edge_index=edge_indices)
                graph_id_key = dict(Index=index,
                                    BatchID=int(bidx),
                                    SemanticID=int(c),
                                    GraphID=graph_id)
                graph_id += 1
                self._info.append(graph_id_key)

                frag_labels = labels_batch[class_mask][:, self.cluster_col]
                truth = self.get_edge_truth(edge_indices, frag_labels)
                data.edge_truth = truth
                data_list.append(data)
            index += 1

        self._info = pd.DataFrame(self._info)
        self.data_list = data_list
        self._graph_batch = self._graph_batch.from_data_list(data_list)
        self._num_total_nodes = self._graph_batch.x.shape[0]
        self._node_dim = self._graph_batch.x.shape[1]
        self._num_total_edges = self._graph_batch.edge_index.shape[1]


    def initialize_graph(self, res : dict,
                               labels: Union[torch.Tensor, list],
                               unwrapped=False):
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
        if unwrapped:
            return self._initialize_graph_unwrapped(res, labels)

        features = res['hypergraph_features'][0]
        batch_indices = res['coordinates'][0][:,0].int()
        coordinates = res['coordinates'][0][:,1:4]
        data_list = []

        graph_id = 0
        index = 0

        self._info = []

        for i, bidx in enumerate(torch.unique(batch_indices)):
            mask = batch_indices == bidx
            coords_batch = coordinates[mask]
            features_batch = features[mask]
            labels_batch = labels[mask].int()

            for c in torch.unique(labels_batch[:, self.seg_col]):
                class_mask = labels_batch[:, self.seg_col] == c
                coords_class = coords_batch[class_mask]
                features_class = features_batch[class_mask]

                edge_indices = self._init_graph(coords_class, **self.kwargs)
                data = GraphData(x=features_class,
                                 pos=coords_class,
                                 edge_index=edge_indices)
                graph_id_key = dict(BatchID=int(bidx),
                                    Index=index,
                                    SemanticID=int(c),
                                    GraphID=graph_id)
                graph_id += 1
                self._info.append(graph_id_key)
                if self.use_cluster_labels:
                    frag_labels = labels_batch[class_mask][:, self.cluster_col]
                    truth = self.get_edge_truth(edge_indices, frag_labels)
                    data.node_truth = frag_labels
                    data.edge_truth = truth
                data_list.append(data)
            index += 1

        self._info = pd.DataFrame(self._info)
        self.data_list = data_list
        self._graph_batch = GraphBatch()
        self._graph_batch = self._graph_batch.from_data_list(data_list)
        self._num_total_nodes = self._graph_batch.x.shape[0]
        self._node_dim = self._graph_batch.x.shape[1]
        self._num_total_edges = self._graph_batch.edge_index.shape[1]


    def replace_state(self, result, prefix='', unwrapped=False):
        concat = torch.cat if isinstance(result[prefix+'features'][0], torch.Tensor) else np.concatenate
        if unwrapped:
            batch_size = len(result[prefix+'features'])
            data_list = []
            for i in range(batch_size):
                data = GraphData(x = torch.Tensor(result[prefix+'features'][i]).float(),
                                #  batch = result[prefix+'coordinates'][:,0],
                                 pos = torch.Tensor(result[prefix+'coordinates'][i][:,1:4]).float(),
                                 edge_index = torch.Tensor(result[prefix+'edge_index'][i].T).long(),
                                 edge_attr = torch.Tensor(result[prefix+'edge_score'][i]).float(),
                                 edge_truth = torch.Tensor(result[prefix+'edge_truth'][i]).long())
                data_list.append(data)
            graph = GraphBatch.from_data_list(data_list)
            if not isinstance(result[prefix+'features'][0], torch.Tensor):
                graph.x = graph.x.numpy()
                graph.pos = graph.pos.numpy()
                graph.edge_index = graph.edge_index.numpy()
                graph.edge_attr = graph.edge_attr.numpy()
                if hasattr(graph, 'edge_truth'):
                    graph.edge_truth = graph.edge_truth.numpy()
        else:
            graph = GraphBatch(x = concat(result[prefix+'features']),
                               batch = concat(result[prefix+'coordinates'])[:,0],
                               pos = concat(result[prefix+'coordinates'])[:,1:4],
                               edge_index = concat(result[prefix+'edge_index']).T,
                               edge_attr = concat(result[prefix+'edge_score']))
            if prefix+'edge_truth' in result:
                 graph.edge_truth = concat(result[prefix+'edge_truth'])
        self._graph_batch = graph
        self._num_total_nodes = self._graph_batch.x.shape[0]
        self._node_dim = self._graph_batch.x.shape[1]
        self._num_total_edges = self._graph_batch.edge_index.shape[1]
        self._info = result[prefix+'graph_info']


    def _set_edge_attributes(self, kernel_fn : Callable):
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
            if self._graph_batch.edge_index.shape[1] > 0:
                edge_attr = kernel_fn(
                    self._graph_batch.x[self._graph_batch.edge_index[0, :]],
                    self._graph_batch.x[self._graph_batch.edge_index[1, :]])
                w = edge_attr.squeeze()
            else:
                w = torch.empty((0,), device=self._graph_batch.edge_index.device)
            self._graph_batch.edge_attr = w
            self._graph_batch.add_edge_features(w, 'edge_attr')


    def get_batch_and_class(self, entry):
        df = self._info.query('GraphID == {}'.format(entry))
        assert df.shape[0] == 1
        batch_id = df['BatchID'].item()
        semantic_id = df['SemanticID'].item()
        return batch_id, semantic_id


    def get_entry(self, batch_id, semantic_id):
        df = self._info.query(
            'BatchID == {} and SemanticID == {}'.format(batch_id, semantic_id))
        assert df.shape[0] < 2
        if df.shape[0] == 0:
            raise ValueError('''Event ID: {} and Class Label: {} does not
                exist in current batch'''.format(batch_id, semantic_id))
            return None
        else:
            entry_num = df['GraphID'].item()
            return entry_num


    def get_graph(self, batch_id, semantic_id):
        '''
        Retrieve single graph from GraphBatch object by batch and semantic id.

        INPUTS:
            - event_id: Event ID (Index)
            - semantic_id: Semantic Class (0-4)

        RETURNS:
            - Subgraph corresponding to class [semantic_id] and event [event_id]
        '''
        entry_num = self.get_entry(batch_id, semantic_id)
        return self._graph_batch.get_example(entry_num)


    def fit_predict_one(self, entry,
                        min_points=0,
                        invert=False) -> Tuple[np.ndarray, nx.Graph]:
        '''
        Generate predicted fragment cluster labels for single subgraph.

        INPUTS:
            - entry number
            - min_points: minimum voxel count required to assign
            unique cluster label during first pass.
            - remainder_alg: algorithm used to handle orphans

        Returns:
            - pred: predicted cluster labels.

        '''
        # min_points is not meant to be used at train time
        # (it defines orphans to be assigned)
        if self.training:
            # raise Exception("Please set min_points: 0 in GraphSpice config at training time.")
            min_points = 0

        subgraph = self._graph_batch.get_example(entry)
        num_nodes = subgraph.num_nodes
        G = nx.Graph()
        G.add_nodes_from(np.arange(num_nodes))

        # Drop edges with low predicted probability score
        edges = subgraph.edge_index.T
        edge_logits = subgraph.edge_attr
        if isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
            edge_logits = edge_logits.detach().cpu().numpy()
        edge_probs = expit(edge_logits)
        if invert:
            pos_edges = edges[edge_probs < self.ths]
            pos_probs = edge_probs[edge_probs < self.ths]
        else:
            pos_edges = edges[edge_probs >= self.ths]
            pos_probs = edge_probs[edge_probs >= self.ths]
        pos_edges = [(e[0], e[1], w) for e, w in zip(pos_edges, pos_probs)]
        G.add_weighted_edges_from(pos_edges)
        pred = -np.ones(num_nodes, dtype=np.int32)
        orphan_mask = np.zeros(num_nodes, dtype=bool)
        for i, comp in enumerate(nx.connected_components(G)):
            x = np.asarray(list(comp))
            pred[x] = i
            if len(comp) < min_points:
                orphan_mask[x] = True

        # Assign orphans
        G.pos = subgraph.pos
        if isinstance(G.pos, torch.Tensor):
            G.pos = G.pos.detach().cpu().numpy()
        if not orphan_mask.all():
            n_orphans = 0
            while orphan_mask.any() and (n_orphans != np.sum(orphan_mask)):
                orphans = G.pos[orphan_mask]
                n_orphans = len(orphans)
                assigner = RadiusNeighborsAssigner(G.pos[~orphan_mask],
                                                   pred[~orphan_mask].astype(int),
                                                   radius=self._orphans_radius,
                                                   outlier_label=-1)
                orphan_labels = assigner.assign_orphans(orphans)
                valid_mask  = orphan_labels > -1
                new_labels = pred[orphan_mask]
                new_labels[valid_mask] = orphan_labels[valid_mask]
                pred[orphan_mask] = new_labels
                orphan_mask[orphan_mask] = ~valid_mask
                if not self._orphans_iterate: break
        if not self._orphans_cluster_all:
            pred[orphan_mask] = -1

        new_labels, _ = unique_label(pred[pred >= 0])
        pred[pred >= 0] = new_labels

        return pred, G, subgraph


    def fit_predict(self, skip=[], **kwargs):
        '''
        Iterate over all subgraphs and assign predicted labels.
        '''
        skip = set(skip)
        num_graphs = self._graph_batch.num_graphs
        entry_list = [i for i in range(num_graphs) if i not in skip]
        node_pred = -np.ones(self._num_total_nodes, dtype=np.int32)

        pred_data_list = []

        for entry in entry_list:
            pred, G, subgraph = self.fit_predict_one(entry, **kwargs)
            batch = self._graph_batch.batch
            if isinstance(batch, torch.Tensor):
                batch = batch.cpu().numpy()
            batch_index = batch == entry
            pred_data_list.append(GraphData(x=torch.Tensor(pred).long(),
                                            pos=torch.Tensor(G.pos)))
            # node_pred[batch_index] = pred
        self._node_pred = GraphBatch.from_data_list(pred_data_list)
        # self._graph_batch.add_node_features(node_pred, 'node_pred',
        #                                     dtype=torch.long)


    @property
    def node_pred(self):
        return self._node_pred

    @property
    def graph_batch(self):
        if self._graph_batch is None:
            raise('The GraphBatch data has not been initialized yet!')
        return self._graph_batch

    @property
    def info(self):
        '''
        Entry mapping (pd.DataFrame):

            - columns: ['Index', 'BatchID', 'SemanticID', 'GraphID']

        By querying on BatchID and SemanticID, for example, one obtains
        the graph id value (entry) used to access a single subgraph in
        self._graph_batch.
        '''
        return self._info


    def __call__(self, res : dict,
                       kernel_fn : Callable,
                       labels: torch.Tensor):
        '''
        Train time labels include cluster column (default: 5)
        and segmentation column (default: -1)
        Test time labels only include segment column (default: -1)
        '''
        self.initialize_graph(res, labels)
        self._set_edge_attributes(kernel_fn)
        return self._graph_batch


    def __repr__(self):
        out = '''
        ClusterGraphConstructor
        '''
        return out