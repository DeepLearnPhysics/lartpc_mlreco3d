import numpy as np
import pandas as pd
import torch

from mlreco.main_funcs import process_config, train, inference
from mlreco.utils.metrics import *
from mlreco.trainval import trainval
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
from mlreco.main_funcs import cycle

from pprint import pprint

import networkx as nx
from torch_cluster import knn_graph, radius_graph
import time
from torch_geometric.data import Data, Batch

class OccuSegPredictor:

    def __init__(self, cfg):
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
    def get_edge_truth(edge_indices: torch.Tensor, labels: torch.Tensor):
        '''

            - edge_indices: 2 x E
            - labels: N
        '''
        u = labels[edge_indices[0, :]]
        v = labels[edge_indices[1, :]]
        return (u == v).long()


    def fit_predict(self, coords: torch.Tensor,
                          sp_emb: torch.Tensor,
                          ft_emb: torch.Tensor,
                          cov: torch.Tensor,
                          occ=None, cluster_all=True):

        edge_indices = self.graph_constructor(coords, **self.kwargs)
        w = self.get_edge_weight(sp_emb, ft_emb, cov, edge_indices, occ=occ.squeeze(), eps=self.eps)
        edge_indices = edge_indices.T[w > self.ths].T
        edges = [(e[0], e[1], w[i].item()) \
            for i, e in enumerate(edge_indices.cpu().numpy())]
        w = w[w > self.ths]
        G = nx.Graph()
        G.add_nodes_from(np.arange(coords.shape[0]))
        G.add_weighted_edges_from(edges)
        pred = -np.ones(coords.shape[0], dtype=np.int32)
        for i, comp in enumerate(nx.connected_components(G)):
            x = np.asarray(list(comp))
            pred[x] = i
        return pred, edge_indices, w


class GraphDataConstructor:

    def __init__(self, predictor, cfg):
        self.predictor = predictor
        self.seg_col = cfg.get('seg_col', -1)
        self.cluster_col = cfg.get('cluster_col', 5)

    def construct_graph(self, coords: torch.Tensor,
                              edge_weights: torch.Tensor,
                              edge_index: torch.Tensor,
                              feats: torch.Tensor):

        graph_data = Data(x=feats, edge_index=edge_index, edge_attr=edge_weights, pos=coords)
        return graph_data

    def construct_batched_graphs(self, res):

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

                pred, edge_index, w = self.predictor.fit_predict(
                    coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())

                data = self.construct_graph(coords_class, w, edge_index, features_class)
                data_list.append(data)

        graph_batch = Batch().from_data_list(data_list)
        return graph_batch


    def construct_batched_graphs_with_labels(self, res, labels: torch.Tensor):
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
            labels_batch = labels[mask].int()

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

                pred, edge_index, w = self.predictor.fit_predict(
                    coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())

                data = self.construct_graph(coords_class, w, edge_index, features_class)
                truth = self.predictor.get_edge_truth(edge_index, frag_labels)
                data.edge_truth = truth
                data_list.append(data)

        graph_batch = Batch().from_data_list(data_list)
        return graph_batch



def main_loop(inference_cfg, **kwargs):

    start_index = kwargs.get('start_index', 0)
    end_index = kwargs.get('end_index', 20000)
    event_list = list(range(start_index, end_index))
    loader = loader_factory(inference_cfg, event_list=event_list)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()
    for m in Trainer._net.modules():
        m.eval()
    Trainer._net.eval()

    output = []

    inference_cfg['trainval']['iterations'] = len(event_list)
    iterations = inference_cfg['trainval']['iterations']

    cluster_label_col = kwargs.get('cluster_label_col', 5)
    batch_col = kwargs.get('batch_col', 3)
    predictor_cfg = kwargs.get('predictor_cfg', {})

    counts = 0

    predictor = OccuSegPredictor(**predictor_cfg)

    while counts < len(event_list):

        start = time.time()
        data_blob, res = Trainer.forward(dataset)
        event_id = data_blob['index']
        end = time.time()

        device = torch.cuda.current_device()
        # Results
        segmentation = torch.Tensor(res['segmentation'][0]).cuda()
        covariance = torch.Tensor(res['covariance'][0]).cuda()
        occupancy = torch.Tensor(res['occupancy'][0]).cuda()
        sp_embeddings = torch.Tensor(res['spatial_embeddings'][0]).cuda()
        ft_embeddings = torch.Tensor(res['feature_embeddings'][0]).cuda()

        # Labels
        input_data = torch.Tensor(data_blob['input_data'][0]).cuda()
        labels = torch.Tensor(data_blob['cluster_label'][0]).cuda()
        batch_index = input_data[:, batch_col].int()
        nbatches = len(torch.unique(batch_index))
        forward_time_per_image = float(end - start) / float(nbatches)

        for i, bidx in enumerate(torch.unique(batch_index)):
            seg_batch = segmentation[batch_index == bidx]
            cov_batch = covariance[batch_index == bidx]
            occ_batch = occupancy[batch_index == bidx]
            sp_batch = sp_embeddings[batch_index == bidx]
            ft_batch = ft_embeddings[batch_index == bidx]
            labels_batch = labels[batch_index == bidx]
            idx = event_id[i]
            print("Counts: {}  |   Index: {}".format(counts, idx))
            counts += 1

            for c in (torch.unique(labels_batch[:, -1]).int()):
                if int(c) == 4:
                    continue
                class_mask = labels_batch[:, -1] == c
                seg_class = seg_batch[class_mask]
                cov_class = cov_batch[class_mask]
                occ_class = occ_batch[class_mask]
                sp_class = sp_batch[class_mask]
                ft_class = ft_batch[class_mask]
                coords_class = labels_batch[:, :3][class_mask]
                frag_labels = labels_batch[:, cluster_label_col][class_mask]

                # Run Post-Processing
                start = time.time()
                pred, _, _ = predictor.fit_predict(coords_class, sp_class, ft_class, cov_class, occ=occ_class.squeeze())
                end = time.time()
                post_time = float(end-start)

                # Compute Metrics
                purity, efficiency = purity_efficiency(pred, frag_labels.cpu().numpy())
                fscore = 2 * (purity * efficiency) / (purity + efficiency)
                ari = ARI(pred, frag_labels.cpu().numpy())
                sbd = SBD(pred, frag_labels.cpu().numpy())
                true_num_clusters = len(torch.unique(frag_labels))
                pred_num_clusters = len(np.unique(pred))
                row = (idx, c, ari, purity, efficiency, fscore, sbd, \
                    true_num_clusters, pred_num_clusters, forward_time_per_image, post_time)
                output.append(row)
                print("Class = {0}  |  ARI = {1:.4f}  |  SBD = {2:.4f}".format(int(c), ari, sbd))
                # print("LL = ", ll)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
        'Purity', 'Efficiency', 'FScore', 'SBD', 'true_num_clusters',
        'pred_num_clusters', 'forward_time', 'post_time'])
    return output
