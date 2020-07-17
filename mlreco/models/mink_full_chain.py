import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.chain.full_chain import FullChainCNN1
from mlreco.mink.chain.factories import *
from collections import defaultdict

from mlreco.utils.gnn.cluster import get_cluster_label, get_cluster_points_label, get_cluster_directions
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, knn_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph
from mlreco.models.cluster_node_gnn import NodeTypeLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

# Node Encoders
from mlreco.models.gnn.factories import node_encoder_construct, edge_encoder_construct, edge_model_construct

from pprint import pprint


def get_cluster_batch(data, clusts):
    """
    Function that returns the batch ID of each cluster.
    This should be unique for each cluster, assert that it is.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of batch IDs
    """
    labels = []
    for c in clusts:
        assert len(data[c,0].unique()) == 1
        labels.append(int(data[c[0],0].item()))

    return np.array(labels)


def form_clusters(data, min_size=-1, column=5):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up each of the clusters in the input tensor.

    Args:
        data (np.ndarray): (N,6-10) [x, y, z, batchid, value, id(, groupid, intid, nuid, shape)]
        min_size (int)   : Minimal cluster size
        column (int)     : Specifies on which column to base the clusters
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    clusts = []
    for b in data[:, 0].unique():
        binds = torch.nonzero(data[:, 0] == b).flatten()
        for c in data[binds,column].unique():
            # Skip if the cluster ID is -1 (not defined)
            if c < 0:
                continue
            clust = torch.nonzero(data[binds,column] == c).flatten()
            if len(clust) < min_size:
                continue
            clusts.append(binds[clust])

    return clusts


class FullChain(nn.Module):

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()
        self.model_cfg = cfg[name]
        pprint(self.model_cfg)
        self.net = chain_construct(self.model_cfg['name'], self.model_cfg)

        self.seg_F = self.model_cfg.get('seg_features', 16)
        self.ins_F = self.model_cfg.get('ins_features', 16)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.embedding_dim = self.model_cfg.get('embeddimg_dim', 3)
        self.sigma_dim = self.model_cfg.get('sigma_dim', 1)
        self.node_min_size = self.model_cfg.get('node_min_size', -1)

        self.fragment_col = self.model_cfg.get('fragment_col', 5)
        self.group_col = self.model_cfg.get('group_col', 6)

        self.segmentation = ME.MinkowskiLinear(self.seg_F, self.num_classes)
        self.embedding = ME.MinkowskiLinear(
            self.ins_F, self.embedding_dim + self.sigma_dim)

        pprint(self.model_cfg)
        # Node and Edge Encoder
        self.node_encoder1 = node_encoder_construct(self.model_cfg)
        self.edge_encoder1 = edge_encoder_construct(self.model_cfg)

        self.node_encoder2 = node_encoder_construct(self.model_cfg)
        self.edge_encoder2 = edge_encoder_construct(self.model_cfg)

        self.gnn1 = edge_model_construct(self.model_cfg)
        self.gnn2 = edge_model_construct(self.model_cfg, model_name='edge_model_types')
        self.gnn3 = edge_model_construct(self.model_cfg)

        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        device = input[0].device
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            input_data = x[:, :5]

            # CNN Phase
            res = self.net(input_data)
            segmentation = self.segmentation(res['seg_features'][0])
            embeddings = self.embedding(res['ins_features'][0])
            out['segmentation'].append(segmentation.F)
            out['embeddings'].append(embeddings.F[:, :self.embedding_dim])
            out['margins'].append(embeddings.F[:, self.embedding_dim:])

            # Shower Grouping and Primary Prediction
            highE = x[x[:, -1] != 4]

            frags = form_clusters(highE, self.node_min_size, self.fragment_col)
            frags = [c.cpu().numpy() for c in frags]
            batch_ids1 = get_cluster_batch(highE, frags)
            edge_index1 = complete_graph(batch_ids1, None, -1)

            x1 = self.node_encoder1(highE, frags)
            e1 = self.edge_encoder1(highE, frags, edge_index1)

            index1 = torch.tensor(edge_index1, device=device, dtype=torch.long)
            xbatch1 = torch.tensor(batch_ids1, device=device)

            gnn_output1 = self.gnn1(x1, index1, e1, xbatch1)

            node_primary_pred = gnn_output1['node_pred'][0]
            edge_grouping_pred = gnn_output1['edge_pred'][0]

            print(node_primary_pred)
            print(edge_grouping_pred)

            # Particle Type and Flow Prediciton

            groups = form_clusters(highE, self.node_min_size, self.group_col)
            groups = [c.cpu().numpy() for c in groups]
            batch_ids2 = get_cluster_batch(highE, groups)
            edge_index2 = complete_graph(batch_ids2, None, -1)

            x2 = self.node_encoder2(highE, groups)
            e2 = self.edge_encoder2(highE, groups, edge_index2)

            index2 = torch.tensor(edge_index2, device=device, dtype=torch.long)
            xbatch2 = torch.tensor(batch_ids2, device=device)

            gnn_output2 = self.gnn2(x2, index2, e2, xbatch2)

            node_type_pred = gnn_output2['node_pred'][0]
            edge_flow_pred = gnn_output2['edge_pred'][0]

            print(node_type_pred)
            print(edge_flow_pred)

            gnn_output3 = self.gnn3(x2, index2, e2, xbatch2)

            edge_interaction_pred = gnn_output3['edge_pred'][0]

            # Interaction Grouping
        # print(out)
        return out


class ChainLoss(nn.Module):

    def __init__(self, cfg, name='segmentation_loss'):
        super(ChainLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        # Semantic Segmentation Loss

        # PPN Loss

        # Dense Clustering Loss

        # Shower Grouping Loss

        # Primary Identification Loss

        # Particle Type Loss

        # Einit Loss

        # Flow Reconstruction Loss

    def forward(self, outputs, label, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        segmentation = outputs['segmentation']
        assert len(segmentation) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        batch_ids = [d[:, 0] for d in label]
        total_loss = 0
        total_acc = 0
        count = 0
        # Loop over GPUS
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = label[i][:, -1][batch_index]
                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                if weight is not None:
                    event_weight = weight[i][batch_index]
                    event_weight = torch.squeeze(event_weight, dim=-1).float()
                    total_loss += torch.mean(loss_seg * event_weight)
                else:
                    total_loss += torch.mean(loss_seg)
                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() / float(predicted_labels.nelement())
                total_acc += acc
                count += 1

        return {
            'accuracy': total_acc/count,
            'loss': total_loss/count
        }
