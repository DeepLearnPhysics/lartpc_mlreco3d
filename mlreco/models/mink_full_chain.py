import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.chain.full_chain import FullChainCNN1
from mlreco.mink.layers.ppn import PPNLoss
from mlreco.mink.chain.factories import *
from collections import defaultdict

from mlreco.models.chain.full_cnn import EdgeFeatureNet
from mlreco.models.cluster_gnn import EdgeChannelLoss
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_node_gnn import NodeTypeLoss

from mlreco.utils.gnn.cluster import (
    form_clusters,
    get_cluster_batch,
    get_cluster_label, 
    get_cluster_points_label, 
    get_cluster_directions
)

from mlreco.utils.gnn.network import (
    complete_graph, 
    delaunay_graph, 
    mst_graph, 
    knn_graph, 
    bipartite_graph, 
    inter_cluster_distance, 
    get_fragment_edges
)

from mlreco.utils.gnn.evaluation import (
    edge_assignment, 
    edge_assignment_from_graph
)

from mlreco.models.cluster_node_gnn import NodeTypeLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

# Node Encoders
from mlreco.models.gnn.factories import (
    node_encoder_construct, 
    edge_encoder_construct, 
    edge_model_construct
)

from pprint import pprint


def construct_edge_features_batch(x, ids):
    device = x.device
    e, edge_indices = [], []
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if i != j:
                edge_indices.append([ids[i], ids[j]])
                e.append(torch.cat([xi, xj]))
    e = torch.stack(e)
    edge_indices = torch.Tensor(edge_indices).to(device, dtype=torch.long)
    return e, edge_indices

def construct_edge_features(x, batch):
    
    edge_features, edge_indices = [], []
    ids = np.arange(len(batch))

    for b in np.unique(batch):
        mask = batch == b
        batch_ids = ids[mask]
        nodes = x[mask]
        e, ei = construct_edge_features_batch(nodes, batch_ids)
        edge_features.append(e)
        edge_indices.append(ei)
    edge_features = torch.cat(edge_features, dim=0)
    # print(edge_indices)
    edge_indices = torch.cat(edge_indices, dim=0)
    return edge_features, edge_indices.t()


class FullChain(nn.Module):

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()
        self.model_cfg = cfg[name]
        # print('-------------Full Chain Model Config----------------')
        # pprint(self.model_cfg)
        self.net = chain_construct(self.model_cfg['name'], self.model_cfg)

        self.seg_F = self.model_cfg.get('seg_features', 16)
        self.ins_F = self.model_cfg.get('ins_features', 16)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.embedding_dim = self.model_cfg.get('embedding_dim', 3)
        self.sigma_dim = self.model_cfg.get('sigma_dim', 1)
        self.node_min_size = self.model_cfg.get('node_min_size', -1)

        self.fragment_col = self.model_cfg.get('fragment_col', 5)
        self.group_col = self.model_cfg.get('group_col', 6)

        node_F = self.model_cfg['edge_model']['node_feats']
        edge_F = self.model_cfg['edge_model']['edge_feats']
        # Node and Edge Encoder
        self.node_encoder1 = node_encoder_construct(self.model_cfg)
        self.edge_net1 = EdgeFeatureNet(node_F * 2, edge_F)
        # print(self.edge_net1)
        # self.edge_encoder1 = edge_encoder_construct(self.model_cfg)

        self.node_encoder2 = node_encoder_construct(self.model_cfg)
        self.edge_net2 = EdgeFeatureNet(node_F * 2, edge_F)
        # print(self.edge_net2)
        # self.edge_encoder2 = edge_encoder_construct(self.model_cfg)

        self.gnn1 = edge_model_construct(self.model_cfg)
        self.gnn2 = edge_model_construct(self.model_cfg, model_name='edge_model_types')
        self.gnn3 = edge_model_construct(self.model_cfg)

        self.skip_gnn = self.model_cfg.get('skip_gnn', True)

        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        device = input[0].device
        out = defaultdict(list)

        for igpu, x in enumerate(input):
            input_data = x[:, :5]

            # CNN Phase
            res = self.net(input_data)
            for key, val in res.items():
                out[key].append(val[igpu])
            # print([k for k in ppn_output.keys()])
            # print([k for k in res.keys()])
            segmentation = res['segmentation'][igpu]
            embeddings = res['embeddings'][igpu]
            margins = res['margins'][igpu]

            if self.skip_gnn:
                continue
            # Shower Grouping and Primary Prediction
            highE = x[x[:, -1].long() != 4]

            frags = form_clusters(highE, min_size=self.node_min_size, 
                                  column=self.fragment_col, batch_index=0)
            frags = [c.cpu().numpy() for c in frags]
            batch_ids1 = get_cluster_batch(highE, frags, batch_index=0)
            xbatch1 = torch.tensor(batch_ids1, device=device)

            # print(edge_index1)
            start = time.time()
            x1 = self.node_encoder1(highE, frags)
            e1, edge_index1 = construct_edge_features(x1, batch_ids1)
            e1 = self.edge_net1(e1)

            gnn_output1 = self.gnn1(x1, edge_index1, e1, xbatch1)

            node_pred = gnn_output1['node_pred'][igpu]
            edge_pred = gnn_output1['edge_pred'][igpu]
            edge_index1 = edge_index1.cpu().numpy()

            # Divide the output out into different arrays (one per batch)
            _, counts = torch.unique(highE[:,0], return_counts=True)
            vids = np.concatenate([np.arange(n.item()) for n in counts])
            cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids1, return_counts=True)[1]])
            bcids = [np.where(batch_ids1 == b)[0] for b in range(len(counts))]
            beids = [np.where(batch_ids1[edge_index1[0]] == b)[0] for b in range(len(counts))]

            node_pred = [node_pred[b] for b in bcids]
            edge_pred = [edge_pred[b] for b in beids]
            edge_index1 = [cids[edge_index1[:,b]].T for b in beids]
            frags = [np.array([vids[c] for c in np.array(frags)[b]]) for b in bcids]

            out['primary_node_pred'].append(node_pred)
            out['grouping_edge_pred'].append(edge_pred)
            out['grouping_clusts'].append(frags)
            out['grouping_edge_index'].append(edge_index1)

            # print(node_primary_pred)
            # print(edge_grouping_pred)

            # Particle Type and Flow Prediciton

            groups = form_clusters(highE, self.node_min_size, self.group_col, batch_index=0)
            groups = [c.cpu().numpy() for c in groups]
            batch_ids2 = get_cluster_batch(highE, groups, batch_index=0)

            x2 = self.node_encoder2(highE, groups)
            e2, edge_index2_ = construct_edge_features(x2, batch_ids2)
            e2 = self.edge_net2(e2)
            xbatch2 = torch.tensor(batch_ids2, device=device)

            gnn_output2 = self.gnn2(x2, edge_index2_, e2, xbatch2)
            node_pred2 = gnn_output2['node_pred'][igpu]
            edge_pred2 = gnn_output2['edge_pred'][igpu]
            edge_index2 = edge_index2_.cpu().numpy()

            # Divide the output out into different arrays (one per batch)
            # _, counts = torch.unique(highE[:,0], return_counts=True)
            # vids = np.concatenate([np.arange(n.item()) for n in counts])
            cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids2, return_counts=True)[1]])
            bcids = [np.where(batch_ids2 == b)[0] for b in range(len(counts))]
            beids = [np.where(batch_ids2[edge_index2[0]] == b)[0] for b in range(len(counts))]

            node_pred2 = [node_pred2[b] for b in bcids]
            edge_pred2 = [edge_pred2[b] for b in beids]
            edge_index2 = [cids[edge_index2[:,b]].T for b in beids]
            groups = [np.array([vids[c] for c in np.array(groups)[b]]) for b in bcids]

            out['type_node_pred'].append(node_pred2)
            out['flow_edge_pred'].append(edge_pred2)
            out['flow_edge_index'].append(edge_index2)
            out['group_clusts'].append(groups)

            gnn_output3 = self.gnn3(x2, edge_index2_, e2, xbatch2)
            edge_pred3 = gnn_output3['edge_pred'][igpu]
            edge_pred3 = [edge_pred3[b] for b in beids]
            out['interaction_edge_pred'].append(edge_pred3)
            end = time.time()
            print("GNN Time = {:.4f}".format(end - start))

        return out


class ChainLoss(nn.Module):

    def __init__(self, cfg, name='segmentation_loss'):
        super(ChainLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.loss_config = cfg['full_chain']
        self.ppn_type_weight = self.loss_config['ppn'].get('ppn_type_weight', 1.0)
        self.ppn_loss_weight = self.loss_config['ppn'].get('ppn_loss_weight', 1.0)
        # Semantic Segmentation Loss

        # PPN Loss
        self.ppn_loss = PPNLoss(self.loss_config)

        # Dense Clustering Loss

        # Shower Grouping Loss
        self.fragment_grouping_loss = EdgeChannelLoss(cfg, name='grouping_loss')
        
        # Primary Identification Loss
        self.primary_node_loss = NodeChannelLoss(cfg, name='primary_loss')

        # Particle Flow and Type Loss
        self.node_type_loss = NodeTypeLoss(cfg, name='flow_type_loss')
        self.flow_loss = EdgeChannelLoss(cfg, name='flow_edge_loss')
        self.interaction_loss = EdgeChannelLoss(cfg, name='interaction_loss')


    def forward(self, outputs, segment_label, particles_label, graph, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        segmentation = outputs['segmentation']
        assert len(segmentation) == len(segment_label)
        batch_ids = [d[:, 0] for d in segment_label]
        highE = [t[t[:, -1].long() != 4] for t in segment_label]
        total_loss = 0
        total_acc = 0
        count = 0

        loss, accuracy = 0, []
        res = {}
        # Semantic Segmentation Loss
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = segment_label[i][:, -1][batch_index]
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
                acc = (predicted_labels == event_label).sum().item() / \
                    float(predicted_labels.nelement())
                total_acc += acc
                count += 1

        loss_seg = total_loss / count
        acc_seg = total_acc / count
        res['loss_seg'] = float(loss_seg)
        res['acc_seg'] = float(acc_seg)
        loss += loss_seg
        accuracy.append(acc_seg)

        # PPN Loss
        # ppn_results = self.ppn_loss(outputs, segment_label, particles_label)
        # loss += ppn_results['ppn_loss'] * self.ppn_loss_weight
        # loss += ppn_results['loss_type'] * self.ppn_type_weight
        # accuracy.append(float(ppn_results['ppn_acc']))
        # accuracy.append(float(ppn_results['acc_ppn_type']))
        # res['ppn_loss'] = float(ppn_results['ppn_loss'] * self.ppn_loss_weight)
        # res['ppn_type_loss'] = float(ppn_results['loss_type'] * self.ppn_type_weight)
        # res['ppn_acc'] = ppn_results['ppn_acc']
        # res['ppn_type_acc'] = ppn_results['acc_ppn_type']

        # Fragment Loss
        gnn1_output = {
            'edge_pred': outputs['grouping_edge_pred'],
            'edge_index': outputs['grouping_edge_index'],
            'node_pred': outputs['primary_node_pred'],
            'clusts': outputs['grouping_clusts']
        }

        fragment_grouping_results = self.fragment_grouping_loss(
            gnn1_output, highE, graph)


        loss += fragment_grouping_results['loss']
        accuracy.append(float(fragment_grouping_results['accuracy']))
        res['fragment_loss'] = float(fragment_grouping_results['loss'])
        res['fragment_acc'] = float(fragment_grouping_results['accuracy'])

        primary_node_results = self.primary_node_loss(gnn1_output, highE)
        res['primary_node_loss'] = float(primary_node_results['loss'])
        accuracy.append(float(primary_node_results['accuracy']))
        res['primary_node_acc'] = float(primary_node_results['accuracy'])
        loss += primary_node_results['loss']

        gnn2_output = {
            'edge_pred': outputs['flow_edge_pred'],
            'node_pred': outputs['type_node_pred'],
            'edge_index': outputs['flow_edge_index'],
            'clusts': outputs['group_clusts']
        }

        pdg_results = self.node_type_loss(gnn2_output, highE)
        flow_results = self.flow_loss(gnn2_output, highE, graph)
        res['pdg_loss']  = float(pdg_results['loss'])
        res['pdg_acc'] = float(pdg_results['accuracy'])
        loss += pdg_results['loss']
        res['flow_loss'] = float(flow_results['loss'])
        res['flow_acc'] = float(flow_results['accuracy'])
        loss += flow_results['loss']

        accuracy.append(float(pdg_results['accuracy']))
        accuracy.append(float(flow_results['accuracy']))

        interaction_results = self.interaction_loss(gnn2_output, highE, graph)
        res['interaction_loss'] = float(interaction_results['loss'])
        res['interaction_acc'] = float(interaction_results['accuracy'])
        loss += interaction_results['loss']
        accuracy.append(float(interaction_results['accuracy']))

        accuracy = sum(accuracy) / len(accuracy)

        res['loss'] = loss
        res['accuracy'] = accuracy
    
        pprint(res)

        return res
