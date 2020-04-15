import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.modular_gnn import *

from mlreco.models.ppn import PPNLoss
from .cluster_cnn import clustering_loss_construct
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss
from mlreco.models.gnn.losses.grouping import *

from pprint import pprint

class FullChain(nn.Module):

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()
        self.model_config = cfg[name]

        self.full_cnn = FullCNN(cfg)
        self.full_gnn = FullGNN(cfg)

        self.edge_net = EdgeFeatureNet(16, 16)

        self.cnn_freeze = self.model_config.get('cnn_freeze', False)
        self.gnn_freeze = self.model_config.get('gnn_freeze', False)

    def forward(self, input):
        '''
        Forward for full reconstruction chain.

        INPUTS:
            - input (N x 8 Tensor): input_data

        RETURNS:
            - result (tuple of dicts): (cnn_result, gnn_result)
        '''
        # Run all CNN modules.
        result = self.full_cnn(input)

        # UResNet Results
        # embeddings = result['embeddings'][0]
        # margins = result['margins'][0]
        # seediness = result['seediness'][0]
        # segmentation = result['segmentation'][0]
        features_gnn = result['features_gnn'][0]
        embeddings = result['embeddings'][0]

        # Ground Truth Labels
        coords = input[0][:, :4]
        batch_index = input[0][:, 3].unique()
        # semantic_labels = input[0][:, -1]
        fragment_labels = input[0][:, -3]
        group_labels = input[0][:, -2]

        # gnn_input = get_gnn_input(
        #     coords,
        #     features_gnn,
        #     self.edge_net,
        #     fit_predict,
        #     train=True,
        #     embeddings=embeddings,
        #     fragment_labels=fragment_labels,
        #     group_labels=group_labels,
        #     batch_index=batch_index)
        #
        # node_features = gnn_input.x
        # edge_indices = gnn_input.edge_index
        # edge_attr = gnn_input.edge_attr
        # gnn_batch_index = gnn_input.batch
        #
        # gnn_output = self.full_gnn(
        #     node_features, edge_indices, edge_attr, gnn_batch_index)
        #
        # result.update(gnn_output)
        # result['node_batch_labels'] = [gnn_input.batch]
        # result['node_group_labels'] = [gnn_input.node_group_labels]
        # result['centroids'] = [gnn_input.centroids]
        # for key, val in result.items():
        #     print(key, val[0].shape)

        return result


class FullChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(FullChainLoss, self).__init__()
        self.segmentation_loss = torch.nn.CrossEntropyLoss()
        self.ppn_loss = PPNLoss(cfg)
        self.loss_config = cfg['clustering_loss']
        print(self.loss_config)
        self.gnn_loss = self.loss_config.get('gnn_loss', False)

        self.clustering_loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.clustering_loss = clustering_loss_construct(self.clustering_loss_name)
        self.clustering_loss = self.clustering_loss(cfg)
        print(self.clustering_loss)

        self.node_loss = GNNGroupingLoss(cfg)
        self.node_loss_weight = self.loss_config.get('node_loss_weight', 1.0)
        # self.edge_loss = EdgeChannelLoss(cfg)
        self.spatial_size = self.loss_config.get('spatial_size', 768)

        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.ppn_weight = self.loss_config.get('ppn_weight', 0.0)

    def forward(self, out, input_data, ppn_label, ppn_segment_label, graph):
        '''
        Forward propagation for FullChain

        INPUTS:
            - out (dict): result from forwarding three-tailed UResNet, with
            1) segmenation decoder 2) clustering decoder 3) seediness decoder,
            and PPN attachment to the segmentation branch.

            - input_data (list of Tensors): input data tensor of shape N x 10
              In row-index order:
              1. x coordinates
              2. y coordinates
              3. z coordinates
              4. batch indices
              5. energy depositions
              6. fragment labels
              7. group labels
              8. segmentation labels (0-5, includes ghosts)

            - ppn_label (list of Tensors): particle labels for ppn ground truth

            - ppn_segment_label (list of Tensors): semantic labels for feeding
            into ppn network.

            NOTE: <input_data> contains duplicate coordinates, which are
            automatically removed once fed into the input layer of the SCN
            network producing the <result> dictionary. Hence we cannot use
            semantic labels from <input_data> for PPN ground truth, unless we
            also remove duplicates. For now we simply take care of the
            duplicate coordinate issue by importing ppn_segment_label that does
            not contain duplicate labels.

            - graph (list of Tensors): N x 3 tensor of directed edges, with rows
            (parent, child, batch_id)
        '''

        loss = defaultdict(list)
        accuracy = defaultdict(list)

        # Get Ground Truth Information
        coords = input_data[0][:, :4].int()
        segment_label = input_data[0][:, -1]
        fragment_label = input_data[0][:, -3]
        group_label = input_data[0][:, -2]
        batch_idx = coords[:, -1].unique()

        # for key, val in result.items():
        #     print(key, val[0].shape)

        embedding = out['embeddings'][0]
        seediness = out['seediness'][0]
        margins = out['margins'][0]
        # print(ppn_segment_label[0])
        # print(ppn_segment_label[0].shape)
        # PPN Loss.
        # FIXME: This implementation will loop over the batch twice.
        ppn_res = self.ppn_loss(out, ppn_segment_label, ppn_label)
        # print(ppn_res)
        ppn_loss = ppn_res['ppn_loss']

        counts = 0
        groups = 0

        node_group_labels = []

        for bidx in batch_idx:

            batch_mask = coords[:, -1] == bidx
            seg_logits = out['segmentation'][0][batch_mask]
            # print(seg_logits)
            embedding_batch = embedding[batch_mask]
            slabels_batch = segment_label[batch_mask]
            clabels_batch = fragment_label[batch_mask]
            counts += int(len(clabels_batch.unique()))
            groups += int(len(group_label[batch_mask].unique()))
            seed_batch = seediness[batch_mask]
            margins_batch = margins[batch_mask]
            coords_batch = coords[batch_mask] / self.spatial_size

            # Segmentation Loss
            segmentation_loss = self.segmentation_loss(
                seg_logits, slabels_batch.long())
            loss['loss_seg'].append(segmentation_loss)

            # Segmentation Accuracy
            segment_pred = torch.argmax(seg_logits, dim=1).long()
            segment_acc = torch.sum(segment_pred == slabels_batch.long())
            segment_acc = float(segment_acc) / slabels_batch.shape[0]
            accuracy['accuracy'].append(segment_acc)

            # Clustering Loss & Accuracy
            highE_mask = slabels_batch != 4
            slabels_highE = slabels_batch[highE_mask]
            embedding_batch_highE = embedding_batch[highE_mask]
            clabels_batch_highE = clabels_batch[highE_mask]
            seed_batch_highE = seed_batch[highE_mask]
            margins_batch_highE = margins_batch[highE_mask]

            loss_class, acc_class = self.clustering_loss.combine_multiclass(
                embedding_batch_highE, margins_batch_highE,
                seed_batch_highE, slabels_highE, clabels_batch_highE)
            for key, val in loss_class.items():
                loss[key].append(sum(val) / len(val))
            for s, acc in acc_class.items():
                accuracy[s].append(acc)
            acc = sum(acc_class.values()) / len(acc_class.values())
            accuracy['accuracy_clustering'].append(acc)

        # print("Number of Fragments: ", counts)
        # print("Number of Groups: ", groups)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)
        # pprint(res)

        res['clustering_loss'] = float(self.clustering_weight * res['loss'])
        res['loss'] = self.segmentation_weight * res['loss_seg'] \
                    + self.clustering_weight * res['loss'] \
                    + self.ppn_weight * ppn_loss
        res['loss_seg'] = float(res['loss_seg'])
        res['ppn_loss'] = float(ppn_loss)
        res['ppn_acc'] = float(ppn_res['ppn_acc'])

        print('Segmentation Accuracy: {:.4f}'.format(acc_avg['accuracy']))
        print('Clustering Accuracy: {:.4f}'.format(acc_avg['accuracy_clustering']))
        print('PPN Accuracy: {:.4f}'.format(ppn_res['ppn_acc']))

        # -----------------END OF CNN LOSS COMPUTATION--------------------

        if self.gnn_loss:
            node_batch_labels = out['node_batch_labels'][0]
            node_group_labels = out['node_group_labels'][0]
            node_pred = out['node_predictions'][0]
            centroids = out['centroids'][0]
            node_pred[:, :3] = node_pred[:, :3] + centroids
            edge_pred = out['edge_predictions'][0]

            gnn_loss = self.node_loss(node_pred, node_batch_labels, node_group_labels)
            res['loss'] += self.node_loss_weight * gnn_loss['loss']
            res['grouping_loss'] = float(self.node_loss_weight * gnn_loss['loss'])
            res['accuracy_grouping'] = gnn_loss['accuracy']
            # GNN Grouping Loss
            print('Clustering Loss = ', res['clustering_loss'])
            print('Grouping Loss = ', res['grouping_loss'])
            print('Clustering Accuracy = ', res['accuracy_clustering'])
            print('Grouping Accuracy  = ', res['accuracy_grouping'])

        return res
