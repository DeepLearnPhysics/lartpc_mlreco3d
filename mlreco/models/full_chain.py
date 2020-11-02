import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict
import copy

from mlreco.models.chain.full_cnn import *
from mlreco.models.ppn import PPNLoss
from .cluster_cnn import spice_loss_construct
from mlreco.models.grappa import GNN, GNNLoss
from mlreco.models.gnn.losses.node_grouping import *

from pprint import pprint

class FullChain(nn.Module):

    MODULES = ['shower_gnn', 'shower_edge_model', 'interaction_gnn', 'interaction_edge_model']

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()
        #self.model_config = cfg[name]

        self.full_cnn = FullCNN(cfg)
        self.shower_gnn = GNN(cfg, 'grappa_shower')
        self.inter_gnn = GNN(cfg, 'grappa_inter')
        self.edge_predictor = self.shower_gnn.edge_predictor
        self.inte_edge_predictor = self.inte_gnn.edge_predictor

        # self.edge_net = EdgeFeatureNet(16, 16)

        # self.cnn_freeze = self.model_config.get('cnn_freeze', False)
        # self.gnn_freeze = self.model_config.get('gnn_freeze', False)

    def forward(self, input):
        '''
        Forward for full reconstruction chain.

        INPUTS:
            - input (N x 8 Tensor): input_data

        RETURNS:
            - result (tuple of dicts): (cnn_result, gnn_result)
        '''
        # Run all CNN modules.
        result = self.full_cnn([input[0]])

        # UResNet Results
        # embeddings = result['embeddings'][0]
        # margins = result['margins'][0]
        # seediness = result['seediness'][0]
        # segmentation = result['segmentation'][0]
        # features_gnn = result['features_gnn'][0]

        # Ground Truth Labels
        # coords = input[0][:, :4]
        # batch_index = input[0][:, 3].unique()
        # semantic_labels = input[0][:, -1]
        # fragment_labels = input[0][:, 5]
        # group_labels = input[0][:, 6]
        #
        # gnn_input = get_gnn_input(
        #      coords,
        #      features_gnn,
        #      self.edge_net,
        #      fit_predict,
        #      train=True,
        #      embeddings=embeddings,
        #      fragment_labels=fragment_labels,
        #      group_labels=group_labels,
        #      batch_index=batch_index)
        #
        # gnn_output = self.full_gnn(
        #      gnn_input.x, gnn_input.edge_index, gnn_input.edge_attr, gnn_input.batch)
        #
        # result.update(gnn_output)
        # result['node_batch_labels'] = [gnn_input.batch]
        # result['node_group_labels'] = [gnn_input.node_group_labels]
        # result['centroids'] = [gnn_input.centroids]
        # for key, val in result.items():
        #     print(key, val[0].shape)

        pred_labels = fit_predict(
            result['embeddings'][0], result['seediness'][0], result['margins'][0], gaussian_kernel)
        pred_tensor = input[0].clone()
        pred_tensor[:,5] = pred_labels
        pred_tensor[:,9] = torch.argmax(result['segmentation'][0],1).flatten()
        gnn_output = self.shower_gnn([pred_tensor])
        result.update(gnn_output)

        pred_tensor = relabel_groups([pred_tensor], gnn_output['clusts'], gnn_output['group_pred'], new_array=False)[0]
        gnn_output = self.inter_gnn([pred_tensor, input[1]])
        result['particle_clusts'] = gnn_output['clusts']
        result['particle_edge_index'] = gnn_output['edge_index']
        result['particle_edge_pred']  = gnn_output['edge_pred']

        return result


class FullChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(FullChainLoss, self).__init__()
        self.segmentation_loss = torch.nn.CrossEntropyLoss()
        self.ppn_loss = PPNLoss(cfg)
        self.loss_config = cfg['full_chain_loss']
        #print(self.loss_config)
        self.gnn_loss = self.loss_config.get('gnn_loss', False)

        self.spice_loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.spice_loss = spice_loss_construct(self.spice_loss_name)
        self.spice_loss = self.spice_loss(cfg, name='full_chain_loss')
        #print(self.spice_loss)

        #self.node_loss = GNNGroupingLoss(cfg)
        #self.node_loss_weight = self.loss_config.get('node_loss_weight', 1.0)

        self.shower_gnn_loss = GNNLoss(cfg, 'grappa_shower_loss')
        self.inter_gnn_loss = GNNLoss(cfg, 'grappa_inter_loss')
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
              9. interaction labels

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
        segment_label = input_data[0][:, 9]
        #print(segment_label.unique(return_counts=True))
        fragment_label = input_data[0][:, 5]
        group_label = input_data[0][:, 6]
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
            #print(segment_pred.unique(return_counts=True))
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

            loss_class, acc_class = self.spice_loss.combine_multiclass(
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

        res['spice_loss'] = float(self.clustering_weight * res['loss'])
        res['acc_seg'] = float(res['accuracy'])
        res['ppn_loss'] = float(ppn_loss)
        res['ppn_acc'] = float(ppn_res['ppn_acc'])
        res['loss'] = self.segmentation_weight * res['loss_seg'] \
                    + self.clustering_weight * res['loss'] \
                    + self.ppn_weight * ppn_loss

        print('Segmentation Accuracy: {:.4f}'.format(acc_avg['accuracy']))
        print('Clustering Accuracy: {:.4f}'.format(acc_avg['accuracy_clustering']))
        print('PPN Accuracy: {:.4f}'.format(ppn_res['ppn_acc']))

        # -----------------END OF CNN LOSS COMPUTATION--------------------

        grouping_loss = self.shower_gnn_loss(out, input_data)
        loss = res['loss'] + grouping_loss['loss']
        res.update(grouping_loss)
        res['loss'] = loss

        print('Shower grouping prediction accuracy: {:.4f}'.format(res['edge_accuracy']))
        print('Shower primary prediction accuracy: {:.4f}'.format(res['node_accuracy']))

        inte_tensor = input_data[0].clone()
        inte_tensor[:,5] = inte_tensor[:,6]
        inte_tensor[:,6] = inte_tensor[:,9]
        inte_out = {}
        inte_out['clusts'] = out['particle_clusts']
        inte_out['edge_index'] = out['particle_edge_index']
        inte_out['edge_pred'] = out['particle_edge_pred']
        interaction_loss = self.inter_gnn_loss(inte_out, [inte_tensor], None)
        res['particle_edge_loss'] = interaction_loss['loss']
        res['particle_edge_accuracy'] = interaction_loss['accuracy']
        res['loss'] += interaction_loss['loss']

        print('Interaction grouping prediction accuracy: {:.4f}'.format(res['particle_edge_accuracy']))

        #if self.gnn_loss:
        #    node_batch_labels = out['node_batch_labels'][0]
        #    node_group_labels = out['node_group_labels'][0]
        #    node_pred = out['node_predictions'][0]
        #    centroids = out['centroids'][0]
        #    node_pred[:, :3] = node_pred[:, :3] + centroids
        #    edge_pred = out['edge_predictions'][0]
        #
        #    gnn_loss = self.node_loss(node_pred, node_batch_labels, node_group_labels)
        #    res['loss'] += self.node_loss_weight * gnn_loss['loss']
        #    res['grouping_loss'] = float(self.node_loss_weight * gnn_loss['loss'])
        #    res['accuracy_grouping'] = gnn_loss['accuracy']
        #    # GNN Grouping Loss
        #    print('Clustering Loss = ', res['spice_loss'])
        #    print('Grouping Loss = ', res['grouping_loss'])
        #    print('Clustering Accuracy = ', res['accuracy_clustering'])
        #    print('Grouping Accuracy  = ', res['accuracy_grouping'])

        return res
