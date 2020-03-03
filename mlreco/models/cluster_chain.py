import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

from mlreco.models.layers.uresnet import UResNet
from mlreco.models.cluster_cnn.utils import add_normalized_coordinates, distance_matrix
from mlreco.models.cluster_cnn.spatial_embeddings import UResNetClust
from mlreco.models.cluster_cnn.utils import *
from mlreco.models.gnn.edge_nnconv import *
from mlreco.models.gnn.factories import *

from .cluster_cnn.losses.spatial_embeddings import *
from .cluster_cnn import cluster_model_construct, backbone_construct, clustering_loss_construct


def gaussian_kernel(centroid, sigma):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1, keepdims=False)
        probs = np.exp(-dists / (2.0 * sigma**2))
        return probs
    return f

def multivariate_kernel(centroid, L):
    def f(x):
        N = x.shape[0]
        cov = torch.zeros(3, 3)
        tril_indices = torch.tril_indices(row=3, col=3, offset=0)
        cov[tril_indices[0], tril_indices[1]] = L
        cov = torch.matmul(cov, cov.T)
        dist = torch.matmul((x - centroid), cov)
        dist = torch.bmm(dist.view(N, 1, -1), (x-centroid).view(N, -1, 1)).squeeze()
        probs = torch.exp(-dist)
        return probs
    return f


class EdgeFeatureNet(nn.Module):

    def __init__(self, num_input, num_output):
        super(EdgeFeatureNet, self).__init__()
        self.linear1 = nn.Bilinear(num_input, num_input, 16)
        # self.norm1 = nn.BatchNorm1d(16)
        self.linear2 = nn.Linear(16, 16)
        # self.norm2 = nn.BatchNorm1d(16)
        self.linear3 = nn.Linear(16, num_output)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.linear1(x)
        # x = self.norm1(x)
        x = self.linear2(x)
        # x = self.norm2(x)
        x = self.linear3(x)


def get_edge_features(nodes, batch_idx, bilinear_net):
    '''
    Compile Fully Connected Edge Features from nodes and batch indices.

    INPUTS:
        - nodes (N x d Tensor): list of node features
        - batch_idx (N x 1 Tensor): list of batch indices for nodes
        - bilinear_net: nn.Module that taks two vectors and returns edge feature vector. 

    RETURNS:
        - edge_features: list of edges features
        - edge_indices: list of edge indices (i->j)
        - edge_batch_indices: list of batch indices (0 to B)
    '''
    unique_batch = batch_idx.unique()
    edge_indices = []
    edge_features = []
    edge_batch_indices = []
    for bidx in unique_batch:
        mask = bidx == batch_idx
        nodes_batch = nodes[mask]
        subindex = torch.arange(nodes_batch.shape[0])
        N = nodes_batch.shape[0]
        for i, row in enumerate(nodes_batch):
            submask = subindex != i
            edge_idx = [(i, j) for j in subindex[submask]]
            edge_indices.extend(edge_idx)
            others = nodes_batch[submask]
            ei2j = bilinear_net(row.expand_as(others), others)
            edge_features.extend(ei2j)
            edge_batch_indices.extend([bidx for _ in range(N)])
    
    edge_indices = torch.Tensor(edge_indices)
    edge_features = torch.Tensor(edge_features)
    edge_batch_indices = torch.Tensor(edge_batch_indices)

    return edge_indices, edge_features, edge_batch_indices



class FullChain(nn.Module):

    def __init__(self, cfg, name='cluster_chain'):
        super(FullChain, self).__init__()
        # Includes UResNet Segmentation + Seediness + Clustering
        self.cluster_module = UResNetClust(cfg)
        # PPN Attachment
        # self.ppn = PPN(cfg)
        self.train_gnn = cfg.get('train_gnn', False)
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)
        self.node_predictor = node_model_construct(cfg)
        self.edge_predictor = edge_model_construct(cfg)
        self.edge_net = EdgeFeatureNet(16, 16)


    def fit_predict(self, embeddings, seediness, margins, 
                    fitfunc, st=0.0, pt=0.5):

        pred_labels = torch.zeros(embeddings.shape[0])
        probs = []
        seediness_copy = np.copy(seediness.detach().cpu().numpy())
        count = 0
        while count < seediness.shape[0]:
            i = np.argsort(seediness_copy)[::-1][0]
            seedScore = seediness[i]
            centroid = embeddings[i]
            sigma = margins[i]
            if seedScore < st:
                break
            f = fitfunc(centroid, sigma)
            pValues = f(embeddings)
            probs.append(pValues.reshape(-1, 1))
            cluster_index = pValues > pt
            seediness_copy[cluster_index] = 0
            count += sum(cluster_index)
        if len(probs) == 0:
            return pred_labels
        probs = np.hstack(probs)
        pred_labels = np.argmax(probs, axis=1)
        return pred_labels


    def get_gnn_input(self, featuresAgg, train=True, **kwargs):
        '''
        Get input features to GNN from CNN clustering output

        INPUTS (TRAINING):

            - featuresAgg (N x F): combined feature tensor per voxel, created
            using semantic/seediness/clustering feature maps. 

            - fragment_labels (N, ): ground truth fragment labels.


        INPUTS (INFERENCE):

            - featuresAgg: same as before
            - embeddings: embedding coordinates from clustering network
            - margins: margin values from CNN clustering
            - seediness: seediness values from CNN clustering
            - kernel_func: choice of probability kernel function to be used
            in CNN clustering inference. 


        RETURNS:

            gnn_result: A dictionary containing input node & edge feature to GNN.
            The items are self-explanatory from their variable names. 
        '''
        nodes_full = []
        nodes_batch_id = []
        batch_index = kwargs['batch_index']

        if train:
            fragment_labels = kwargs['fragment_labels']
            for bidx in batch_index:
                batch_mask = coords[:, 3] == bidx
                featuresAgg_batch = featuresAgg[batch_mask]
                fragment_batch = fragment_labels[batch_mask]
                fragment_ids = fragment_batch.unique()
                for fid in fragment_ids:
                    mean_features = featuresAgg_batch[fragment_batch == fid]
                    nodes_full.append(mean_features)
                    nodes_batch_id.append(bidx)
        else:
            with torch.no_grad():
                embeddings = kwargs['embeddings']
                margins = kwargs['margins']
                semantic_labels = kwargs['semantic_labels']
                seediness = kwargs['seediness']
                kernel_func = kwargs.get('kernel_func', multivariate_kernel)
                for bidx in batch_index:
                    batch_mask = coords[:, 3] == bidx
                    slabel = semantic_labels[batch_mask]
                    embedding_batch = embeddings[batch_mask]
                    featuresAgg_batch = featuresAgg[batch_mask]
                    margins_batch = margins[batch_mask]
                    seediness_batch = seediness[batch_mask]
                    for s in slabel.unique():
                        segment_mask = slabel == s
                        featuresAgg_class = featuresAgg_batch[segment_mask]
                        embedding_class = embedding_batch[segment_mask]
                        margins_class = margins_batch[segment_mask]
                        seediness_class = seediness_batch[segment_mask]
                        pred_labels = self.fit_predict(
                            embedding_class, seediness_class, margins_class, kernel_func)
                        # Adding Node Features
                        for c in pred_labels.unique():
                            mask = pred_labels == c
                            mean_features = featuresAgg_class[mask].mean()
                            nodes_full.append(mean_features)
                            nodes_batch_id.append(bidx)

        nodes_full = torch.Tensor(nodes_full)
        node_batch_id = torch.Tensor(nodes_batch_id)

        # print(nodes_full)
        # print(node_batch_id)
        
        # Compile Edges Features
        edge_indices, edge_features, edge_batch_indices = get_edge_features(
            nodes_full, node_batch_id, self.edge_net)

        # print(edge_indices)
        # print(edge_features)
        # print(edge_batch_indices)
        gnn_result = {
            'edge_indices': edge_indices,
            'edge_features': edge_features,
            'edge_batch_id': edge_batch_indices,
            'node_features': nodes_full,
            'node_batch_id': node_batch_id
        }
        return gnn_result


    def forward(self, input):
        '''
        Forward for full reconstruction chain.

        INPUTS:
            - input (N x 8 Tensor): input_data

        RETURNS:
            - result (tuple of dicts): (cnn_result, gnn_result)
        '''
        # Run all CNN modules. 
        cnn_result = self.cluster_module(input)
        
        # UResNet Results
        embeddings = cnn_result['embeddings'][0]
        margins = cnn_result['margins'][0]
        seediness = cnn_result['seediness'][0]
        segmentation = cnn_result['segmentation'][0]
        featuresAgg = cnn_result['features_aggregate'][0]

        # Ground Truth Labels
        coords = input[0][:, :4]
        batch_index = input[0][:, 3].unique()
        semantic_labels = input[0][:, -1]
        fragment_labels = input[0][:, -3]

        result = {
            'coords': coords,
            'batch_index': batch_index,
            'semantic_labels': semantic_labels,
            'segmentation': segmentation,
            'embeddings': embeddings,
            'margins': margins,
            'seediness': seediness,
            'kernel_func': multivariate_kernel
        }

        if self.train_gnn:
            gnn_input = self.get_gnn_input(featuresAgg, train=True, 
                fragment_labels=fragment_labels, batch_index=batch_index)
            print(gnn_input)

        else:
            del result['coords']
            del result['batch_index']
            del result['semantic_labels']
            result = (cnn_result, None)

        return result


class ChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.segmentation_loss = torch.nn.CrossEntropyLoss()
        self.ppn_loss = PPNLoss(cfg)
        self.loss_config = cfg['clustering_loss']

        self.clustering_loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.clustering_loss = clustering_loss_construct(self.clustering_loss_name)
        self.clustering_loss = self.clustering_loss(cfg)

        self.node_loss = NodeChannelLoss(cfg)
        self.edge_loss = EdgeChannelLoss(cfg)
        self.spatial_size = self.loss_config.get('spatial_size', 768)

        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.ppn_weight = self.loss_config.get('ppn_weight', 1.0)

    def forward(self, result, input_data, ppn_label, ppn_segment_label, graph):
        '''
        Forward propagation for AlphaNeut

        INPUTS:
            - result (dict): result from forwarding three-tailed UResNet, with 
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

        embedding = result['embeddings'][0]
        seediness = result['seediness'][0]
        margins = result['margins'][0]
        # PPN Loss. 
        # FIXME: This implementation will loop over the batch twice.
        print(ppn_segment_label[0].shape)
        print(ppn_label[0].shape)
        ppn_res = self.ppn_loss(result, ppn_segment_label, ppn_label)
        ppn_loss = ppn_res['ppn_loss']

        for bidx in batch_idx:

            batch_mask = coords[:, -1] == bidx
            seg_logits = result['segmentation'][0][batch_mask]
            # print(seg_logits)
            embedding_batch = embedding[batch_mask]
            slabels_batch = segment_label[batch_mask]
            clabels_batch = fragment_label[batch_mask]
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
            loss_class, acc_class = self.clustering_loss.combine_multiclass(
                embedding_batch, margins_batch, 
                seed_batch, slabels_batch, clabels_batch, coords_batch)
            for key, val in loss_class.items():
                loss[key].append(sum(val) / len(val))
            for s, acc in acc_class.items():
                accuracy[s].append(acc)
            acc = sum(acc_class.values()) / len(acc_class.values())
            accuracy['clustering_accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        res['loss'] = self.segmentation_weight * res['loss_seg'] \
                    + self.clustering_weight * res['loss'] \
                    + self.ppn_weight * ppn_loss 
        res['loss_seg'] = float(res['loss_seg'])

        # -----------------END OF CNN LOSS COMPUTATION--------------------

        return res


