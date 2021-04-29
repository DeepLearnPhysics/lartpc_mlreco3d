import torch
import numpy as np
from collections import defaultdict

from mlreco.models.chain.full_cnn import *
from .gnn import node_encoder_construct, edge_encoder_construct

from mlreco.models.uresnet_lonely import SegmentationLoss
from mlreco.models.ppn import PPNLoss
from .cluster_cnn import spice_loss_construct

class FullChain(torch.nn.Module):
    """
    Driver class for the CNN component of the end-to-end reconstruction chain
    1) UResNet
        1) Semantic - for point classification
        2) PPN - for particle point locations
        3) Fragment - to form particle fragments

    For use in config:
    model:
      name: full_chain
      modules:
        full_cnn:
          <full CNN global parameters, see mlreco/models/chain/full_cnn.py>
        network_base:
          <uresnet archicteture core parameters, see mlreco/models/layers/base.py>
        uresnet_encoder:
          <uresnet encoder parameters, see mlreco/models/layers/uresnet.py>
        segmentation_decoder:
          <uresnet segmention decoder paraters, see mlreco/models/chain/full_cnn.py>
        seediness_decoder:
          <uresnet seediness decoder paraters, see mlreco/models/chain/full_cnn.py>
        embedding_decoder:
          <uresnet embedding decoder paraters, see mlreco/models/chain/full_cnn.py>
        full_chain_loss:
          name: <name of the loss function for the CNN fragment clustering model>
          spatial_size: <spatial size of input images>
          segmentation_weight: <relative weight of the segmentation loss>
          clustering_weight: <relative weight of the clustering loss>
          seediness_weight: <relative weight of the seediness loss>
          embedding_weight: <relative weight of the embedding loss>
          smoothing_weight: <relative weight of the smoothing loss>
          ppn_weight: <relative weight of the ppn loss>
    """

    MODULES = ['full_cnn', 'network_base', 'uresnet_encoder', 'segmentation_decoder',
            'seediness_decoder', 'embedding_decoder', 'full_chain_loss', 'ppn']

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()

        # Initialize the full CNN model (includes UResNet+PPN+Fragmentation)
        self.full_cnn = FullCNN(cfg)

    def forward(self, input):
        '''
        Forward for full reconstruction chain.

        INPUTS:
            - input (N x 5 Tensor): Input data [x, y, z, batch_id, val]

        RETURNS:
            - result (tuple of dicts): (cnn_result)
        '''
        # Run all CNN modules
        return self.full_cnn(input)


class FullChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(FullChainLoss, self).__init__()

        # Initialize loss components
        self.loss_config = cfg['full_chain_loss']
        self.segmentation_loss = SegmentationLoss({'uresnet_lonely':cfg['full_cnn']})
        self.ppn_loss = PPNLoss(cfg)
        self.spice_loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.spice_loss = spice_loss_construct(self.spice_loss_name)
        self.spice_loss = self.spice_loss(cfg, name='full_chain_loss')

        # Initialize the loss weights
        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.ppn_weight = self.loss_config.get('ppn_weight', 0.0)

    def forward(self, out, cluster_label, ppn_label):
        '''
        Forward propagation for FullChain

        INPUTS:
            - out (dict): result from forwarding three-tailed UResNet, with
            1) segmenation decoder 2) clustering decoder 3) seediness decoder,
            and PPN attachment to the segmentation branch.

            - cluster_label (list of Tensors): input data tensor of shape N x 10
              In row-index order:
              1. x coordinates
              2. y coordinates
              3. z coordinates
              4. batch indices
              5. energy depositions
              6. fragment labels
              7. group labels
              8. interaction labels
              9. neutrino labels
              10. segmentation labels (0-5, includes ghosts)

            - ppn_label (list of Tensors): particle labels for ppn ground truth
        '''

        # Apply the segmenation loss
        coords = cluster_label[0][:, :4]
        segment_label = cluster_label[0][:, -1]
        segment_label_tensor = torch.cat((coords, segment_label.reshape(-1,1)), dim=1)
        res_seg = self.segmentation_loss(out, [segment_label_tensor])
        seg_acc, seg_loss = res_seg['accuracy'], res_seg['loss']

        # Apply the PPN loss
        res_ppn = self.ppn_loss(out, [segment_label_tensor], ppn_label)

        # Apply the CNN dense clustering loss
        fragment_label = cluster_label[0][:, 5]
        batch_idx = coords[:, -1].unique()
        res_cnn_clust = defaultdict(int)
        for bidx in batch_idx:
            # Get the loss input for this batch
            batch_mask = coords[:, -1] == bidx
            highE_mask = segment_label[batch_mask] != 4
            embedding_batch_highE = out['embeddings'][0][batch_mask][highE_mask]
            margins_batch_highE = out['margins'][0][batch_mask][highE_mask]
            seed_batch_highE = out['seediness'][0][batch_mask][highE_mask]
            slabels_highE = segment_label[batch_mask][highE_mask]
            clabels_batch_highE = fragment_label[batch_mask][highE_mask]

            # Get the clustering loss, append results
            loss_class, acc_class = self.spice_loss.combine_multiclass(
                embedding_batch_highE, margins_batch_highE,
                seed_batch_highE, slabels_highE, clabels_batch_highE)

            loss, accuracy = 0, 0
            for key, val in loss_class.items():
                res_cnn_clust[key+'_loss'] += (sum(val) / len(val))
                loss += (sum(val) / len(val))
            for key, val in acc_class.items():
                res_cnn_clust[key+'_accuracy'] += val
                accuracy += val

            res_cnn_clust['loss'] += loss/len(loss_class.values())/len(batch_idx)
            res_cnn_clust['accuracy'] += accuracy/len(acc_class.values())/len(batch_idx)

        cnn_clust_acc, cnn_clust_loss = res_cnn_clust['accuracy'], res_cnn_clust['loss']

        # Combine the results
        accuracy = (res_seg['accuracy'] + res_ppn['ppn_acc'] + res_cnn_clust['accuracy'])/3.
        loss = self.segmentation_weight*res_seg['loss'] \
             + self.ppn_weight*res_ppn['ppn_loss'] \
             + self.clustering_weight*res_cnn_clust['loss']

        res = {}
        res.update(res_seg)
        res.update(res_ppn)
        res.update(res_cnn_clust)
        res['seg_accuracy'] = seg_acc
        res['seg_loss'] = seg_loss
        res['ppn_accuracy'] = res_ppn['ppn_acc']
        res['ppn_loss'] = res_ppn['ppn_loss']
        res['cnn_clust_accuracy'] = cnn_clust_acc
        res['cnn_clust_loss'] = cnn_clust_loss
        res['loss'] = loss
        res['accuracy'] = accuracy

        print('Segmentation Accuracy: {:.4f}'.format(res_seg['accuracy']))
        print('PPN Accuracy: {:.4f}'.format(res_ppn['ppn_acc']))
        print('Clustering Accuracy: {:.4f}'.format(res_cnn_clust['accuracy']))

        return res
