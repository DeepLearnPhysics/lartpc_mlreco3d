import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME

from pprint import pprint
from mlreco.models.experimental.cluster.transformer_spice import TransformerSPICE
from mlreco.models.experimental.cluster.criterion import *
from mlreco.utils.globals import *
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class Mask3DModel(nn.Module):
    '''
    Transformer-Instance Query based particle clustering

    Configuration
    -------------
    skip_classes: list, default [2, 3, 4]
        semantic labels for which to skip voxel clustering
        (ex. Michel, Delta, and Low Es rarely require neural network clustering)
    dimension: int, default 3
        Spatial dimension (2 or 3).
    min_points: int, default 0
        If a value > 0 is specified, this will enable the orphans assignment for
        any predicted cluster with voxel count < min_points.
    '''

    MODULES = ['mask3d', 'query_module', 'fourier_embeddings', 'transformer_decoder']

    def __init__(self, cfg, name='mask3d'):
        super(Mask3DModel, self).__init__()
        self.net = TransformerSPICE(cfg)
        self.skip_classes = cfg[name].get('skip_classes')

    def filter_class(self, x):
        '''
        Filter classes according to segmentation label.
        '''
        mask = ~np.isin(x[:, -1].detach().cpu().numpy(), self.skip_classes)
        point_cloud = x[mask]
        return point_cloud


    def forward(self, input):
        '''

        '''
        x = input[0]
        point_cloud = self.filter_class(x)
        res = self.net(point_cloud)
        return res


class Mask3dLoss(nn.Module):
    """
    Loss function for GraphSpice.

    Configuration
    -------------
    name: str, default 'se_lovasz_inter'
        Loss function to use.
    invert: bool, default True
        You want to leave this to True for statistical weighting purpose.
    kernel_lossfn: str
    edge_loss_cfg: dict
        For example

        .. code-block:: yaml

          edge_loss_cfg:
            loss_type: 'LogDice'

    eval: bool, default False
        Whether we are in inference mode or not.

        .. warning::

            Currently you need to manually switch ``eval`` to ``True``
            when you want to run the inference, as there is no way (?)
            to know from within the loss function whether we are training
            or not.

    Output
    ------
    To be completed.

    See Also
    --------
    MinkGraphSPICE
    """
    def __init__(self, cfg, name='mask3d'):
        super(Mask3dLoss, self).__init__()
        self.model_config = cfg[name]
        self.skip_classes = self.model_config.get('skip_classes', [2, 3, 4])
        self.num_queries = self.model_config.get('num_queries', 200)
        
        
        self.weight_class = torch.Tensor([0.1, 5.0])
        self.xentropy = nn.CrossEntropyLoss(weight=self.weight_class, reduction='mean')
        self.dice_loss_mode = self.model_config.get('dice_loss_mode', 'log_dice')

        self.loss_fn = LinearSumAssignmentLoss(mode=self.dice_loss_mode)
        # self.loss_fn = CEDiceLoss(mode=self.dice_loss_mode)

    def filter_class(self, cluster_label):
        '''
        Filter classes according to segmentation label.
        '''
        mask = ~np.isin(cluster_label[0][:, -1].cpu().numpy(), self.skip_classes)
        clabel = [cluster_label[0][mask]]
        return clabel
    
    def compute_layerwise_loss(self, aux_masks, aux_classes, clabel, query_index):
        
        batch_col = clabel[0][:, BATCH_COL].int()
        num_batches = batch_col.unique().shape[0]
        
        loss = defaultdict(list)
        loss_class = defaultdict(list)
        
        for bidx in range(num_batches):
            for layer, mask_layer in enumerate(aux_masks):
                batch_mask = batch_col == bidx
                labels = clabel[0][batch_mask][:, GROUP_COL].long()
                query_idx_batch = query_index[bidx]
                # Compute instance mask loss
                targets = get_instance_masks(labels).float()
                # targets = get_instance_masks_from_queries(labels, query_idx_batch).float()
                loss_batch, acc_batch = self.loss_fn(mask_layer[batch_mask], targets)
                loss[bidx].append(loss_batch)
                
                # Compute instance class loss 
                # logits_batch = aux_classes[layer][bidx]
                # targets_class = torch.zeros(logits_batch.shape[0]).to(
                #     dtype=torch.long, device=logits_batch.device)
                # targets_class[indices[0]] = 1
                # loss_class_batch = self.xentropy(logits_batch, targets_class)
                # loss_class[bidx].append(loss_class_batch)
        
        return loss, loss_class


    def forward(self, result, cluster_label):
        '''

        '''
        clabel = self.filter_class(cluster_label)

        aux_masks = result['aux_masks'][0]
        aux_classes = result['aux_classes'][0]
        query_index = result['query_index'][0]
        
        batch_col = clabel[0][:, BATCH_COL].int()
        num_batches = batch_col.unique().shape[0]
        
        loss, acc = defaultdict(list), defaultdict(list)
        loss_class = defaultdict(list)
        
        loss_layer, loss_class_layer = self.compute_layerwise_loss(aux_masks, 
                                                            aux_classes, 
                                                            clabel,
                                                            query_index)
        
        loss.update(loss_layer)
        # loss_class.update(loss_class_layer)
        
        acc_class = 0
        
        for bidx in range(num_batches):
            batch_mask = batch_col == bidx
            
            output_mask = result['pred_masks'][0][batch_mask]
            output_class = result['pred_logits'][0][bidx]
            
            labels = clabel[0][batch_mask][:, GROUP_COL].long()
            
            targets = get_instance_masks(labels).float()
            query_idx_batch = query_index[bidx]
            # targets = get_instance_masks_from_queries(labels, query_idx_batch).float()
        
            loss_batch, acc_batch = self.loss_fn(output_mask, targets)
            loss[bidx].append(loss_batch)
            acc[bidx].append(acc_batch)
            
            # Compute instance class loss 
            # targets_class = torch.zeros(output_class.shape[0]).to(
            #     dtype=torch.long, device=output_class.device)
            # targets_class[indices[0]] = 1
            # loss_class_batch = self.xentropy(output_class, targets_class)
            # loss_class[bidx].append(loss_class_batch)
            
            # with torch.no_grad():
            #     pred = torch.argmax(output_class, dim=1)
            #     obj_acc = (pred == targets_class).sum() / pred.shape[0]
            #     acc_class += obj_acc / num_batches
            
        loss = [sum(val) / len(val) for val in loss.values()]
        acc =  [sum(val) / len(val) for val in acc.values()]
        # loss_class = [sum(val) / len(val) for val in loss_class.values()]
        
        loss = sum(loss) / len(loss)
        # loss_class = sum(loss_class) / len(loss_class)
        acc = sum(acc) / len(acc)
            
        res = {
            'loss': loss,
            'accuracy': acc,
            # 'loss_class': float(loss_class),
            'loss_mask': float(loss),
            # 'acc_class': float(acc_class)
        }
        
        return res