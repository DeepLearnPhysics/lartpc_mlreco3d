# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for lartpc_mlreco3d

import torch
import torch.nn.functional as F
from torch import nn
from mlreco.utils.globals import *
from scipy.optimize import linear_sum_assignment
from mlreco.models.layers.cluster_cnn.losses.misc import iou_batch, LovaszHingeLoss

class LinearSumAssignmentLoss(nn.Module):
    
    def __init__(self, weight_dice=2.0, weight_ce=5.0, mode='dice'):
        super(LinearSumAssignmentLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        
        self.lovasz = LovaszHingeLoss()
        self.mode = mode
        print(f"Setting LinearSumAssignment loss to '{self.mode}'")
        
    def compute_accuracy(self, masks, targets, indices):
        with torch.no_grad():
            valid_masks = masks[:, indices[0]] > 0
            valid_targets = targets[:, indices[1]] > 0.5
            iou = iou_batch(valid_masks, valid_targets, eps=1e-6)
            return float(iou)
        
    def forward(self, masks, targets):
        
        with torch.no_grad():
            dice_loss = batch_dice_loss(masks.T, targets.T)
            ce_loss = batch_sigmoid_ce_loss(masks.T, targets.T)
            cost_matrix = self.weight_dice * dice_loss + self.weight_ce * ce_loss
            indices = linear_sum_assignment(cost_matrix.detach().cpu())
        
        if self.mode == 'log_dice':
            dice_loss = log_dice_loss_flat(masks[:, indices[0]], targets[:, indices[1]])
        elif self.mode == 'dice':
            dice_loss = dice_loss_flat(masks[:, indices[0]], targets[:, indices[1]])
        # elif self.mode == 'lovasz':
        #     dice_loss = self.lovasz(masks[:, indices[0]], targets[:, indices[1]])
        else:
            raise ValueError(f"LSA loss mode {self.mode} is not supported!")
        ce_loss = sigmoid_ce_loss(masks.T[indices[0]], targets.T[indices[1]])
        loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        acc = self.compute_accuracy(masks, targets, indices)
        
        return loss, acc
    
    
class CEDiceLoss(nn.Module):
    
    def __init__(self, weight_dice=1.0, weight_ce=1.0, mode='dice'):
        super(CEDiceLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.mode = mode
        print(f"Setting LinearSumAssignment loss to '{self.mode}'")
        
    def compute_accuracy(self, masks, targets):
        with torch.no_grad():
            valid_masks = masks > 0
            valid_targets = targets > 0.5
            
            print(masks.sum(dim=0))
            print(targets.sum(dim=0))
        
            iou = iou_batch(valid_masks, valid_targets, eps=1e-6)
            return float(iou)
        
    def forward(self, masks, targets):
        
        dice_loss = dice_loss_flat(masks, targets)
        # if self.mode == 'log_dice':
        #     dice_loss = batch_log_dice_loss(masks.T[indices[0]], targets.T[indices[1]])
        # elif self.mode == 'dice':
        #     dice_loss = batch_dice_loss(masks.T[indices[0]], targets.T[indices[1]])
        # elif self.mode == 'lovasz':
        #     dice_loss = self.lovasz(masks[:, indices[0]], targets[:, indices[1]])
        # else:
        #     raise ValueError(f"LSA loss mode {self.mode} is not supported!")
        ce_loss = sigmoid_ce_loss(masks.T, targets.T)
        loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        acc = self.compute_accuracy(masks, targets)
        
        return loss, acc


@torch.jit.script
def get_instance_masks(cluster_label : torch.LongTensor, 
                       max_num_instances: int = -1):
    """Given integer coded cluster instance labels, construct a
    (N x max_num_instances) bool tensor in which each colume is a 
    binary instance mask.
    
    """
    groups, counts = torch.unique(cluster_label, return_counts=True)
    if max_num_instances < 0:
        max_num_instances = groups.shape[0]
    instance_masks = torch.zeros((cluster_label.shape[0], 
                                  max_num_instances)).to(device=cluster_label.device, 
                                                         dtype=torch.bool)                              
    perm = torch.argsort(counts, descending=True)[:max_num_instances]
                                  
    for i, group_id in enumerate(groups[perm]):
        instance_masks[:, i] = (cluster_label == group_id).to(torch.bool)
        
    return instance_masks


@torch.jit.script
def get_instance_masks_from_queries(cluster_label: torch.LongTensor,
                                    query_index: torch.Tensor):
    max_num_instances = query_index.shape[0]
    instance_masks = torch.zeros((cluster_label.shape[0], 
                                  max_num_instances)).to(device=cluster_label.device, 
                                                         dtype=torch.bool)  
    for i, qidx in enumerate(query_index):
        instance_masks[:, i] = (cluster_label == cluster_label[qidx]).to(torch.bool)
    
    return instance_masks
    


def dice_loss(logits, targets):
    """
    
    Parameters
    ----------
    logits: (N x num_queries)
    targets: (N x num_queries)
    """
    num_masks = logits.shape[1]
    scores = torch.sigmoid(logits)
    numerator = (2 * scores * targets).sum(dim=0)
    denominator = scores.sum(dim=0) + targets.sum(dim=0)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: (num_masks, num_points) Tensor
        targets: (num_masks, num_points) Tensor
    """
    scores = inputs.sigmoid()
    scores = scores.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", scores, targets)
    denominator = scores.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

@torch.jit.script
def batch_log_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: (num_masks, num_points) Tensor
        targets: (num_masks, num_points) Tensor
    """
    scores = inputs.sigmoid()
    scores = scores.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", scores, targets)
    denominator = scores.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = -torch.log((numerator + 1) / (denominator + 1))
    return loss
    
@torch.jit.script
def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

@torch.jit.script
def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    num_masks = inputs.shape[0]
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks

@torch.jit.script
def dice_loss_flat(logits, targets):
    """
    
    Parameters
    ----------
    logits: (N x num_queries)
    targets: (N x num_queries)
    """
    num_masks = logits.shape[1]
    scores = torch.sigmoid(logits)
    numerator = (2 * scores * targets).sum(dim=0)
    denominator = scores.sum(dim=0) + targets.sum(dim=0)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks

@torch.jit.script
def log_dice_loss_flat(logits, targets):
    """
    
    Parameters
    ----------
    logits: (N x num_queries)
    targets: (N x num_queries)
    """
    num_masks = logits.shape[1]
    scores = torch.sigmoid(logits)
    numerator = (2 * scores * targets).sum(dim=0)
    denominator = scores.sum(dim=0) + targets.sum(dim=0)
    return (-torch.log(1 - (numerator + 1) / (denominator + 1))).sum() / num_masks