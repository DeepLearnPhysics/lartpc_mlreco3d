import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

from mlreco.models.layers.uresnet_layers import UResNet, ACASUNet, ASPPUNet
from collections import defaultdict
from mlreco.models.layers.activation_normalization_factories import activations_construct

class UResNet_Chain(nn.Module):


    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,), (3, 1)]
    ]

    MODULES = ['uresnet_lonely']

    def __init__(self, cfg, name='uresnet_lonely'):
        super(UResNet_Chain, self).__init__()
        self.model_config = cfg[name]
        #print("MODEL CONFIG = ", self.model_config)
        mode = self.model_config.get('aspp_mode', None)
        self.D = self.model_config.get('data_dim', 3)
        self.F = self.model_config.get('num_filters', 16)
        self.num_classes = self.model_config.get('num_classes', 5)\

        # Parameters for Deghosting
        self.ghost = self.model_config.get('ghost', False)
        self.ghost_label = self.model_config.get('ghost_label', -1)


        if mode == 'acas':
            self.net = ACASUNet(cfg)
        elif mode == 'aspp':
            self.net = ASPPUNet(cfg)
        else:
            self.net = UResNet(cfg)

        self.output = [
            ME.MinkowskiBatchNorm(self.F,
                eps=self.net.norm_args.get('eps', 0.00001),
                momentum=self.net.norm_args.get('momentum', 0.1)),
            activations_construct('lrelu', negative_slope=0.33),
            ME.MinkowskiLinear(self.F, self.num_classes)]
        self.output = nn.Sequential(*self.output)

        if self.ghost:
            print("Ghost Masking is enabled for UResNet Segmentation")
            self.linear_ghost = ME.MinkowskiLinear(self.F, 2)

        print('Total Number of Trainable Parameters (mink_uresnet)= {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))
        #print(self)
    def forward(self, input):
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            res = self.net(x)
            feats = res['decoderTensors'][-1]
            seg = self.output(feats)
            out['segmentation'].append(seg.F)
            out['finalTensor'].append(res['finalTensor'])
            out['encoderTensors'].append(res['encoderTensors'])
            out['decoderTensors'].append(res['decoderTensors'])
            if self.ghost:
                ghost = self.linear_ghost(feats)
                out['ghost'].append(ghost.F)
                out['ghost_sptensor'].append(ghost)
        return out


# class SegmentationLoss(nn.Module):
#
#     def __init__(self, cfg, name='segmentation_loss'):
#         super(SegmentationLoss, self).__init__()
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, outputs, label, weight=None):
#         '''
#         segmentation[0], label and weight are lists of size #gpus = batch_size.
#         segmentation has as many elements as UResNet returns.
#         label[0] has shape (N, dim + batch_id + 1)
#         where N is #pts across minibatch_size events.
#         '''
#         # TODO Add weighting
#         segmentation = outputs['segmentation']
#
#         assert len(segmentation) == len(label)
#         # if weight is not None:
#         #     assert len(data) == len(weight)
#         batch_ids = [d[:, 0] for d in label]
#         total_loss = 0
#         total_acc = 0
#         count = 0
#         # Loop over GPUS
#         for i in range(len(segmentation)):
#             for b in batch_ids[i].unique():
#                 batch_index = batch_ids[i] == b
#                 event_segmentation = segmentation[i][batch_index]
#                 event_label = label[i][:, -1][batch_index]
#                 event_label = torch.squeeze(event_label, dim=-1).long()
#                 loss_seg = self.cross_entropy(event_segmentation, event_label)
#                 if weight is not None:
#                     event_weight = weight[i][batch_index]
#                     event_weight = torch.squeeze(event_weight, dim=-1).float()
#                     total_loss += torch.mean(loss_seg * event_weight)
#                 else:
#                     total_loss += torch.mean(loss_seg)
#                 # Accuracy
#                 predicted_labels = torch.argmax(event_segmentation, dim=-1)
#                 acc = (predicted_labels == event_label).sum().item() / float(predicted_labels.nelement())
#                 total_acc += acc
#                 count += 1
#
#         return {
#             'accuracy': total_acc/count,
#             'loss': total_loss/count
#         }

class SegmentationLoss(torch.nn.modules.loss._Loss):
    """
    Loss definition for UResNet.

    For a regular flavor UResNet, it is a cross-entropy loss.
    For deghosting, it depends on a configuration parameter `ghost`:

    - If `ghost=True`, we first compute the cross-entropy loss on the ghost
    point classification (weighted on the fly with sample statistics). Then we
    compute a mask = all non-ghost points (based on true information in label)
    and within this mask, compute a cross-entropy loss for the rest of classes.

    - If `ghost=False`, we compute a N+1-classes cross-entropy loss, where N is
    the number of classes, not counting the ghost point class.

    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (int,), (3, 1)]
    ]

    def __init__(self, cfg, reduction='sum', batch_col=3):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['uresnet_lonely']
        self._ghost = self._cfg.get('ghost', False)
        self._ghost_label = self._cfg.get('ghost_label', -1)
        self._num_classes = self._cfg.get('num_classes', 5)
        self._alpha = self._cfg.get('alpha', 1.0)
        self._beta = self._cfg.get('beta', 1.0)
        self._weight_loss = self._cfg.get('weight_loss', False)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self._batch_col = batch_col

    def forward(self, result, label, weights=None):
        """
        result[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.

        Assumptions
        ===========
        The ghost label is the last one among the classes numbering.
        If ghost = True, then num_classes should not count the ghost class.
        If ghost_label > -1, then we perform only ghost segmentation.
        """
        assert len(result['segmentation']) == len(label)
        batch_ids = [d[:, self._batch_col] for d in label]
        # print("batch ids", batch_ids)
        uresnet_loss, uresnet_acc = 0., 0.
        uresnet_acc_class = [0.] * self._num_classes
        count_class = [0.] * self._num_classes
        mask_loss, mask_acc = 0., 0.
        ghost2ghost, nonghost2nonghost = 0., 0.
        count = 0
        for i in range(len(label)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b

                event_segmentation = result['segmentation'][i][batch_index]  # (N, num_classes)
                event_label = label[i][batch_index][:, -1][:, None]  # (N, 1)
                event_label = torch.squeeze(event_label, dim=-1).long()
                if self._ghost_label > -1:
                    event_label = (event_label == self._ghost_label).long()

                elif self._ghost:
                    # check and warn about invalid labels
                    unique_label,unique_count = torch.unique(event_label,return_counts=True)
                    if (unique_label > self._num_classes).long().sum():
                        print('Invalid semantic label found (will be ignored)')
                        print('Semantic label values:',unique_label)
                        print('Label counts:',unique_count)
                    event_ghost = result['ghost'][i][batch_index]  # (N, 2)
                    # 0 = not a ghost point, 1 = ghost point
                    mask_label = (event_label == self._num_classes).long()
                    # loss_mask = self.cross_entropy(event_ghost, mask_label)
                    num_ghost_points = (mask_label == 1).sum().float()
                    num_nonghost_points = (mask_label == 0).sum().float()
                    fraction = num_ghost_points / (num_ghost_points + num_nonghost_points)
                    weight = torch.stack([fraction, 1. - fraction]).float()
                    loss_mask = torch.nn.functional.cross_entropy(event_ghost, mask_label, weight=weight)
                    mask_loss += loss_mask
                    # mask_loss += torch.mean(loss_mask)

                    # Accuracy of ghost mask: fraction of correcly predicted
                    # points, whether ghost or nonghost
                    with torch.no_grad():
                        predicted_mask = torch.argmax(event_ghost, dim=-1)

                        # Accuracy ghost2ghost = fraction of correcly predicted
                        # ghost points as ghost points
                        if float(num_ghost_points.item()) > 0:
                            ghost2ghost += (predicted_mask[event_label == 5] == 1).sum().item() / float(num_ghost_points.item())

                        # Accuracy noghost2noghost = fraction of correctly predicted
                        # non ghost points as non ghost points
                        if float(num_nonghost_points.item()) > 0:
                            nonghost2nonghost += (predicted_mask[event_label < 5] == 0).sum().item() / float(num_nonghost_points.item())

                        # Global ghost predictions accuracy
                        acc_mask = predicted_mask.eq_(mask_label).sum().item() / float(predicted_mask.nelement())
                        mask_acc += acc_mask

                    # Now mask to compute the rest of UResNet loss
                    mask = event_label < self._num_classes
                    event_segmentation = event_segmentation[mask]
                    event_label = event_label[mask]
                else:
                    # check and warn about invalid labels
                    unique_label,unique_count = torch.unique(event_label,return_counts=True)
                    if (unique_label >= self._num_classes).long().sum():
                        print('Invalid semantic label found (will be ignored)')
                        print('Semantic label values:',unique_label)
                        print('Label counts:',unique_count)
                    # Now mask to compute the rest of UResNet loss
                    mask = event_label < self._num_classes
                    event_segmentation = event_segmentation[mask]
                    event_label = event_label[mask]

                if event_label.shape[0] > 0:  # FIXME how to handle empty mask?
                    # Loss for semantic segmentation
                    if self._weight_loss:
                        class_count = [(event_label == c).sum().float() for c in range(self._num_classes)]
                        sum_class_count = len(event_label)
                        w = torch.Tensor([sum_class_count / c if c.item() > 0 else 0. for c in class_count]).float()
                        if torch.cuda.is_available():
                            w = w.cuda()
                        #print(class_count, w, class_count[0].item() > 0)
                        loss_seg = torch.nn.functional.cross_entropy(event_segmentation, event_label, weight=w)
                    else:
                        loss_seg = self.cross_entropy(event_segmentation, event_label)
                        if weights is not None:
                            loss_seg *= weights[i][batch_index][:, -1].float()
                    if weights is not None:
                        uresnet_loss += torch.sum(loss_seg)/torch.sum(weights[i][batch_index][:,-1].float())
                    else:
                        uresnet_loss += torch.mean(loss_seg)

                    # Accuracy for semantic segmentation
                    with torch.no_grad():
                        predicted_labels = torch.argmax(event_segmentation, dim=-1)
                        acc = predicted_labels.eq_(event_label).sum().item() / float(predicted_labels.nelement())
                        uresnet_acc += acc

                        # Class accuracy
                        for c in range(self._num_classes):
                            class_mask = event_label == c
                            class_count = class_mask.sum().item()
                            if class_count > 0:
                                uresnet_acc_class[c] += predicted_labels[class_mask].sum().item() / float(class_count)
                                count_class[c] += 1

                count += 1

        if self._ghost:
            results = {
                'accuracy': uresnet_acc/count,
                'loss': (self._alpha * uresnet_loss + self._beta * mask_loss)/count,
                'mask_acc': mask_acc / count,
                'mask_loss': self._beta * mask_loss / count,
                'uresnet_loss': self._alpha * uresnet_loss / count,
                'uresnet_acc': uresnet_acc / count,
                'ghost2ghost': ghost2ghost / count,
                'nonghost2nonghost': nonghost2nonghost / count
            }
        else:
            results = {
                'accuracy': uresnet_acc/count,
                'loss': uresnet_loss/count
            }
        for c in range(self._num_classes):
            if count_class[c] > 0:
                results['accuracy_class_%d' % c] = uresnet_acc_class[c]/count_class[c]
            else:
                results['accuracy_class_%d' % c] = -1.
        return results
