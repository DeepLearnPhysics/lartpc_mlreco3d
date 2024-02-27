import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

from mlreco.models.layers.common.uresnet_layers import UResNet
from collections import defaultdict
from mlreco.models.layers.common.activation_normalization_factories import activations_construct, normalizations_construct


class UResNet_Chain(nn.Module):
    """
    UResNet implementation. Typical configuration should look like:

    .. code-block:: yaml

        model:
          name: uresnet
          modules:
            uresnet_lonely:
              # Your config here

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    num_classes: int, default 5
    ghost: bool, default False
    ghost_label: int, default -1
    weight_loss: bool, default False
        Whether to weight the loss using class counts.
    alpha: float, default 1.0
        Weight for UResNet semantic segmentation loss.
    beta: float, default 1.0
        Weight for ghost/non-ghost segmentation loss.

    Returns
    ------
    segmentation : torch.Tensor
    finalTensor : torch.Tensor
    encoderTensors : list of torch.Tensor
    decoderTensors : list of torch.Tensor
    ghost : torch.Tensor
    ghost_sptensor : torch.Tensor

    See Also
    --------
    SegmentationLoss, mlreco.models.layers.common.uresnet_layers
    """

    INPUT_SCHEMA = [
        ['parse_sparse3d', (float,), (3, 1)]
    ]

    MODULES = ['uresnet_lonely']

    RETURNS = {
        'segmentation': ['tensor', 'input_data'],
        'finalTensor': ['tensor'],
        'encoderTensors': ['tensor_list'],
        'decoderTensors': ['tensor_list'],
        'ghost': ['tensor', 'input_data'],
        'ghost_sptensor': ['tensor']
    }

    def __init__(self, cfg, name='uresnet_lonely'):
        super(UResNet_Chain, self).__init__()
        self.model_config = cfg.get(name, {})
        self.num_classes = self.model_config.get('num_classes', 5)\

        # Parameters for Deghosting
        self.ghost = self.model_config.get('ghost', False)
        self.ghost_label = self.model_config.get('ghost_label', -1)

        self.net = UResNet(cfg, name=name)
        self.F = self.net.num_filters
        self.D = self.net.D

        self.output = [
            normalizations_construct(self.net.norm, self.F, **self.net.norm_args),
            #activations_construct(self.net.activation_name, **self.net.activation_args),
            activations_construct(self.net.activation_name, negative_slope=0.33),
            ]
        self.output = nn.Sequential(*self.output)
        self.linear_segmentation = ME.MinkowskiLinear(self.F, self.num_classes)

        if self.ghost:
            print("Ghost Masking is enabled for UResNet Segmentation")
            self.linear_ghost = ME.MinkowskiLinear(self.F, 2)

        # print('Total Number of Trainable Parameters (UResNet)= {}'.format(
        #             sum(p.numel() for p in self.parameters() if p.requires_grad)))
        # print(self)

    def forward(self, input):
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            res = self.net(x)
            feats = res['decoderTensors'][-1]
            feats = self.output(feats)
            seg = self.linear_segmentation(feats)
            out['segmentation'].append(seg.F)
            out['finalTensor'].append(res['finalTensor'])
            out['encoderTensors'].append(res['encoderTensors'])
            out['decoderTensors'].append(res['decoderTensors'])
            if self.ghost:
                ghost = self.linear_ghost(feats)
                out['ghost'].append(ghost.F)
                out['ghost_sptensor'].append(ghost)
        return out


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

    See Also
    --------
    UResNet_Chain
    """
    INPUT_SCHEMA = [
        ['parse_sparse3d', (int,), (3, 1)]
    ]

    RETURNS = {
        'accuracy': ('scalar',),
        'loss': ('scalar', ),
        'ghost_mask_accuracy': ('scalar',),
        'ghost_mask_loss': ('scalar',),
        'uresnet_accuracy': ('scalar',),
        'uresnet_loss': ('scalar',),
        'ghost2ghost_accuracy': ('scalar',),
        'nonghost2nonghost_accuracy' : ('scalar',)
    }

    def __init__(self, cfg, reduction='sum', batch_col=0):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._cfg = cfg.get('uresnet_lonely', {})
        self._ghost = self._cfg.get('ghost', False)
        self._ghost_label = self._cfg.get('ghost_label', -1)
        self._num_classes = self._cfg.get('num_classes', 5)
        self._alpha = self._cfg.get('alpha', 1.0)
        self._beta = self._cfg.get('beta', 1.0)
        self._weight_loss = self._cfg.get('weight_loss', False)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self._batch_col = batch_col

        for c in range(self._num_classes):
            self.RETURNS[f'accuracy_class_{c}'] = ('scalar',)

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
                    num_ghost_points = (mask_label == 1).sum().float()
                    num_nonghost_points = (mask_label == 0).sum().float()
                    fraction = num_ghost_points \
                             / (num_ghost_points + num_nonghost_points)
                    weight = torch.stack([fraction, 1. - fraction]).float()
                    loss_mask = torch.nn.functional.cross_entropy(event_ghost,
                                                                  mask_label,
                                                                  weight=weight)
                    mask_loss += loss_mask
                    # mask_loss += torch.mean(loss_mask)

                    # Accuracy of ghost mask: fraction of correcly predicted
                    # points, whether ghost or nonghost
                    with torch.no_grad():
                        predicted_mask = torch.argmax(event_ghost, dim=-1)

                        # Accuracy ghost2ghost = fraction of correcly predicted
                        # ghost points as ghost points
                        if float(num_ghost_points.item()) > 0:
                            ghost2ghost += (predicted_mask[event_label == self._num_classes] == 1).sum().item() \
                                        / float(num_ghost_points.item())

                        # Accuracy noghost2noghost = fraction of correctly predicted
                        # non ghost points as non ghost points
                        if float(num_nonghost_points.item()) > 0:
                            nonghost2nonghost += (predicted_mask[event_label < self._num_classes] == 0).sum().item() \
                                              / float(num_nonghost_points.item())

                        # Global ghost predictions accuracy
                        acc_mask = predicted_mask.eq_(mask_label).sum().item() \
                                 / float(predicted_mask.nelement())
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
                        w = w.to(event_label.device)
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
                'accuracy': uresnet_acc/count if count else 1.,
                'loss': (self._alpha * uresnet_loss + self._beta * mask_loss)/count if count else self._alpha * uresnet_loss + self._beta * mask_loss,
                'ghost_mask_accuracy': mask_acc / count if count else 1.,
                'ghost_mask_loss': self._beta * mask_loss / count if count else self._beta * mask_loss,
                'uresnet_accuracy': uresnet_acc / count if count else 1.,
                'uresnet_loss': self._alpha * uresnet_loss / count if count else self._alpha * uresnet_loss,
                'ghost2ghost_accuracy': ghost2ghost / count if count else 1.,
                'nonghost2nonghost_accuracy': nonghost2nonghost / count if count else 1.
            }
        else:
            results = {
                'accuracy': uresnet_acc/count if count else 1.,
                'loss': uresnet_loss/count if count else uresnet_loss
            }
        for c in range(self._num_classes):
            if count_class[c] > 0:
                results['accuracy_class_%d' % c] = uresnet_acc_class[c]/count_class[c]
            else:
                results['accuracy_class_%d' % c] = 1.
        return results
