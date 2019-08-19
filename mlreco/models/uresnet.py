from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class UResNet(torch.nn.Module):
    def __init__(self, cfg):
        import sparseconvnet as scn
        super(UResNet, self).__init__()
        self._model_config = cfg['modules']['uresnet']

        self._dimension = self._model_config.get('data_dim', 3)
        num_strides = self._model_config.get('num_strides', 5)
        spatial_size = self._model_config.get('spatial_size', 512)
        num_classes = self._model_config.get('num_classes', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)

        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level

        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(self._dimension, spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           scn.UNet(self._dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(self._dimension))
        self.linear = torch.nn.Linear(m, num_classes)

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        point_cloud, = input
        coords = point_cloud[:, :self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:].float()
        x = self.sparseModel((coords, features))
        x = self.linear(x)
        return {"segmentation" : [x]}


class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, label, weight=None):
        """
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        """
        # TODO Add weighting
        segmentation = outputs['segmentation']
        assert len(segmentation) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        batch_ids = [d[:, -2] for d in label]
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
