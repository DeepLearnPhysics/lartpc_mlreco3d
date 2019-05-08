from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class UResNet(torch.nn.Module):
    def __init__(self, flags):
        import sparseconvnet as scn
        super(UResNet, self).__init__()
        self._flags = flags
        dimension = flags.DATA_DIM
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = flags.URESNET_FILTERS  # Unet number of features
        nPlanes = [i*m for i in range(1, flags.URESNET_NUM_STRIDES+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 1
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, flags.SPATIAL_SIZE, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           #scn.UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = torch.nn.Linear(m, flags.NUM_CLASS)

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        point_cloud, = input
        coords = torch.cat([point_cloud[:, :-2], point_cloud[:, -1][:, None]], dim=1).float()
        features = point_cloud[:, -1][:, None].float()
        x = self.sparseModel((coords, features))
        x = self.linear(x)
        return [[x]]


class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, flags, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, segmentation, label, weight=None):
        """
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has only 1 element because UResNet returns only 1 element.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        weight can be None.
        """
        assert len(segmentation[0]) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        batch_ids = [d[:, -1] for d in label]
        total_loss = 0
        total_acc = 0
        total_count = 0
        # Loop over ?
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[0][i][batch_index]
                event_label = label[i][:, -2][batch_index]
                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                if weight is not None:
                    event_weight = weight[i][batch_index]
                    event_weight = torch.squeeze(event_weight, dim=-1).float()
                    total_loss += torch.mean(loss_seg * event_weight)
                else:
                    total_loss += torch.mean(loss_seg)
                total_count += 1

                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() / float(predicted_labels.nelement())
                total_acc += acc

        return {
            'accuracy': total_acc,
            'loss_seg': total_loss
        }
