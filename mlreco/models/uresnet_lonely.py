from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class UResNet(torch.nn.Module):
    """
    UResNet

    For semantic segmentation, using sparse convolutions from SCN, but not the
    ready-made UNet from SCN library. The option `ghost` allows to train at the
    same time for semantic segmentation between N classes (e.g. particle types)
    and ghost points masking.

    Can also be used in a chain, for example stacking PPN layers on top.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    ghost : bool, optional
        Whether to compute ghost mask separately or not. See SegmentationLoss
        for more details.
    ghost_label: int, optional
        If specified, then will collapse all classes other than ghost_label into
        single non-ghost class and perform 2-classes segmentatiion (deghosting).
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially.

    Returns
    -------
    list
        In order:
        - segmentation scores (N, num_classes)
        - feature maps of encoding path
        - feature maps of decoding path
        - if `ghost`, segmentation scores for deghosting (N, 2)
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,), (3, 1)]
    ]

    def __init__(self, cfg, name="uresnet_lonely"):
        super(UResNet, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg['modules'][name]

        # Whether to compute ghost mask separately or not
        self._ghost = self._model_config.get('ghost', False)
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        spatial_size = self._model_config.get('spatial_size', 512)
        num_classes = self._model_config.get('num_classes', 5)

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None
        leakiness = 0

        def block(m, a, b):  # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, False)))
             ).add(scn.AddTable())

        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        # Encoding
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=leakiness)
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        module = scn.Sequential()
        for i in range(num_strides):
            module = scn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            self.encoding_block.add(module)
            module2 = scn.Sequential()
            if i < num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))
            self.encoding_conv.add(module2)
        self.encoding = module

        # Decoding
        self.decoding_conv, self.decoding_blocks = scn.Sequential(), scn.Sequential()
        for i in range(num_strides-2, -1, -1):
            module1 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
            self.decoding_conv.add(module1)
            module2 = scn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i] * (2 if j == 0 else 1), nPlanes[i])
            self.decoding_blocks.add(module2)

        self.output = scn.Sequential().add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(self._dimension))

        self.linear = torch.nn.Linear(m, num_classes)
        if self._ghost:
            self.linear_ghost = torch.nn.Linear(m, 2)

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        point_cloud, = input
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:].float()

        x = self.input((coords, features))
        feature_maps = [x]
        feature_ppn = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)
            feature_ppn.append(x)

        # U-ResNet decoding
        feature_ppn2 = [x]
        for i, layer in enumerate(self.decoding_conv):
            encoding_block = feature_maps[-i-2]
            x = layer(x)
            x = self.concat([encoding_block, x])
            x = self.decoding_blocks[i](x)
            feature_ppn2.append(x)

        x = self.output(x)
        x_seg = self.linear(x)  # Output of UResNet
        if self._ghost:
            x_ghost = self.linear_ghost(x)

        res = {
            'segmentation': [x_seg],
            'ppn_feature_enc' : [feature_ppn],
            'ppn_feature_dec' : [feature_ppn2]
        }
        if self._ghost:
            res['ghost'] = [x_ghost]
        return res


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

    def __init__(self, cfg, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['modules']['uresnet_lonely']
        self._ghost = self._cfg.get('ghost', False)
        self._ghost_label = self._cfg.get('ghost_label', -1)
        self._num_classes = self._cfg.get('num_classes', 5)
        self._alpha = self._cfg.get('alpha', 1.0)
        self._beta = self._cfg.get('beta', 1.0)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, result, label):
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
        batch_ids = [d[:, -2] for d in label]
        uresnet_loss, uresnet_acc = 0., 0.
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

                if event_label.shape[0] > 0:  # FIXME how to handle empty mask?
                    # Loss for semantic segmentation
                    loss_seg = self.cross_entropy(event_segmentation, event_label)
                    uresnet_loss += torch.mean(loss_seg)

                    # Accuracy for semantic segmentation
                    with torch.no_grad():
                        predicted_labels = torch.argmax(event_segmentation, dim=-1)
                        acc = predicted_labels.eq_(event_label).sum().item() / float(predicted_labels.nelement())
                        uresnet_acc += acc

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
        return results
