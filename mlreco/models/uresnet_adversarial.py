from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from .uresnet_lonely import SegmentationLoss

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class UResNetAdversarial(torch.nn.Module):
    """
    UResNetAdversarial

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
        single non-ghost class and perform 2-classes segmentation (deghosting).
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

    MODULES = ['uresnet_adversarial']

    def __init__(self, cfg, name="uresnet_lonely"):
        super(UResNetAdversarial, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg[name]

        # Whether to compute ghost mask separately or not
        self._ghost = self._model_config.get('ghost', False)
        self._ghost_label = self._model_config.get('ghost_label', -1)
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        spatial_size = self._model_config.get('spatial_size', 512)
        num_classes = self._model_config.get('num_classes', 5)
        self.lam = cfg['discriminator'].get('lambda', 1.)

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

        self.grad_reverse = GradientReversal(self.lam)

        self.linear_disc = torch.nn.Linear(m, 2)

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

        x = self.grad_reverse(x)
        x_disc = self.linear_disc(x)

        res = {
            'segmentation': [x_seg],
            'discrimination': [x_disc],
            'ppn_feature_enc' : [feature_ppn],
            'ppn_feature_dec' : [feature_ppn2]
        }
        if self._ghost:
            res['ghost'] = [x_ghost]
        elif self._ghost_label > -1:
            res['ghost'] = [x_seg]
        return res


class AdversarialLoss(torch.nn.modules.loss._Loss):
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

    def __init__(self, cfg, reduction='mean'):
        super(AdversarialLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.discriminator_loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, result, label, data_label):
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
        # Apply the UResNet classification loss only on simulation voxels
        sim_masks = [data_label[i][:,-1] > 0. for i in range(len(label))]
        results = self.uresnet_loss({'segmentation': [result['segmentation'][i][sim_masks[i]] for i in range(len(label))]},
                                    [label[i][sim_masks[i]] for i in range(len(label))])
        results['seg_loss'] = float(results['loss'])
        results['seg_accuracy'] = float(results['accuracy'])

        # Apply the discriminator loss
        disc_loss = self.discriminator_loss(result['discrimination'][0], data_label[0][:,-1].long())
        disc_acc  = float(torch.sum(torch.argmax(result['discrimination'][0],dim=1) == data_label[0][:,-1].long()))/len(data_label[0])
        results['disc_loss'] = float(disc_loss)
        results['disc_accuracy'] = disc_acc

        results['loss'] += disc_loss
        results['accuracy'] += disc_acc
        results['accuracy'] /= 2.

        print('Segmentation accuracy:', results['seg_accuracy'])
        print('Disctriminator accuracy:', results['disc_accuracy'])

        return results
