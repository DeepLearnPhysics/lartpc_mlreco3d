# a basic sparse UResNet layer that expects to be fed data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class UResNet(torch.nn.Module):
    """
    UResNet

    For semantic segmentation, using sparse convolutions from SCN, but not the
    ready-made UNet from SCN library. 
    
    Can also be used in a chain, for example stacking PPN layers on top.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially.
    leak: float, optional
        Leakiness of ReLUs

    Returns
    -------
    N x K torch.float tensor
    where N is the number of voxels
    K is number of filters (set by filters)

    """

    def __init__(self, cfg, name="uresnet_lonely"):
        super(UResNet, self).__init__()
        import sparseconvnet as scn

        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg

        # Whether to compute ghost mask separately or not

        self._dimension = self.model_config.get('data_dim', 3)
        reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self.model_config.get('kernel_size', 2)
        num_strides = self.model_config.get('num_strides', 5)
        m = self.model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self.model_config.get('features', 1)
        spatial_size = self.model_config.get('spatial_size', 512)
        leakiness = self.model_config.get('leak', 0.0)

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        print("nPlanes: ", nPlanes)
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None

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


    def forward(self, point_cloud):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + k features
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:].float()

        x = self.input((coords, features))
        feature_maps = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            # print("encoding: ", i, "size: ", x.features.shape)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)

        # U-ResNet decoding
        for i, layer in enumerate(self.decoding_conv):
            # print("decoding: ", i, "size: ", x.features.shape)
            encoding_block = feature_maps[-i-2]
            # print("           ", "size: ", encoding_block.features.shape)
            x = layer(x)
            x = self.concat([encoding_block, x])
            x = self.decoding_blocks[i](x)

        x = self.output(x)
        return x