from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer, NNConv
import sparseconvnet as scn

from mlreco.models.layers.uresnet import UResNetEncoder
from pprint import pprint

class EncoderModel(torch.nn.Module):

    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        import sparseconvnet as scn

        # Get the model input parameters
        model_config = cfg

        # Take the parameters from the config
        self._dimension = model_config.get('dimension', 3)
        self.num_strides = model_config.get('num_stride', 4)
        self.m =  model_config.get('feat_per_pixel', 4)
        self.nInputFeatures = model_config.get('input_feat_enc', 1)
        self.leakiness = model_config.get('leakiness_enc', 0)
        self.spatial_size = model_config.get('inp_spatial_size', 1024) #Must be a power of 2
        self.feat_aug_mode = model_config.get('feat_aug_mode', 'constant')
        self.use_linear_output = model_config.get('use_linear_output', False)
        self.num_output_feats = model_config.get('num_output_feats', 64)

        self.out_spatial_size = int(self.spatial_size/4**(self.num_strides-1))
        self.output = self.m*self.out_spatial_size**3

        nPlanes = [self.m for i in range(1, self.num_strides+1)]  # UNet number of features per level
        if self.feat_aug_mode == 'linear':
            nPlanes = [self.m * i for i in range(1, self.num_strides + 1)]
        elif self.feat_aug_mode == 'power':
            nPlanes = [self.m * pow(2, i) for i in range(self.num_strides)]
        elif self.feat_aug_mode != 'constant':
            raise ValueError('Feature augmentation mode not recognized')
        kernel_size = 2
        downsample = [kernel_size, 2]  # [filter size, filter stride]


        #Input for tpc voxels
        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, self.spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, self.nInputFeatures, self.m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()

        # Encoding TPC
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=self.leakiness)
        self.encoding_conv = scn.Sequential()
        for i in range(self.num_strides):
            module2 = scn.Sequential()
            if i < self.num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False)).add(
                    scn.AveragePooling(self._dimension, 2, 2))

            self.encoding_conv.add(module2)

        self.output = scn.Sequential().add(
           scn.SparseToDense(self._dimension,nPlanes[-1]))

        if self.use_linear_output:
            input_size = nPlanes[-1] * (self.out_spatial_size ** self._dimension)
            self.linear = torch.nn.Linear(input_size, self.num_output_feats)

    def forward(self, point_cloud):
        # We separate the coordinate tensor from the feature tensor
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:].float()

        x = self.input((coords, features))

        # We send x through all the encoding layers
        feature_maps = [x]
        feature_ppn = [x]
        for i, layer in enumerate(self.encoding_conv):
            x = self.encoding_conv[i](x)

        x = self.output(x)

        #Then we flatten the vector
        x = x.view(-1,(x.size()[2]*x.size()[2]*x.size()[2]*x.size()[1]))

        # Go through linear layer if necessary
        if self.use_linear_output:
            x = self.linear(x)
            x = x.view(-1, self.num_output_feats)

        return x


class Flatten(nn.Module):

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        batch_size = input.shape[0]
        return input.view(batch_size, -1)


class ResidualEncoder(UResNetEncoder):

    def __init__(self, cfg, name='res_encoder'):
        super(ResidualEncoder, self).__init__(cfg,'uresnet_encoder')
        self.model_config = cfg[name]
        self.num_features = self.model_config.get('num_features', 32)

        self.input_layer = scn.Sequential().add(
           scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(
               self.dimension, self.nInputFeatures, self.num_filters, 3, False))

        self.coordConv = self.model_config.get('coordConv', True)
        self.pool_mode = self.model_config.get('pool_mode', 'max')

        self.final_tensor_shape = self.spatial_size // (2**(self.num_strides-1))
        #print("Final Tensor Shape = ", self.final_tensor_shape)

        if self.pool_mode == 'max':
            self.output = scn.SparseToDense(self.dimension, self.nPlanes[-1])
            self.pool = nn.MaxPool3d(self.final_tensor_shape)
        elif self.pool_mode == 'flatten':
            self.output = scn.SparseToDense(self.dimension, self.nPlanes[-1])
            self.pool = Flatten()
        elif self.pool_mode == 'avg':
            self.output = scn.SparseToDense(self.dimension, self.nPlanes[-1])
            self.pool = nn.AvgPool3d(self.final_tensor_shape)
        else:
            self.output = scn.Sequential().add(
                scn.Convolution(
                    self.dimension, self.nPlanes[-1], self.nPlanes[-1],
                    self.final_tensor_shape, 1,
                    self.allow_bias)).add(
                scn.SparseToDense(self.dimension, self.nPlanes[-1]))
            self.pool = nn.MaxPool3d(1)
        self.linear = nn.Linear(self.nPlanes[-1], self.num_features)


    def forward(self, point_cloud):
        '''
        Vanilla UResNet Encoder

        INPUTS:
            - x (scn.SparseConvNetTensor): output from inputlayer (self.input)

        RETURNS:
            - features_encoder (list of SparseConvNetTensor): list of feature
            tensors in encoding path at each spatial resolution.
        '''
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        features = features[:, -1].view(-1, 1)
        batch_size = coords[:, 3].unique().shape[0]
        # Concat normalized image coordinates
        if self.coordConv:
            normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input_layer((coords, features))

        features_enc = [x]
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            features_enc.append(x)
            x = self.encoding_conv[i](x)
        out = self.output(x)
        out = self.pool(out).view(batch_size, -1)
        out = self.linear(out)
        return out
