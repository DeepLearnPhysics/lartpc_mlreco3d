from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch_geometric.nn import MetaLayer, NNConv

    
class EncoderModel(torch.nn.Module):

    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        import sparseconvnet as scn
        
        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg
            
        #Take the parameters from the config
        self._dimension = self.model_config.get('dimension', 3)
        self.num_strides = self.model_config.get('num_stride', 4)
        self.m =  self.model_config.get('features_per_pixel', 4)
        self.nInputFeatures = self.model_config.get('input_features_per_pixel', 1)
        self.leakiness = self.model_config.get('leakiness_encoder', 0)
        self.spatial_size = self.model_config.get('input_spatial_size', 1024) #Must be a power of 2
        
        
        self.out_spatial_size = int(self.spatial_size/4**(self.num_strides-1))
        self.output = self.m*self.out_spatial_size**3       
        
        nPlanes = [self.m for i in range(1, self.num_strides+1)]  # UNet number of features per level, was m*i        
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
        
        return x
