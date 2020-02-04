from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch_geometric.nn import MetaLayer, NNConv

    
class EncoderModel(torch.nn.Module):

    def __init__(self):
        super(EncoderModel, self).__init__()
        import sparseconvnet as scn
        
        self._dimension = 3
        
        reps = 2 # Conv block repetition factor
        kernel_size = 2
        num_strides = 4 #Was 5
        m = 4  # Final
        nInputFeatures = 1
        
        spatial_size = 1024
        out_spatial_size = int(spatial_size/4**(num_strides-1))
        
        self.output = m*out_spatial_size**3
        
        nPlanes = [m for i in range(1, num_strides+1)]  # UNet number of features per level, was m*i
        
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        leakiness = 0
       
        
        #Input for tpc voxels
        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        
        # Encoding TPC
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=leakiness)
        self.encoding_conv = scn.Sequential()
        for i in range(num_strides):
            module2 = scn.Sequential()
            if i < num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False)).add(
                    scn.AveragePooling(self._dimension, 2, 2))
                
            self.encoding_conv.add(module2)
        
        self.output = scn.Sequential().add(
           scn.SparseToDense(self._dimension,nPlanes[-1]))

                             
                             

    def forward(self, point_cloud):
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:].float()
        
        x = self.input((coords, features))
        
        print("checkpoint1, X: ", x.spatial_size)
        
        feature_maps = [x]
        feature_ppn = [x]        
        for i, layer in enumerate(self.encoding_conv):
            x = self.encoding_conv[i](x)
            #print("X: ", x.spatial_size)
            
        print("After encoding, X: ", x.spatial_size)
        
        x = self.output(x)
        
        x = x.view(-1,(x.size()[2]*x.size()[2]*x.size()[2]*x.size()[1]))
        
        print(x.size())
        
        return x
