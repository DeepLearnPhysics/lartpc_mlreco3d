from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch_geometric.nn import MetaLayer, NNConv


#Strided convolution + average pooling, without PCA.
class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak):
        """
        Basic model for making edge predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
        """
        super(EdgeModel, self).__init__()

        self.edge_pred_mlp = torch.nn.Sequential(torch.nn.Linear(2*node_in + edge_in, 64),
                                 torch.nn.LeakyReLU(leak),
                                 torch.nn.Linear(64, 32),
                                 torch.nn.LeakyReLU(leak),
                                 torch.nn.Linear(32, 16),
                                 torch.nn.LeakyReLU(leak),
                                 torch.nn.Linear(16,8),
                                 torch.nn.LeakyReLU(leak),
                                 torch.nn.Linear(8,2)
                                )

    def forward(self, src, dest, edge_attr, u, batch):
        return self.edge_pred_mlp(torch.cat([src, dest, edge_attr], dim=1))


    
class UResGNet(torch.nn.Module):

    def __init__(self):
        super(UResGNet, self).__init__()
        import sparseconvnet as scn

        # Whether to compute ghost mask separately or not
        self._dimension_tpc = 3
        
        self.nb_pmt_features = 1
        self.nb_pmt = 180
        
        self.nb_pca_features = 16
        
        reps = 2 # Conv block repetition factor
        kernel_size = 2
        num_strides_tpc = 6 #Was 5
        m = 4  # Unet number of features
        nInputFeatures_tpc = 1
        
        self.num_mp = 3
        
        spatial_size_tpc = 8192
        out_spatial_size_tpc = int(spatial_size_tpc/4**(num_strides_tpc-1))
        
        self.edge_in = m*out_spatial_size_tpc**3 + self.nb_pmt*self.nb_pmt_features
        self.node_in = 2
        
        nPlanes_tpc = [m for i in range(1, num_strides_tpc+1)]  # UNet number of features per level, was m*i
        
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        leakiness = 0
       
        
        #Input for tpc voxels
        self.input_tpc = scn.Sequential().add(
           scn.InputLayer(self._dimension_tpc, spatial_size_tpc, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension_tpc, nInputFeatures_tpc, m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        
        # Encoding TPC
        self.bn = scn.BatchNormLeakyReLU(nPlanes_tpc[0], leakiness=leakiness)
        self.encoding_conv_tpc = scn.Sequential()
        for i in range(num_strides_tpc):
            module2 = scn.Sequential()
            if i < num_strides_tpc-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes_tpc[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension_tpc, nPlanes_tpc[i], nPlanes_tpc[i+1],
                        downsample[0], downsample[1], False)).add(
                    scn.AveragePooling(self._dimension_tpc, 2, 2))
                
            self.encoding_conv_tpc.add(module2)
        
        self.output_tpc = scn.Sequential().add(
           scn.SparseToDense(self._dimension_tpc,nPlanes_tpc[-1]))

        
         # perform batch normalization
        self.bn_edge = torch.nn.BatchNorm1d(self.edge_in)
        
        self.nn = torch.nn.ModuleList()
        self.layer = torch.nn.ModuleList()
        ninput = self.node_in
        noutput = max(2*self.node_in, 32)
        
        for i in range(self.num_mp):
            self.nn.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.edge_in, ninput),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(ninput, ninput*noutput),
                    torch.nn.LeakyReLU()
                )
            )
            self.layer.append(
                NNConv(ninput, noutput, self.nn[i], aggr='add')
            )
            ninput = noutput
        
        # final prediction layer
        self.edge_predictor = MetaLayer(EdgeModel(noutput, self.edge_in, 0.1))
                             
                             

    def forward(self, input):
        """
        point_cloud[0] is a list of tpc voxels (3 spatial coordinates + 1 batch coordinate + 1 feature)
        point_cloud[1] is a list of pmt voxels (3 spatial coordinates + 1 batch coordinate + 1 feature)
        """
        point_cloud_tpc = input[0]
        coords_tpc = point_cloud_tpc[:, 0:self._dimension_tpc+1].float()
        features_tpc = point_cloud_tpc[:, self._dimension_tpc+1:].float()
        
        point_cloud_tpc_pca = input[1].float()
        
        x_tpc_pca = point_cloud_tpc_pca.view(-1,self.nb_pca_features)
        
        point_cloud_pmt = input[2].float()
        
        x_pmt = point_cloud_pmt.view(-1,self.nb_pmt*self.nb_pmt_features)
        
        x_tpc = self.input_tpc((coords_tpc, features_tpc))
        
        print(" XTPC: ", x_tpc)
        #print("checkpoint1, XTPC: ", x_tpc.spatial_size)
        
        feature_maps_tpc = [x_tpc]
        feature_ppn_tpc = [x_tpc]        
        for i, layer in enumerate(self.encoding_conv_tpc):
            x_tpc = self.encoding_conv_tpc[i](x_tpc)
            #print("XTPC: ", x_tpc.spatial_size)
            
        #print("After encoding, XTPC: ", x_tpc.spatial_size)
        
        x_tpc = self.output_tpc(x_tpc)
        
        x_tpc = x_tpc.view(-1,(x_tpc.size()[2]*x_tpc.size()[2]*x_tpc.size()[2]*x_tpc.size()[1]))
           
        x = input[3].float()
        
        x_tpc = torch.cat((x_tpc, x_pmt), -1)
        
        #print(x_tpc.size())
        
        e = self.bn_edge(x_tpc)
        
        edge_index = input[4].long()
        
        xbatch = input[5].long() 
        
        # go through layers
        for i in range(self.num_mp):
            x = self.layer[i](x, edge_index, e)
        
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        
        return e