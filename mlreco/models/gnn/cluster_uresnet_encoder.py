# Using trained UResNet for feature extractor
import torch
from mlreco.models.uresnet_lonely import UResNet
import sparseconvnet as scn
import os

class ClustUResNetNodeEncoder(torch.nn.Module):
    """
    Uses a UResNet to produce node features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustUResNetNodeEncoder, self).__init__()

        # Initialize the UResNet
        cfg = model_config['uresnet_lonely']
        self.encoder = UResNet(model_config)

        # Manual loading weight
        self.uresnet_weight_file = model_config.get('uresnet_weight_file', None)
        if self.uresnet_weight_file != None:
            if not os.path.isfile(self.uresnet_weight_file):
                raise ValueError('File ' + str(self.uresnet_weight_file) + ' not exist!')
            # continue loading
            checkpoint = torch.load(self.uresnet_weight_file)
            # regulate the checkpoint
            for name in self.encoder.state_dict().keys():
                other_name = 'module.'+name
                if other_name in checkpoint['state_dict'].keys():
                    checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(other_name)
            self.encoder.load_state_dict(checkpoint['state_dict'])
            self.encoder.eval()
            print("Manually restoring uresnet (node feature extractor) weight from " + str(self.uresnet_weight_file))


        # flag for whether to freeze, default True
        self._freeze = model_config.get("freeze_uresnet", True)
        if self._freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Freeze UResNet!")

        # to dense layer
        spatial_size = cfg.get('spatial_size', 512)
        num_strides = cfg.get('num_strides', 5)
        filters = cfg.get('filters', 16)
        self.input_features = ((spatial_size / (2**(num_strides-1)))**3)*num_strides*filters
        self.to_dense = scn.Sequential().add(
            scn.SparseToDense(
                cfg.get('data_dim', 3),
                num_strides*filters
            )
        )

        # linear layer
        self.output_features = model_config.get('output_feats', 64) # if output_feats<0, meaning no linear
        if self.output_features>0:
            self.linear = torch.nn.Linear(
                int(self.input_features),
                int(self.output_features)
            )


    def forward(self, data, clusts):
        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0, 5), device=device, dtype=torch.float)
        for i, c in enumerate(clusts):
            cnn_data = torch.cat((cnn_data, data[c, :5].float()))
            cnn_data[-len(c):, 3] = i * torch.ones(len(c)).to(device)

        out = self.encoder([cnn_data])['ppn_feature_dec'][0][0]
        out = self.to_dense(out)
        out = out.view(len(clusts), -1)

        if self.output_features>0:
            out = self.linear(out)

        return out


class ClustUResNetEdgeEncoder(torch.nn.Module):
    """
    Uses a UResNet to produce node features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustUResNetEdgeEncoder, self).__init__()

        # Initialize the UResNet
        cfg = model_config['uresnet_lonely']
        self.encoder = UResNet(model_config)

        # Manual loading weight
        self.uresnet_weight_file = model_config.get('uresnet_weight_file', None)
        if self.uresnet_weight_file != None:
            if not os.path.isfile(self.uresnet_weight_file):
                raise ValueError('File ' + str(self.uresnet_weight_file) + ' not exist!')
            # continue loading
            checkpoint = torch.load(self.uresnet_weight_file)
            # regulate the checkpoint
            for name in self.encoder.state_dict().keys():
                other_name = 'module.'+name
                if other_name in checkpoint['state_dict'].keys():
                    checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(other_name)
            self.encoder.load_state_dict(checkpoint['state_dict'])
            self.encoder.eval()
            print("Manually restoring uresnet (edge feature extractor) weight from " + str(self.uresnet_weight_file))

        # flag for whether to freeze, default True
        self._freeze = model_config.get("freeze_uresnet", True)
        if self._freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Freeze UResNet!")

        # to dense layer
        spatial_size = cfg.get('spatial_size', 512)
        num_strides = cfg.get('num_strides', 5)
        filters = cfg.get('filters', 16)
        self.input_features = ((spatial_size / (2**(num_strides-1)))**3)*num_strides*filters
        self.to_dense = scn.Sequential().add(
            scn.SparseToDense(
                cfg.get('data_dim', 3),
                num_strides * filters
            )
        )

        # linear layer
        self.output_features = model_config.get('output_feats', 64)  # if output_feats<0, meaning no linear
        if self.output_features > 0:
            self.linear = torch.nn.Linear(
                int(self.input_features),
                int(self.output_features)
            )


    def forward(self, data, clusts, edge_index):

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1]/2)
        undirected = (not edge_index.shape[1]%2 and [edge_index[1,0], edge_index[0,0]] == edge_index[:,half_idx].tolist())
        if undirected: edge_index = edge_index[:,:half_idx]

        # Use edge ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0, 5), device=device, dtype=torch.float)
        for i, e in enumerate(edge_index.T):
            ci, cj = clusts[e[0]], clusts[e[1]]
            cnn_data = torch.cat((cnn_data, data[ci,:5].float()))
            cnn_data = torch.cat((cnn_data, data[cj,:5].float()))
            cnn_data[-len(ci)-len(cj):,3] = i*torch.ones(len(ci)+len(cj)).to(device)

        feats = self.to_dense(self.encoder([cnn_data])['ppn_feature_dec'][0][0]).view(len(edge_index.T), -1)

        if self.output_features>0:
            feats = self.linear(feats)

        # If the graph is undirected, duplicate features
        if undirected:
            feats = torch.cat([feats, feats])

        return feats


