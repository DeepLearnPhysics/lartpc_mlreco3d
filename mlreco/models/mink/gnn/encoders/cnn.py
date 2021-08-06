import torch
import torch.nn as nn
from mlreco.models.mink.layers.cnn_encoder import SparseResidualEncoder


class ClustCNNMinkNodeEncoder(nn.Module):
    '''
    CNN Node Encoder using MinkowskiEngine Backend
    '''
    def __init__(self, model_config, **kwargs):
        super(ClustCNNMinkNodeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = SparseResidualEncoder(model_config)

    def forward(self, data, clusts):

        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0,5), device=device, dtype=torch.float)
        for i, c in enumerate(clusts):
            cnn_data = torch.cat((cnn_data, data[c,:5].float()))
            cnn_data[-len(c):,0] = i*torch.ones(len(c)).to(device)

        return self.encoder(cnn_data)


class ClustCNNMinkEdgeEncoder(nn.Module):
    """
    Uses a CNN to produce node features for cluster GNN

    """
    def __init__(self, model_config, **kwargs):
        super(ClustCNNMinkEdgeEncoder, self).__init__()
        # Initialize the CNN
        self.encoder = SparseResidualEncoder(model_config)

    def forward(self, data, clusts, edge_index):

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1]/2)
        undirected = not edge_index.shape[1] or (not edge_index.shape[1]%2 and [edge_index[1,0], edge_index[0,0]] == edge_index[:,half_idx].tolist())
        if undirected: edge_index = edge_index[:,:half_idx]

        # Use edge ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0, 5), device=device, dtype=torch.float)
        for i, e in enumerate(edge_index.T):
            ci, cj = clusts[e[0]], clusts[e[1]]
            cnn_data = torch.cat((cnn_data, data[ci,:5].float()))
            cnn_data = torch.cat((cnn_data, data[cj,:5].float()))
            cnn_data[-len(ci)-len(cj):,0] = i*torch.ones(len(ci)+len(cj)).to(device)

        feats = self.encoder(cnn_data)

        # If the graph is undirected, duplicate features
        if undirected:
            feats = torch.cat([feats,feats])

        return feats
