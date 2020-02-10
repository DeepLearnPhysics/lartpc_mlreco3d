# CNN feature extractor for Cluster GNN
import torch
from mlreco.models.layers.cnn_encoder import EncoderModel

class ClustCNNNodeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce node features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustCNNNodeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = EncoderModel(model_config)
        
    def forward(self, data, clusts):

        # Use cluster ID as a batch ID, pass through CNN
        cnn_data = torch.empty((0,5), dtype=torch.float)
        device = cnn_data.device
        for i, c in enumerate(clusts):
            cnn_data = torch.cat((cnn_data, data[c,:5].float()))
            cnn_data[-len(c):,3] = i*torch.ones(len(c)).to(device)

        return self.encoder(cnn_data) 

class ClustCNNEdgeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce edge features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustCNNEdgeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = EncoderModel(model_config)
        
    def forward(self, data, clusts, edge_index):

        # Use edge ID as a batch ID, pass through CNN
        cnn_data = torch.empty((0, 5), dtype=torch.float)
        device = cnn_data.device
        for i, e in enumerate(edge_index.T):
            ci, cj = clusts[e[0]], clusts[e[1]]
            cnn_data = torch.cat((cnn_data, data[ci,:5].float()))
            cnn_data = torch.cat((cnn_data, data[cj,:5].float()))
            cnn_data[-len(ci)-len(cj):,3] = i*torch.ones(len(ci)+len(cj)).to(device)

        return self.encoder(cnn_data) 

