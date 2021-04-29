# Geometric feature extractor for Cluster GNN
import torch
from mlreco.models.gnn.message_passing.nnconv_2 import NNConvModel
from mlreco.utils.gnn.data import vtx_features, edge_features, cluster_vtx_features
from mlreco.utils.gnn.network import complete_graph, kdtree_graph

class ClustGNNNodeEncoder(torch.nn.Module):
    """
    Each cluster is considered to be a batch. Each voxel within a
    cluster is given node features. A graph is built
    on the cluster, edge are provided features and message is passed.
    The updated global features are the node features.

    """
    def __init__(self, model_config):
        super(ClustGNNNodeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder  = NNConvModel(model_config)
        self.network  = model_config.get('network','knn')
        self.max_dist = model_config.get('max_dist','max_dist')

    def forward(self, data, clusts):

        # Use cluster ID as a batch ID, pass through GNN
        # Build graph, extract voxel/global features
        device = data.device
        dtype = data.dtype
        gnn_data = torch.empty((0,4), device=device, dtype=torch.float)
        x  = torch.empty((0,16), device=device, dtype=torch.float)
        u  = torch.empty((0,16), device=device, dtype=torch.float)
        xb = torch.empty((0), device=device, dtype=torch.long)
        for i, c in enumerate(clusts):
            gnn_data = torch.cat((gnn_data, data[c,:4].float()))
            gnn_data[-len(c):,3] = i*torch.ones(len(c)).to(device)
            x  = torch.cat((x, torch.tensor(vtx_features(data[c].detach().cpu().numpy(), max_dist=self.max_dist), device=device, dtype=torch.float)))
            u  = torch.cat((u, torch.tensor(cluster_vtx_features(data.detach().cpu().numpy(), [c]), device=device, dtype=torch.float)))
            xb = torch.cat((xb, torch.full([len(c)], i, dtype=torch.long)))

        # Build a network that connects neighbour nodes together (touching nodes)
        #edge_index = complete_graph(gnn_data[:,3], max_dist=1.99999)
        edge_index = kdtree_graph(gnn_data, 3)

        # Get edge features
        e = torch.tensor(edge_features(gnn_data, edge_index), device=device)
        index = torch.tensor(edge_index, device=device, dtype=torch.long)

        output = self.encoder(x, index, e, u, xb)

        return output['global_pred'][0]

class ClustGNNEdgeEncoder(torch.nn.Module):
    """
    Each cluster is considered to be a batch. Each voxel within a
    cluster is given node features. A graph (knn/delaunay) is built
    on the cluster, edge are provided features and message is passed.
    The updated global features are the node features.

    """
    def __init__(self, model_config):
        super(ClustGNNEdgeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = NNConvModel(model_config)

    def forward(self, data, clusts, edge_index):

        raise NotImplementedError('Not implemented, use geometric edge features')
