# Mixed feature extractor. (Geo, CNN)
import torch
from mlreco.models.gnn.cluster_geo_encoder import ClustGeoNodeEncoder, ClustGeoEdgeEncoder
from mlreco.models.gnn.cluster_cnn_encoder import ClustCNNNodeEncoder, ClustCNNEdgeEncoder


class ClustMixNodeEncoder(torch.nn.Module):
    """
    Produces node features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self,model_config):
        super(ClustMixNodeEncoder, self).__init__()

        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = ClustGeoNodeEncoder(model_config['geo_encoder'])
        self.cnn_encoder = ClustCNNNodeEncoder(model_config['cnn_encoder'])

    def forward(self, data, clusts):
        return torch.cat(
            [
                self.geo_encoder(data, clusts),
                self.cnn_encoder(data, clusts),
            ],
            dim=1
        )

class ClustMixEdgeEncoder(torch.nn.Module):
    """
    Produces edge features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self,model_config):
        super(ClustMixEdgeEncoder, self).__init__()

        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = ClustGeoEdgeEncoder(model_config['geo_encoder'])
        self.cnn_encoder = ClustCNNEdgeEncoder(model_config['cnn_encoder'])

    def forward(self, data, clusts, edge_index):
        return torch.cat(
            [
                self.geo_encoder(data, clusts, edge_index),
                self.cnn_encoder(data, clusts, edge_index),
            ],
            dim=1
        )
