# Mixed feature extractor. (Geo, CNN)
import torch
from mlreco.models.gnn.cluster_geo_encoder import *
from mlreco.models.gnn.cluster_cnn_encoder import *


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


class ClustMixNodeEncoder2(torch.nn.Module):
    """
    Produces node features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self,model_config):
        super(ClustMixNodeEncoder2, self).__init__()
        # print(model_config)
        self.normalize = model_config.get('normalize', True)
        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = ClustGeoNodeEncoder(model_config['geo_encoder'])
        self.cnn_encoder = ClustCNNNodeEncoder2(model_config['cnn_encoder'])

    def forward(self, data, clusts):
        features_geo = self.geo_encoder(data, clusts)
        features_geo = features_geo / torch.norm(features_geo, dim=0)
        # print(features_geo.shape, features_geo)
        features_cnn = self.cnn_encoder(data, clusts)
        features_cnn = features_cnn / torch.norm(features_cnn, dim=0)
        # print(features_cnn.shape, features_cnn)
        features_mix = torch.cat([features_geo, features_cnn], dim=1)
        print(features_mix.shape)
        return features_mix


class ClustMixEdgeEncoder2(torch.nn.Module):
    """
    Produces edge features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self,model_config):
        super(ClustMixEdgeEncoder2, self).__init__()
        # print(model_config)
        self.normalize = model_config.get('normalize', True)
        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = ClustGeoEdgeEncoder(model_config['geo_encoder'])
        self.cnn_encoder = ClustCNNEdgeEncoder2(model_config['cnn_encoder'])

    def forward(self, data, clusts, edge_index):
        features_geo = self.geo_encoder(data, clusts, edge_index)
        features_geo = features_geo / torch.norm(features_geo, dim=0)
        # print(features_geo.shape, features_geo)
        features_cnn = self.cnn_encoder(data, clusts, edge_index)
        features_cnn = features_cnn / torch.norm(features_cnn, dim=0)
        # print(features_cnn.shape, features_cnn)
        features_mix = torch.cat([features_geo, features_cnn], dim=1)
        print(features_mix.shape)
        return features_mix
