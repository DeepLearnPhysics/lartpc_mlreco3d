# Mixed feature extractor. (Geo, CNN)
import torch
from .geometric import *
from .cnn import *
from pprint import pprint

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
        # print(features_mix.shape)
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
        # print(features_mix.shape)
        return features_mix


class ClustMixNodeEncoder3(torch.nn.Module):
    """
    Produces node features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self,model_config):
        super(ClustMixNodeEncoder3, self).__init__()
        # print("ClustMixNodeEncoder3 = ", model_config)
        self.normalize = model_config.get('normalize', True)
        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = ClustGeoNodeEncoder(model_config['geo_encoder'])
        # pprint(model_config['cnn_encoder'])
        self.cnn_encoder = ClustCNNNodeEncoder2(model_config['cnn_encoder'])

        if self.geo_encoder.more_feats:
            node_feats = 19
        else:
            node_feats = 16

        self.bn1 = torch.nn.BatchNorm1d(node_feats)
        self.bn2 = torch.nn.BatchNorm1d(self.cnn_encoder.encoder.num_features)
        self.num_features = node_feats + self.cnn_encoder.encoder.num_features
        self.linear = torch.nn.Linear(self.num_features, self.num_features)
        self.elu = torch.nn.functional.elu


    def forward(self, data, clusts):
        features_geo = self.geo_encoder(data, clusts)
        features_geo = self.bn1(features_geo)
        features_cnn = self.cnn_encoder(data, clusts)
        features_cnn = self.bn2(features_cnn)
        features_mix = torch.cat([features_geo, features_cnn], dim=1)
        out = self.elu(features_mix)
        out = self.linear(out)
        # print(out.shape)
        return out


class ClustMixEdgeEncoder3(torch.nn.Module):
    """
    Produces edge features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self,model_config):
        super(ClustMixEdgeEncoder3, self).__init__()
        # print(model_config)
        self.normalize = model_config.get('normalize', True)
        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = ClustGeoEdgeEncoder(model_config['geo_encoder'])
        self.cnn_encoder = ClustCNNEdgeEncoder2(model_config['cnn_encoder'])

        node_feats = 19
        self.bn1 = torch.nn.BatchNorm1d(node_feats)
        self.bn2 = torch.nn.BatchNorm1d(self.cnn_encoder.encoder.num_features)
        self.num_features = node_feats + self.cnn_encoder.encoder.num_features
        self.linear = torch.nn.Linear(self.num_features, self.num_features)
        self.elu = torch.nn.functional.elu

    def forward(self, data, clusts, edge_index):
        features_geo = self.geo_encoder(data, clusts, edge_index)
        features_geo = self.bn1(features_geo)
        features_cnn = self.cnn_encoder(data, clusts, edge_index)
        features_cnn = self.bn2(features_cnn)
        features_mix = torch.cat([features_geo, features_cnn], dim=1)
        out = self.elu(features_mix)
        out = self.linear(out)
        # print(out.shape)
        return out
