# Mixed feature extractor. (Geo, CNN)
import torch
from pprint import pprint
from ..factories import node_encoder_construct, edge_encoder_construct


class ClustMixNodeEncoder(torch.nn.Module):
    """
    Produces node features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self, model_config, **kwargs):
        super(ClustMixNodeEncoder, self).__init__()

        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = node_encoder_construct(model_config, model_name='geo_encoder', **kwargs)
        # pprint(model_config['cnn_encoder'])
        self.cnn_encoder = node_encoder_construct(model_config, model_name='cnn_encoder', **kwargs)

        if self.geo_encoder.more_feats:
            node_feats = 19
        else:
            node_feats = 16

        self.num_features = node_feats + self.cnn_encoder.encoder.latent_size
        self.linear = torch.nn.Linear(self.num_features, self.num_features)
        self.elu = torch.nn.functional.elu


    def forward(self, data, clusts):
        features_geo = self.geo_encoder(data, clusts)
        features_cnn = self.cnn_encoder(data, clusts)
        features_mix = torch.cat([features_geo, features_cnn], dim=1)
        out = self.elu(features_mix)
        out = self.linear(out)
        print("mixed node = ", out.shape)
        return out


class ClustMixEdgeEncoder(torch.nn.Module):
    """
    Produces edge features using both geometric and cnn encoder based feature extraction
    """
    def __init__(self, model_config, **kwargs):
        super(ClustMixEdgeEncoder, self).__init__()
        # print(model_config)
        self.normalize = model_config.get('normalize', True)
        # require sub-config key
        if 'geo_encoder' not in model_config:
            raise ValueError("Require geo_encoder config!")
        if 'cnn_encoder' not in model_config:
            raise ValueError("Require cnn_encoder config!")

        self.geo_encoder = edge_encoder_construct(model_config, model_name='geo_encoder', **kwargs)
        self.cnn_encoder = edge_encoder_construct(model_config, model_name='cnn_encoder', **kwargs)

        node_feats = 19
        self.num_features = node_feats + self.cnn_encoder.encoder.latent_size
        self.linear = torch.nn.Linear(self.num_features, self.num_features)
        self.elu = torch.nn.functional.elu

    def forward(self, data, clusts, edge_index):
        features_geo = self.geo_encoder(data, clusts, edge_index)
        features_cnn = self.cnn_encoder(data, clusts, edge_index)
        features_mix = torch.cat([features_geo, features_cnn], dim=1)
        out = self.elu(features_mix)
        out = self.linear(out)
        print("mixed edge = ", out.shape)
        return out
