import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

'''Adapted from https://github.com/JonasSchult/Mask3D with modification.'''

def get_normalized_coordinates(coords, spatial_size):
    assert len(coords.shape) == 2
    normalized_coords = (coords[:, :3].float() - spatial_size / 2) \
                        / (spatial_size / 2)
    return normalized_coords

class FourierEmbeddings(nn.Module):

    def __init__(self, cfg, name='fourier_embeddings'):
        super(FourierEmbeddings, self).__init__()
        self.model_config = cfg[name]
        
        self.D            = self.model_config.get('D', 3)
        self.num_input    = self.model_config.get('num_input_features', 3)
        self.pos_dim      = self.model_config.get('positional_encoding_dim', 32)
        self.normalize    = self.model_config.get('normalize_coordinates', False)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.spatial_size = self.model_config.get('spatial_size', [2753, 1056, 5966])
        self.spatial_size = torch.Tensor(self.spatial_size).float().to(device)
        
        assert self.pos_dim % 2 == 0
        self.gauss_scale = self.model_config.get('gauss_scale', 1.0)
        B = torch.empty((self.num_input, self.pos_dim // 2)).normal_()
        B *= self.gauss_scale
        self.register_buffer("gauss_B", B)

    def normalize_coordinates(self, coords):
        if len(coords.shape) == 2:
            return get_normalized_coordinates(coords, spatial_size=self.spatial_size)
        elif len(coords.shape) == 3:
            normalized_coords = (coords[:, :, :self.D].float() \
                                 - self.spatial_size / 2) \
                                 / (self.spatial_size / 2)
            return normalized_coords
        else:
            raise ValueError("Normalize coordinates saw {}D tensor!".format(len(coords.shape)))

    def forward(self, coords: torch.Tensor, features: torch.Tensor = None):
        if self.normalize:
            coordinates = self.normalize_coordinates(coords)
        else:
            coordinates = coords

        coordinates *= 2 * np.pi
        freqs = coordinates @ self.gauss_B
        if features is not None:
            embeddings = torch.cat([freqs.cos(), freqs.sin(), features], dim=-1)
        else:
            embeddings = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        return embeddings
        
