import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

# From Pytorch Geometric Examples for PointNet:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet(torch.nn.Module):
    '''
    Pytorch Geometric's implementation of PointNet, modified for
    use in lartpc_mlreco3d and generalized. 
    '''
    def __init__(self, cfg, name='pointnet'):
        super(PointNet, self).__init__()

        self.model_config = cfg[name]

        self.depth = self.model_config.get('depth', 2)

        self.sampling_ratio = self.model_config.get('sampling_ratio', 0.5)
        if isinstance(self.sampling_ratio, float):
            self.sampling_ratio = [self.sampling_ratio] * self.depth
        elif isinstance(self.sampling_ratio, list):
            assert len(self.sampling_ratio) == self.depth
        else:
            raise ValueError("Sampling ratio must either be given as \
                             float or list of floats.")
        
        self.neighbor_radius = self.model_config.get('neighbor_radius', 3.0)
        if isinstance(self.neighbor_radius, float):
            self.neighbor_radius = [self.neighbor_radius] * self.depth
        elif isinstance(self.neighbor_radius, list):
            assert len(self.neighbor_radius) == self.depth
        else:
            raise ValueError("Neighbor aggregation radius must either \
                             be given as float or list of floats.")
        
        self.mlp_specs = []
        self.sa_modules = nn.ModuleList()

        for i in range(self.depth):
            mlp_specs = self.model_config['mlp_specs_{}'.format(i)]
            self.sa_modules.append(
                SAModule(self.sampling_ratio[i], self.neighbor_radius[i], MLP(mlp_specs))
            )
            self.mlp_specs.append(mlp_specs)

        self.mlp_specs_glob = self.model_config.get('mlp_specs_glob', [256 + 3, 256, 512, 1024])
        self.mlp_specs_final = self.model_config.get('mlp_specs_final', [1024, 512, 256, 128])
        self.dropout = self.model_config.get('dropout', 0.5)
        self.latent_size = self.mlp_specs_final[-1]

        self.sa3_module = GlobalSAModule(MLP(self.mlp_specs_glob))
        self.mlp = MLP(self.mlp_specs_final, dropout=self.dropout, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)

        out = sa0_out

        for m in self.sa_modules:
            out = m(*out)

        sa3_out = self.sa3_module(*out)
        x, pos, batch = sa3_out

        return self.mlp(x)
    

class PointNetEncoder(torch.nn.Module):

    def __init__(self, cfg, name='pointnet_encoder'):
        super(PointNetEncoder, self).__init__()
        self.net = PointNet(cfg)
        self.latent_size = self.net.latent_size

    def forward(self, batch):
        out = self.net(batch)
        return out