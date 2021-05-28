import torch
import torch.nn as nn

# class EdgeKernel(nn.module):

#     def __init__(self, in1_features, in2_features, out_features=1, **kwargs):
#         super(EdgeKernel, self).__init__()
#         self.in1_features = in1_features
#         self.in2_features = in2_features
#         self.out_features = out_features

#     def forward(self, x):
#         raise NotImplementedError

class DefaultKernel(nn.Module):

    def __init__(self, num_features):
        super(DefaultKernel, self).__init__()

    def compute_edge_weight(self,
                            sp_emb1: torch.Tensor,
                            sp_emb2: torch.Tensor,
                            ft_emb1: torch.Tensor,
                            ft_emb2: torch.Tensor,
                            cov1: torch.Tensor,
                            cov2: torch.Tensor,
                            occ1, occ2, eps=0.001):

        device = sp_emb1.device

        sp_cov_i = cov1[:, 0]
        sp_cov_j = cov2[:, 0]
        sp_i = ((sp_emb1 - sp_emb2)**2).sum(dim=1) / (sp_cov_i**2 + eps)
        sp_j = ((sp_emb1 - sp_emb2)**2).sum(dim=1) / (sp_cov_j**2 + eps)

        ft_cov_i = cov1[:, 1]
        ft_cov_j = cov2[:, 1]
        ft_i = ((ft_emb1 - ft_emb2)**2).sum(dim=1) / (ft_cov_i**2 + eps)
        ft_j = ((ft_emb1 - ft_emb2)**2).sum(dim=1) / (ft_cov_j**2 + eps)

        p_ij = torch.exp(-sp_i-ft_i)
        p_ji = torch.exp(-sp_j-ft_j)

        pvec = torch.clamp(p_ij + p_ji - p_ij * p_ji, min=0, max=1)

        r1 = occ1
        r2 = occ2
        r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
        pvec = pvec / r
        return pvec

    def forward(self, x1, x2):

        w = self.compute_edge_weight(
            x1[:, :3],
            x2[:, :3],
            x1[:, 3:3+16],
            x2[:, 3:3+16],
            x1[:, 19:21],
            x2[:, 19:21],
            x1[:, -1],
            x2[:, -1])

        print(w)

        w = torch.clamp(w, min=0+1e-6, max=1-1e-6)

        out = torch.logit(w)

        return out


class MixedKernel(nn.Module):

    def __init__(self, num_features):
        super(MixedKernel, self).__init__()
        self.num_ft_features = num_features
        self.cos = nn.CosineSimilarity(dim=1)

    def compute_edge_weight(self,
                            sp_emb1: torch.Tensor,
                            sp_emb2: torch.Tensor,
                            ft_emb1: torch.Tensor,
                            ft_emb2: torch.Tensor,
                            cov1: torch.Tensor,
                            cov2: torch.Tensor,
                            occ1, occ2, eps=0.001):

        device = sp_emb1.device

        sp_cov_i = cov1[:, 0]
        sp_cov_j = cov2[:, 0]
        sp_i = ((sp_emb1 - sp_emb2)**2).sum(dim=1) / (sp_cov_i**2 + eps)
        sp_j = ((sp_emb1 - sp_emb2)**2).sum(dim=1) / (sp_cov_j**2 + eps)

        ft_cov_i = cov1[:, 1]
        ft_cov_j = cov2[:, 1]
        ft_i = ((ft_emb1 - ft_emb2)**2).sum(dim=1) / (ft_cov_i**2 + eps)
        ft_j = ((ft_emb1 - ft_emb2)**2).sum(dim=1) / (ft_cov_j**2 + eps)

        p_ij = torch.exp(-sp_i-ft_i)
        p_ji = torch.exp(-sp_j-ft_j)

        pvec = torch.clamp(p_ij + p_ji - p_ij * p_ji, min=0, max=1)

        r1 = occ1
        r2 = occ2
        r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
        pvec = pvec / r
        return pvec


    def compute_weight_coordinate_space(self,
                                        coords1,
                                        coords2,
                                        tangent1,
                                        tangent2,
                                        coords_cov1,
                                        coords_cov2,
                                        tangent_cov1,
                                        tangent_cov2):

        device = tangent1.device

        coords_cov = (coords_cov1 + coords_cov2) / 2.0
        chord = coords1 - coords2
        chord_dist = torch.pow(chord, 2)
        dist = torch.sum(chord_dist * coords_cov, dim=1)
        coords_weight = torch.exp(dist)

        # Affinity
        a1 = torch.abs(self.cos(chord, tangent1))
        a2 = torch.abs(self.cos(chord, tangent2))
        norm_factor = torch.sum(chord_dist, dim=1)
        tangent_cov = (tangent_cov1 + tangent_cov2) / 2.0
        a = a1 * a2 * tangent_cov / (norm_factor + 1e-5)
        affinity_weight = torch.exp(-a)

        pvec = coords_weight * affinity_weight
        return pvec


    def forward(self, x1, x2):

        w1 = self.compute_edge_weight(
            x1[:, 6:9],
            x2[:, 6:9],
            x1[:, 9:9+self.num_ft_features],
            x2[:, 9:9+self.num_ft_features],
            x1[:, 9+self.num_ft_features:9+self.num_ft_features+2],
            x2[:, 9+self.num_ft_features:9+self.num_ft_features+2],
            x1[:, -1],
            x2[:, -1])

        w2 = self.compute_weight_coordinate_space(
            x1[:, :3],
            x2[:, :3],
            x1[:, 3:3+3],
            x2[:, 3:3+3],
            x1[:, 9+self.num_ft_features+2:9+self.num_ft_features+5],
            x2[:, 9+self.num_ft_features+2:9+self.num_ft_features+5],
            x1[:, 9+self.num_ft_features+5],
            x2[:, 9+self.num_ft_features+5]
        )

        w = torch.clamp(w1 * w2, min=0+1e-6, max=1-1e-6)

        out = torch.logit(w)

        return out

class BilinearKernel(nn.Module):

    def __init__(self, num_features, bias=False):
        super(BilinearKernel, self).__init__()
        self.m = nn.Bilinear(num_features, num_features, 1, bias=bias)

    def forward(self, x1, x2):
        return self.m(x1, x2)



class BilinearNNKernel(nn.Module):

    def __init__(self, num_features, bias=False):
        super(BilinearNNKernel, self).__init__()
        
        self.m = nn.Linear(64, 1, bias=bias)

        self.nn1 = nn.Sequential(
            nn.Linear(num_features, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(num_features, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

    def forward(self, x1, x2):

        f1 = self.nn1(x1)
        f2 = self.nn2(x2)

        out = torch.cat([f1, f2], dim=1)
        
        return self.m(out)