import torch
import numpy as np
from torch_scatter import scatter_min

from mlreco.utils import local_cdist
from mlreco.utils.globals import COORD_COLS, VALUE_COL, SHAPE_COL
from mlreco.utils.gnn.data import cluster_features, cluster_edge_features

class ClustGeoNodeEncoder(torch.nn.Module):
    """
    Produces geometric cluster node features.

    The first 19 features are composed of:
        - Center (3)
        - Covariance matrix (9)
        - Principal axis (3)
        - Voxel count (1)
        - Mean energy (1)
        - RMS energy (1)
        - Semantic type (1), i.e. most represented type in cluster

    6 features for the end points (duplicated for shower, 
        randomly ordered for tracks)
    3 features for direction estimate (mean direction w.r.t. start point)

    Total of 28 hand-engineered features

    """
    def __init__(self, model_config):
        super(ClustGeoNodeEncoder, self).__init__()

        # Initialize the encoder parameters
        self.use_numpy = model_config.get('use_numpy', True)
        self.add_value = model_config.get('add_value', False)
        self.add_shape = model_config.get('add_shape', False)

        # Deprecated
        if 'more_feats' in model_config:
            from warnings import warn
            warn('`more_feats` is deprecated, use `add_value` and/or `add_shape` instead', DeprecationWarning, 2)
            if model_config['more_feats']:
                self.add_value = self.add_shape = True

    def forward(self, data, clusts):

        # If numpy is to be used, bring data to CPU, pass through Numba function
        if self.use_numpy:
            return cluster_features(data, clusts, self.add_value, self.add_shape)

        # Get the voxel set
        voxels = data[:, COORD_COLS].float()

        # Get the value & semantic types
        values    = data[:, VALUE_COL].float()
        sem_types = data[:, SHAPE_COL].float()

        # Below is a torch-based implementation of cluster_features
        feats = []
        for c in clusts:

            # Get list of voxels in the cluster
            x = voxels[c]
            size = torch.tensor([len(c)], dtype=voxels.dtype, device=voxels.device)

            # Do not waste time with computations with size 1 clusters, default to zeros
            if len(c) < 2:
                feats_v = torch.cat((x.flatten(), torch.zeros(12, dtype=voxels.dtype, device=voxels.device), size))
                if add_value:
                    feats_v = torch.cat((feats_v, torch.tensor([values[c[0]], 0.], dtype=voxels.dtype, device=voxels.device)))
                if add_shape:
                    feats_v = torch.cat((feats_v, torch.tensor([sem_types[c[0]]], dtype=voxels.dtype, device=voxels.device)))

                feats.append(feats_v)
                continue

            # Center data
            center = x.mean(dim=0)
            x = x - center

            # Get orientation matrix
            A = x.t().mm(x)

            # Get eigenvectors, normalize orientation matrix and eigenvalues to largest
            # This step assumes points are not superimposed, i.e. that largest eigenvalue != 0
            #w, v = torch.symeig(A, eigenvectors=True)
            w, v = torch.linalg.eigh(A, UPLO='U')
            dirwt = 1.0 - w[1] / w[2]
            B = A / w[2]

            # Get the principal direction, identify the direction of the spread
            v0 = v[:,2]

            # Projection all points, x, along the principal axis
            x0 = x.mv(v0)

            # Evaluate the distance from the points to the principal axis
            xp0 = x - torch.ger(x0, v0)
            np0 = torch.norm(xp0, dim=1)

            # Flip the principal direction if it is not pointing towards the maximum spread
            sc = torch.dot(x0, np0)
            if sc < 0:
                v0 = -v0

            # Weight direction
            v0 = dirwt * v0

            # Append (center, B.flatten(), v0, size)
            feats_v = torch.cat((center, B.flatten(), v0, size))
            if add_value:
                feats_v = torch.cat((feats_v, torch.tensor([values[c].mean(), values[c].std()], dtype=voxels.dtype, device=voxels.device)))
            if add_shape:
                feats_v = torch.cat((feats_v, torch.tensor([sem_types[c].mode()], dtype=voxels.dtype, device=voxels.device)))

            feats.append(feats_v)

        return torch.stack(feats, dim=0)


class ClustGeoEdgeEncoder(torch.nn.Module):
    """
    Produces geometric cluster edge features.

    """
    def __init__(self, model_config):
        super(ClustGeoEdgeEncoder, self).__init__()

        # Initialize the chain parameters
        self.use_numpy = model_config.get('use_numpy', True)

    def forward(self, data, clusts, edge_index, closest_index=None):

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1] / 2)
        undirected = not edge_index.shape[1] or (not edge_index.shape[1] % 2 and [edge_index[1, 0], edge_index[0, 0]] == edge_index[:, half_idx].tolist())
        if undirected: edge_index = edge_index[:, :half_idx]

        # If numpy is to be used, bring data to cpu, pass through Numba function
        # Otherwise use torch-based implementation of cluster_edge_features
        if self.use_numpy:
            feats = cluster_edge_features(data, clusts, edge_index.T, closest_index=closest_index)
        else:
            # Get the voxel set
            voxels = data[:, COORD_COLS].float()

            # Here is a torch-based implementation of cluster_edge_features
            feats = []
            for e in edge_index.T:

                # Get the voxels in the clusters connected by the edge
                x1 = voxels[clusts[e[0]]]
                x2 = voxels[clusts[e[1]]]

                # Find the closest set point in each cluster
                d12 = local_cdist(x1, x2)
                imin = torch.argmin(d12)
                i1, i2 = imin//len(x2), imin%len(x2)
                v1 = x1[i1,:] # closest point in c1
                v2 = x2[i2,:] # closest point in c2

                # Displacement
                disp = v1 - v2

                # Distance
                lend = torch.norm(disp)
                if lend > 0:
                    disp = disp / lend

                # Outer product
                B = torch.ger(disp, disp).flatten()

                feats.append(torch.cat([v1, v2, disp, lend.reshape(1), B]))

            feats = torch.stack(feats, dim=0)

        # If the graph is undirected, infer reciprocal features
        if undirected:
            feats_flip = feats.clone()
            feats_flip[:,:3] = feats[:,3:6]
            feats_flip[:,3:6] = feats[:,:3]
            feats_flip[:,6:9] = -feats[:,6:9]
            feats = torch.cat([feats,feats_flip])

        return feats
