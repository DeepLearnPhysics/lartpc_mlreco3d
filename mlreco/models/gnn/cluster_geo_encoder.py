# Geometric feature extractor for Cluster GNN
import torch
from mlreco.utils import local_cdist

class ClustGeoNodeEncoder(torch.nn.Module):
    """
    Produces geometric cluster node features.

    """
    def __init__(self, model_config):
        super(ClustGeoNodeEncoder, self).__init__()

        # Initialize the chain parameters
        self.use_numpy = model_config.get('use_numpy', False)

    def forward(self, data, clusts, delta=0.):

        # Get the voxel set
        voxels = data[:,:3].float()
        dtype = voxels.dtype
        device = voxels.device

        # If numpy is to be used, bring data to cpu, pass through function
        if self.use_numpy:
            from mlreco.utils.gnn.data import cluster_vtx_features
            return torch.tensor(cluster_vtx_features(voxels.detach().cpu().numpy(), clusts), dtype=voxels.dtype, device=voxels.device)

        # Here is a torch-based implementation of cluster_vtx_features
        feats = []
        for c in clusts:
            # Get list of voxels in the cluster
            x = voxels[c]
            size = torch.tensor([len(c)], dtype=dtype).to(device)

            # Handle size 1 clusters seperately
            if len(c) < 2:
                # Don't waste time with computations, default to regularized
                # orientation matrix, zero direction
                center = x.flatten()
                B = delta * torch.eye(3, dtype=dtype).to(device)
                v0 = torch.zeros(3, dtype=dtype).to(device)
                feats.append(torch.cat((center, B.flatten(), v0, size)))
                continue

            # Center data
            center = x.mean(dim=0)
            x = x - center

            # Get orientation matrix
            A = x.t().mm(x)

            # Get eigenvectors
            w, v = torch.eig(A, eigenvectors=True)
            w = w[:,0].flatten() # Real part of eigenvalues
            idxs = torch.argsort(w) # Sort in increasing order of eigenval
            w = w[idxs]
            v = v[:,idxs]
            dirwt = 0.0 if w[2] == 0 else 1.0 - w[1] / w[2]
            w = w + delta
            w = w / w[2]

            # Orientation matrix with regularization
            B = (1.-delta) * v.mm(torch.diag(w)).mm(v.t()) + delta * torch.eye(3, dtype=dtype).to(device)

            # Get direction - look at direction of spread orthogonal to v[:,maxind]
            v0 = v[:,2]

            # Projection of x along v0
            x0 = x.mv(v0)

            # Projection orthogonal to v0
            xp0 = x - torch.ger(x0, v0)
            np0 = torch.norm(xp0, dim=1)

            # Spread coefficient
            sc = torch.dot(x0, np0)
            if sc < 0:
                # Reverse
                v0 = -v0

            # Weight direction
            v0 = dirwt * v0

            # Append (center, B.flatten(), v0, size)
            feats.append(torch.cat((center, B.flatten(), v0, size)))

        return torch.stack(feats, dim=0)


class ClustGeoEdgeEncoder(torch.nn.Module):
    """
    Produces geometric cluster edge features.

    """
    def __init__(self, model_config):
        super(ClustGeoEdgeEncoder, self).__init__()

        # Initialize the chain parameters
        self.use_numpy = model_config.get('use_numpy', False)

    def forward(self, data, clusts, edge_index):

        # Get the voxel set
        voxels = data[:,:3].float()
        dtype = voxels.dtype
        device = voxels.device

        # If numpy is to be used, bring data to cpu, pass through function
        if self.use_numpy:
            from mlreco.utils.gnn.data import cluster_edge_features
            return torch.tensor(cluster_edge_features(voxels.detach().cpu().numpy(), clusts, edge_index), dtype=voxels.dtype, device=voxels.device)

        # Here is a torch-based implementation of cluster_edge_features
        feats = []
        for e in edge_index.T:

            # Get the voxels in the clusters connected by the edge
            x1 = voxels[clusts[e[0]]]
            x2 = voxels[clusts[e[1]]]

            # Find the closest set point in each cluster
            d12 = local_cdist(x1,x2)
            imin = torch.argmin(d12)
            i1, i2 = imin//len(x2), imin%len(x2)
            v1 = x1[i1,:] # closest point in c1
            v2 = x2[i2,:] # closest point in c2

            # Displacement
            disp = v1 - v2

            # Distance
            lend = torch.norm(disp) # length of displacement
            if lend > 0:
                disp = disp / lend
            B = torch.ger(disp, disp).flatten()
            feats.append(torch.cat([v1, v2, disp, torch.tensor([lend], dtype=dtype).to(device), B]))

        return torch.stack(feats, dim=0)

