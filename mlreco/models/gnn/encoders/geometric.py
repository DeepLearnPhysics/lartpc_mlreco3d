# Geometric feature extractor for Cluster GNN
import torch
import numpy as np
from torch_scatter import scatter_min
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_vtx_features_extended
from mlreco.utils.gnn.cluster import cluster_start_point

class ClustGeoNodeEncoder(torch.nn.Module):
    """
    Produces geometric cluster node features.

    """
    def __init__(self, model_config):
        super(ClustGeoNodeEncoder, self).__init__()

        # Initialize the chain parameters
        self.use_numpy = model_config.get('use_numpy', False)

        # flag for whether including the semantic type, mean energy per voxel, and std energy per voxel
        # If true, the output feature number will be 19
        self.more_feats = model_config.get('more_feats', False)

        # flag for whether to turn on the adjusting the direction
        # the direction from PCA is actually directionless
        # It can be the particle outgoing direction but can also be the inversed
        # If this flag is on, it will adjust the direction feature to be always the outgoing direction of particle
        self.adjust_node_direction = model_config.get('adjust_node_direction', False)


    def forward(self, data, clusts):

        # Get the voxel set
        voxels = data[:,:3].float()
        dtype = voxels.dtype
        device = voxels.device

        # Get the value & semantic types
        values = data[:,4].float()
        sem_types = data[:,-1].float()

        # If numpy is to be used, bring data to cpu, pass through function
        if self.use_numpy:
            if not self.more_feats:
                return torch.tensor(cluster_vtx_features(voxels.detach().cpu().numpy(), clusts, whether_adjust_direction = self.adjust_node_direction), dtype=voxels.dtype, device=voxels.device)
            feats = np.concatenate(
                (
                    cluster_vtx_features(voxels.detach().cpu().numpy(), clusts, whether_adjust_direction=self.adjust_node_direction),
                    cluster_vtx_features_extended(values.detach().cpu().numpy(), sem_types.detach().cpu().numpy(), clusts)
                ),
                axis=1
            )
            return torch.tensor(
                feats,
                dtype=voxels.dtype,
                device=voxels.device
            )

        # Here is a torch-based implementation of cluster_vtx_features
        feats = []
        for c in clusts:

            # Get list of voxels in the cluster
            x = voxels[c]
            size = torch.tensor([len(c)], dtype=dtype, device=device)

            # Do not waste time with computations with size 1 clusters, default to zeros
            if len(c) < 2:
                if not self.more_feats:
                    feats.append(torch.cat((x.flatten(), torch.zeros(9, dtype=dtype, device=device), v0, size)))
                else:
                    extra_feats = torch.tensor([values[c[0]], 0., sem_types[c[0]]], dtype=dtype, device=device)
                    feats.append(torch.cat((x.flatten(), torch.zeros(9, dtype=dtype, device=device), v0, size, extra_feats)))
                continue

            # Center data
            center = x.mean(dim=0)
            x = x - center

            # Get orientation matrix
            A = x.t().mm(x)

            # Get eigenvectors, normalize orientation matrix and eigenvalues to largest
            # This step assumes points are not superimposed, i.e. that largest eigenvalue != 0
            w, v = torch.symeig(A, eigenvectors=True)
            dirwt = 1.0 - w[1] / w[2]
            B = A / w[2]
            w = w / w[2]

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

            # Adjust direction to the start direction if requested
            if self.adjust_node_direction:
                if torch.dot(
                        x[cluster_start_point(x.detach().cpu().numpy())],
                        v0
                )>0:
                    v0 = -v0

            # Append (center, B.flatten(), v0, size)
            if not self.more_feats:
                feats.append(torch.cat((center, B.flatten(), v0, size)))
            else:
                extra_feats = torch.tensor([values[c].mean(), values[c].std(), sem_types[c].mode()[0]], dtype=dtype, device=device)
                feats.append(torch.cat((center, B.flatten(), v0, size, extra_feats)))

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

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1] / 2)
        undirected = not edge_index.shape[1] or (not edge_index.shape[1] % 2 and [edge_index[1, 0], edge_index[0, 0]] == edge_index[:, half_idx].tolist())
        if undirected: edge_index = edge_index[:, :half_idx]

        # Get the voxel set
        voxels = data[:,:3].float()
        dtype = voxels.dtype
        device = voxels.device

        # If numpy is to be used, bring data to cpu, pass through function
        # Otherwise use torch-based implementation of cluster_edge_features
        if self.use_numpy:
            from mlreco.utils.gnn.data import cluster_edge_features
            feats = torch.tensor(cluster_edge_features(voxels.detach().cpu().numpy(), clusts, edge_index), dtype=voxels.dtype, device=voxels.device)
        else:
            # Here is a torch-based implementation of cluster_edge_features
            feats = []
            for e in edge_index.T:

                # Get the voxels in the clusters connected by the edge
                x1 = voxels[clusts[e[0]]]
                x2 = voxels[clusts[e[1]]]

                # Find the closest set point in each cluster
                d12 = torch.cdist(x1,x2)
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


class ClustGeoEdgeEncoderVectorized(torch.nn.Module):
    """
    Produces geometric cluster edge features much faster if:
    - It is run on **GPU**
    - The graph is complete or close to complete

    This does not help if the number of edges is small or if ran on CPU.
    """
    def __init__(self, model_config):
        super(ClustGeoEdgeEncoderVectorized, self).__init__()

    def forward(self, data, clusts, edge_index):

        # Get the voxel set (torch geometric expects float features so might as well cast here)
        voxels = data[:,:3].float()

        # Loop over the different batches
        feats, edge_ids = [], []
        batch_ids = torch.cat([data[c[0],3].reshape(-1) for c in clusts])
        for b in batch_ids.unique():

            # Get the ids of the voxels that belong to the list of clusters and their cluster ids
            batch_mask = torch.nonzero(batch_ids == b, as_tuple=True)[0]
            voxel_ids  = torch.cat([torch.tensor(clusts[c], dtype=torch.long, device=voxels.device) for c in batch_mask])
            clust_ids  = torch.cat([i*torch.ones(len(clusts[c]), dtype=torch.long, device=voxels.device) for i, c in enumerate(batch_mask)])

            # Get the edge index of all the pair of voxels in the batch
            batch_voxels   = voxels[voxel_ids]
            batch_edge_ids = (len(batch_mask)*clust_ids.reshape(-1,1)+clust_ids).flatten()

            # Get the distance matrix
            dist_mat = torch.cdist(batch_voxels, batch_voxels).flatten()

            # Get the ids of the voxel pairs which minimize the distance
            lend, idxs = scatter_min(dist_mat, batch_edge_ids)

            # Get the points that correspond to the first voxels
            v1 = batch_voxels[idxs//len(batch_voxels)]

            # Get the points that correspond to the second voxels
            v2 = batch_voxels[idxs%len(batch_voxels)]

            # Get the displacement
            disp = v1 - v2

            # Reshape the distance vector to a column vector
            lend = lend.reshape(-1,1)

            # Normalize the displacement
            disp = disp/(lend + (lend == 0))

            # Compute the outer product of the displacement
            n = len(disp)
            B = torch.matmul(disp.reshape(n,-1,1), disp.reshape(n,1,-1)).reshape(n,-1)

            feats.append(torch.cat([v1, v2, disp, lend, B], dim=1))
            edge_ids.append((len(clusts)*batch_mask.reshape(-1,1)+batch_mask).flatten())

        # Narrow down the edge features to the request list
        edge_ids = torch.cat(edge_ids)
        edge_map = torch.empty(len(clusts)*len(clusts), dtype=torch.long, device=voxels.device)
        edge_map[edge_ids] = torch.arange(len(edge_ids))
        req_edge_ids = len(clusts)*edge_index[0]+edge_index[1]

        # Return
        return torch.cat(feats)[edge_map[req_edge_ids]]
