# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters, get_cluster_voxels
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features
from mlreco.utils.gnn.evaluation import edge_assignment
from . import flashmatching_gnn

def get_traj_features(data, clusts, delta=0.0):
    """
    get features for N clusters:
    * center (N, 3) array
    * orientation (N, 9) array
    * direction (N, 3) array
    output is (N, 15) matrix

    Optional arguments:
    delta = orientation matrix regularization

    """
    # first make sure data is numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    # types of data
    feats = []
    for c in clusts:
        # get center of cluster
        x = get_cluster_voxels(data, c)
        #Change x coordinate into relative x coordinate
        x_coord_mean = np.mean(x[:,0])
        x[:,0] = x[:,0] - x_coord_mean
        if len(c) < 2:
            # don't waste time with computations
            # default to regularized orientation matrix, zero direction
            center = x.flatten()
            B = delta * np.eye(3)
            v0 = np.zeros(3)
            feats.append(np.concatenate((center, B.flatten(), v0)))
            continue

        center = np.mean(x, axis=0)
        # center data
        x = x - center
        # get orientation matrix
        A = x.T.dot(x)
        # get eigenvectors - convention with eigh is that eigenvalues are ascending
        w, v = np.linalg.eigh(A)
        dirwt = 0.0 if w[2] == 0 else 1.0 - w[1] / w[2] # weight for direction
        w = w + delta # regularization
        w = w / w[2] # normalize top eigenvalue to be 1
        # orientation matrix
        B = v.dot(np.diag(w)).dot(v.T)

        # get direction - look at direction of spread orthogonal to v[:,2]
        v0 = v[:,2]
        # projection of x along v0 
        x0 = x.dot(v0)
        # projection orthogonal to v0
        xp0 = x - np.outer(x0, v0)
        np0 = np.linalg.norm(xp0, axis=1)
        # spread coefficient
        sc = np.dot(x0, np0)
        if sc < 0:
            # reverse 
            v0 = -v0
        # weight direction
        v0 = dirwt*v0
        # append, center, B.flatten(), v0
        feats.append(np.concatenate((center, B.flatten(), v0, [len(x)])))
    return np.array(feats)
    
    
class FlashMatchingModel(torch.nn.Module):
    """
    Driver class for edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.

    """
    def __init__(self, cfg):
        super(FullEdgeModel, self).__init__()

        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg

    def forward(self, data):
        tpc_data = torch.tensor(data['tpc_sample'], dtype=torch.float, requires_grad=False).to(device)
        pmt_data = torch.tensor(data['pmt_sample'], dtype=torch.float, requires_grad=False).to(device)       

        # Find index of points that have the same event_id
        tpc_events = form_clusters(tpc_data)
        # Get the event ids of each processed event
        tpc_event_ids = get_cluster_label(tpc_data, tpc_events)
        # Get the batch ids of each event
        #Every event is associate with the main batch id in it.
        tpc_batch_ids = get_cluster_batch(tpc_data, tpc_events)

        # Find index of points that have the same flash_id
        pmt_flashes = form_clusters_new(pmt_data)
        # Get the flash ids of each processed flash
        pmt_flash_ids = get_cluster_label(pmt_data, pmt_flashes)
        # Get the batch ids of each event
        pmt_batch_ids = get_cluster_batch(pmt_data, pmt_flashes)

        #Merge the obtains batch ids.
        batch_ids = np.append(tpc_batch_ids,pmt_batch_ids)

        #Create the graph. Here we want primary_ids to be the PMT flashes
        #Here PMT nodes and TPC nodes does not have the same dimension but the features linked with an edge does.

        n = len(tpc_batch_ids)
        m = len(pmt_flash_ids)

        # Obtain vertex features. (PCA, 15 features for now)
        # NEED TO ADD THE CHARGE
        x_TPC_data = get_traj_features(tpc_data, tpc_events)
        x_TPC_PCA = torch.tensor(x_TPC_data, dtype=torch.float, requires_grad=False)

        #Node data for the PMTs. We want position + pecount
        pmt_data_array = []
        for f in pmt_flashes:
            #aux1 = np.array(pmt_data[f,:3].cpu())
            aux2 = np.array([[np.array(pmt_data[m,5].cpu())] for m in f])
            #pmt_data_array.append(np.concatenate((aux1,aux2),axis=1))
            pmt_data_array.append(aux2)


        #Node data for the TPCs. We want position + batch index + pecount
        tpc_data_array = []
        #pos stands for the batch size and will allow the identification of the edge.
        pos = 0
        for f in tpc_events:
            aux1 = np.array(tpc_data[f,:3].cpu())
            aux2 = np.array([[pos] for m in f])
            aux3 = np.array([[np.array(tpc_data[m,5].cpu())] for m in f])
            #x_batch
            aux4 = np.array([[np.array(tpc_data[m,3].cpu())] for m in f])
            tpc_data_array.append(np.concatenate((aux1,aux2,aux3,aux4),axis=1))
            pos+=1

        edge_index = torch.tensor([[i, n + j] for i in range(n) for j in range(m) if batch_ids[i] == batch_ids[n + j]], dtype=torch.long, requires_grad=False).t().contiguous().reshape(2,-1)

        #We construct edge features
        tpc_voxel_data = []
        tpc_pca_data = []
        pmt_voxel_data = []
        node_data = []
        matching = []
        x_batch = []

        for node in range(len(edge_index[0])):
            tpc_pca_node = np.array(x_TPC_PCA[edge_index[0][node]]).astype(np.double)
            tpc_node = np.array(tpc_data_array[edge_index[0][node]]).astype(np.double)
            pmt_node = np.array(pmt_data_array[edge_index[1][node]-n]).astype(np.double)        

            x_batch.append(tpc_node[0][-1])

            for d in tpc_node:
                d[3]=node
                tpc_voxel_data.append(d[:-1])

            tpc_pca_data.append(tpc_pca_node)

            pmt_voxel_data.append(pmt_node)

            node_data.append([1,1])

            matching.append(int(edge_index[1][node]-n == edge_index[0][node]))

        x_batch =  torch.tensor(x_batch, dtype=torch.long, requires_grad=False).reshape(-1)
        
        tpc_voxel_tensor = torch.tensor(np.array(tpc_voxel_data), dtype=torch.float, requires_grad=False).to(device)
        tpc_pca_tensor = torch.tensor(np.array(tpc_pca_data), dtype=torch.float, requires_grad=False).to(device)
        pmt_voxel_tensor = torch.tensor(np.array(pmt_voxel_data), dtype=torch.float, requires_grad=False).to(device)
        node_tensor = torch.tensor(np.array(node_data), dtype=torch.float, requires_grad=False).to(device)
        matching_tensor = torch.tensor(np.array(matching), dtype=torch.long, requires_grad=False).to(device)
        
        x = [tpc_voxel_tensor, tpc_pca_tensor, pmt_voxel_tensor, node_tensor, edge_index, x_batch]
        y = matching_tensor
        
        model = flashmatching_gnn.UResGNet()
        model = model.to(device)
        
        out = model(x)
        
        return {**out, 
               'true':[matching_tensor]}
    
