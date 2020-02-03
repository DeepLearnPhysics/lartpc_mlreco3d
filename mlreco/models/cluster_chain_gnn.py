from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import node_model_construct, edge_model_construct
from mlreco.models.layers.dbscan import DBScan, DBScanClusts2
from mlreco.models.uresnet_ppn_chain import ChainLoss as UResNetPPNLoss
from mlreco.models.uresnet_ppn_chain import Chain as UResNetPPN
from mlreco.utils.ppn import uresnet_ppn_point_selector
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features
import mlreco.utils
# chain UResNet + PPN + DBSCAN + GNN for showers

class ChainDBSCANGNN(torch.nn.Module):
    """
    Chain of Networks
    1) UResNet - for voxel labels
    2) PPN - for particle locations
    3) DBSCAN - to form cluster
    4) GNN - to assign EM shower groups

    INPUT DATA:
        just energy deposision data
        "input_data": ["parse_sparse3d_scn", "sparse3d_data"],
    """
    MODULES = ['dbscan', 'uresnet_ppn', 'attention_gnn']

    def __init__(self, model_config):
        super(ChainDBSCANGNN, self).__init__()

        # Initialize the chain parameters
        chain_config = model_config['modules']['chain']
        self.shower_class = int(chain_config['shower_class'])
        self.node_min_size = chain_config['node_min_size']
        self.node_encoder = chain_config['node_encoder']
        self.network = chain_config['network']
        self.edge_max_dist = chain_config['edge_max_dist']

        # Initialize the modules
        self.dbscan = DBScanClusts2(model_config)
        self.uresnet_ppn = UResNetPPN(model_config)
        node_model = node_model_construct(model_config['modules']['node_model']['name'])
        self.node_predictor = node_model(model_config['modules']['node_model'])
        edge_model = edge_model_construct(model_config['modules']['edge_model']['name'])
        self.edge_predictor = edge_model(model_config['modules']['edge_model'])

    def forward(self, data):
        
        # Pass the input data through UResNet+PPN (semantic segmentation + point prediction)
        result = self.uresnet_ppn(data)

        # Run DBSCAN
        semantic = torch.argmax(result['segmentation'][0],1).view(-1,1)
        dbscan_input = torch.cat([data[0].to(torch.float32),semantic.to(torch.float32)],dim=1)
        frags = self.dbscan(dbscan_input, onehot=False)

        # Create cluster id, group id, and shape tensor
        cluster_info = torch.ones([data[0].size()[0], 3], dtype=data[0].dtype, device=data[0].device)
        cluster_info *= -1.
        for shape, shape_frags in enumerate(frags):
            for frag_id, frag in enumerate(shape_frags):
                cluster_info[frag,0] = frag_id
                cluster_info[frag,2] = shape

        # Save the list of EM clusters, return if empty
        if len(frags[self.shower_class]):
            result.update(dict(shower_fragments=[frags[self.shower_class]]))
        else:
            return result

        # Prepare cluster ID, batch ID for shower clusters
        clusts = frags[self.shower_class]
        clust_ids = np.arange(len(clusts))
        batch_ids = []
        for clust in clusts:
            batch_id = data[0][clust,3].unique()
            if not len(batch_id) == 1:
                raise ValueError('Found a cluster with mixed batch ids:',batch_id)
            batch_ids.append(batch_id[0].item())

        # Compute the cluster distance matrix, if necessary
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst':
            dist_mat = np.zeros(shape=(len(clusts),len(clusts)),dtype=np.float32)
            for idx0 in range(len(clusts)):
                pts0 = data[0][clusts[idx0]][:,:3]
                for idx1 in range(len(clusts)):
                    if idx0 < idx1:
                        pts1 = data[0][clusts[idx1]][:,:3]
                        dist_mat[idx0,idx1]=mlreco.utils.cdist(pts0,pts1)
                    else:
                        dist_mat[idx0,idx1]=dist_mat[idx1,idx0]

        # Get the node features
        x = torch.tensor(cluster_vtx_features(data[0].numpy(), clusts), device=data[0].device, dtype=torch.float)

        # If the bipartite graph is used, predict primaries
        if self.network == 'bipartite':
            # Initialize a complete graph for node prediction, get edge features
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
            if edge_index.shape[1] < 2: # Batch norm 1D does not handle batch_size < 2
                return result
            e = torch.tensor(cluster_edge_features(data[0], clusts, edge_index), device=data[0].device, dtype=torch.float)

            # Pass through the node model, get node prections
            index = torch.tensor(edge_index, device=data[0].device, dtype=torch.long)
            xbatch = torch.tensor(batch_ids, device=data[0].device, dtype=torch.long)
            out = self.node_predictor(x, index, e, xbatch)

            # Convert the node output to a list of primaries
            primaries = torch.nonzero(torch.argmax(out['node_pred'][0], dim=1)).flatten()
            result.update(shower_primaries=primaries)

        # Initialize the requested network for edge prediction, get edge features
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(cluster_label, clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'bipartite':
            group_ids = get_cluster_group(cluster_label, clusts)
            edge_index = bipartite_graph(batch_ids, primaries, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)
        
        result.update(edge_index=edge_index)
        e = torch.tensor(cluster_edge_features(data[0], clusts, edge_index), device=data[0].device, dtype=torch.float)

        # Pass through the node model, get edge prections
        index = torch.tensor(edge_index, device=data[0].device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=data[0].device, dtype=torch.long)
        out = self.edge_predictor(x, index, e, xbatch)
        result.update(out)

        return result


class ChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.loss = UResNetPPNLoss(cfg)

    def forward(self, result, label, particles):
        return self.loss(result, label, particles)

