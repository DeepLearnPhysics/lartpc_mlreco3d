from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import node_model_construct, edge_model_construct, node_encoder_construct, edge_encoder_construct
from mlreco.models.layers.dbscan import DBScan, DBScanClusts2
from mlreco.models.uresnet_ppn_chain import ChainLoss as UResNetPPNLoss
from mlreco.models.uresnet_ppn_chain import Chain as UResNetPPN
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss
from mlreco.utils.ppn import uresnet_ppn_point_selector
from mlreco.utils.gnn.evaluation import edge_assignment
from mlreco.utils.gnn.network import complete_graph, bipartite_graph, inter_cluster_distance
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
        "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    """
    MODULES = ['dbscan', 'uresnet_ppn', 'attention_gnn']

    def __init__(self, model_config):
        super(ChainDBSCANGNN, self).__init__()

        # Initialize the chain parameters
        chain_config = model_config['chain']
        self.shower_class = int(chain_config['shower_class'])
        self.node_min_size = chain_config['node_min_size']
        self.edge_max_dist = chain_config['edge_max_dist']

        # Initialize the modules
        self.dbscan = DBScanClusts2(model_config)
        self.uresnet_ppn = UResNetPPN(model_config['uresnet_ppn'])
        self.uresnet_lonely = self.uresnet_ppn.uresnet_lonely
        self.ppn = self.uresnet_ppn.ppn
        self.node_encoder = node_encoder_construct(model_config)
        self.edge_encoder = edge_encoder_construct(model_config)
        self.node_predictor = node_model_construct(model_config)
        self.edge_predictor = edge_model_construct(model_config)

    def forward(self, data):

        # Pass the input data through UResNet+PPN (semantic segmentation + point prediction)
        result = self.uresnet_ppn(data)
        #return result
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
        if not len(frags[self.shower_class]):
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
        batch_ids = np.array(batch_ids)

        # Compute the cluster distance matrix, if necessary
        dist_mat = None
        if self.edge_max_dist > 0:
            dist_mat = inter_cluster_distance(data[0][:,:3], clusts)

        # Get the node features
        x = self.node_encoder(data[0], clusts)

        # Initialize a complete graph for node prediction, get edge features
        edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        if not edge_index.shape[1]:
            return result
        e = self.edge_encoder(data[0], clusts, edge_index)

        # Pass through the node model, get node predictions
        index = torch.tensor(edge_index, device=data[0].device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=data[0].device, dtype=torch.long)
        out = self.node_predictor(x, index, e, xbatch)
        node_pred = out['node_pred'][0]

        # Split the node prediction output, append result
        _, counts = torch.unique(data[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]

        split_clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]
        node_pred = [node_pred[b] for b in bcids]

        result.update(dict(
            node_pred = [node_pred],
            shower_fragments = [split_clusts]
        ))

        # Convert the node output to a list of primaries
        primary_ids = torch.argmax(out['node_pred'][0], dim=1)
        primaries = torch.nonzero(primary_ids).flatten()

        # Initialize the network for edge prediction, get edge features
        edge_index = bipartite_graph(batch_ids, primaries, dist_mat, self.edge_max_dist)
        if edge_index.shape[1] < 2: # Batch norm 1D does not handle batch_size < 2
            return result
        e = self.edge_encoder(data[0], clusts, edge_index)

        # Pass through the node model, get edge predictions
        index = torch.tensor(edge_index, device=data[0].device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=data[0].device, dtype=torch.long)
        out = self.edge_predictor(x, index, e, xbatch)
        edge_pred = out['edge_pred'][0]

        # Split the edge prediction output, append result
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]

        result.update(dict(
            edge_pred = [edge_pred],
            edge_index = [edge_index]
        ))

        return result


class ChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.sem_loss = UResNetPPNLoss(cfg['uresnet_ppn'])
        self.node_loss = NodeChannelLoss(cfg)
        self.edge_loss = EdgeChannelLoss(cfg)

    def forward(self, result, sem_label, particles, clust_label):

        loss = {}
        uresnet_ppn_loss = self.sem_loss(result, sem_label, particles)
        if 'shower_fragments' in result:
            result['clusts'] = result['shower_fragments']
        node_loss = self.node_loss(result, clust_label)
        edge_loss = self.edge_loss(result, clust_label, None)
        if 'clusts' in result:
            del result['clusts']
        loss.update(uresnet_ppn_loss)
        loss.update(node_loss)
        loss.update(edge_loss)
        loss['loss'] = uresnet_ppn_loss['loss'] + node_loss['loss'] + edge_loss['loss']
        loss['accuracy'] = (uresnet_ppn_loss['accuracy'] + node_loss['accuracy'] + edge_loss['accuracy'])/3
        return loss
