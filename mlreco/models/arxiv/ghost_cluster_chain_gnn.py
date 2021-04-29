from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import gnn_model_construct, node_encoder_construct, edge_encoder_construct
from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.layers.dbscan import DBScanClusts2
from mlreco.models.grappa import GNNLoss
from mlreco.utils.gnn.cluster import relabel_groups, cluster_direction
from mlreco.utils.gnn.evaluation import node_assignment, node_assignment_score
from mlreco.utils.gnn.network import complete_graph
import mlreco.utils
from mlreco.utils.deghosting import adapt_labels
# chain UResNet + PPN + DBSCAN + GNN for showers

class GhostChainDBSCANGNN(torch.nn.Module):
    """
    Chain of Networks
    1) UResNet - for voxel labels
    2) PPN - for particle locations
    3) DBSCAN - to form cluster
    4) GNN - to assign EM shower groups and identify EM primaries

    INPUT DATA:
        just energy deposision data
        "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    """
    MODULES = ['chain', 'dbscan', 'uresnet_lonely', 'attention_gnn', 'ppn', 'node_encoder', 'edge_encoder', 'edge_model', 'node_model']

    def __init__(self, model_config):
        super(GhostChainDBSCANGNN, self).__init__()

        # Initialize the chain parameters
        chain_config = model_config['chain']
        self.shower_class = int(chain_config.get('shower_class', 0))
        self.node_min_size = chain_config.get('node_min_size', -1)
        self.group_pred = chain_config.get('group_pred', 'threshold')
        self.start_dir_max_dist = chain_config.get('start_dir_max_dist', -1)
        self.start_dir_opt = chain_config.get('start_dir_opt', False)
        self.input_features = model_config['uresnet_lonely'].get('features', 1)

        # Initialize the modules
        self.uresnet_lonely = UResNet(model_config)
        self.ppn            = PPN(model_config)
        self.dbscan         = DBScanClusts2(model_config)
        self.node_encoder   = node_encoder_construct(model_config['grappa'])
        self.edge_encoder   = edge_encoder_construct(model_config['grappa'])
        self.full_predictor = gnn_model_construct(model_config['grappa'])

    def full_chain(self, data, result):
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

        # If there is cut on EM cluster size, abide
        clusts = frags[self.shower_class]
        if self.node_min_size > 0:
            clusts = [c for c in frags[self.shower_class] if len(c) > self.node_min_size]

        # Prepare cluster ID, batch ID for shower clusters
        clust_ids = np.arange(len(clusts))
        batch_ids = []
        for clust in clusts:
            batch_id = data[0][clust,3].unique()
            if not len(batch_id) == 1:
                raise ValueError('Found a cluster with mixed batch ids:', batch_id)
            batch_ids.append(batch_id[0].item())
        batch_ids = np.array(batch_ids)

        # Initialize a complete graph for edge prediction, get node and edge features
        edge_index = complete_graph(batch_ids)
        if not edge_index.shape[1]:
            return result
        x = self.node_encoder(data[0], clusts)
        e = self.edge_encoder(data[0], clusts, edge_index)

        # Add best PPN point prediction + cluster direction estimate to each fragment + scores
        ppn_feats = torch.empty((0,8), device=data[0].device, dtype=torch.float)
        for clust in clusts:
            scores = torch.softmax(result['points'][0][clust][:,3:5], dim=1)
            argmax = torch.argmax(scores[:,-1])
            start  = data[0][clust][argmax,:3].float()+result['points'][0][clust][argmax,:3]+0.5
            dir = cluster_direction(data[0][clust][:,:3].float(), start, self.start_dir_max_dist, self.start_dir_opt)
            ppn_feats = torch.cat((ppn_feats, torch.cat([start, dir, scores[argmax]]).reshape(1,-1)), dim=0)

        x = torch.cat([x, ppn_feats], dim=1)

        # Pass through the edge model, get edge predictions
        index = torch.tensor(edge_index, device=data[0].device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=data[0].device, dtype=torch.long)
        out = self.full_predictor(x, index, e, xbatch)
        node_pred = out['node_pred'][0]
        edge_pred = out['edge_pred'][0]

        # Divide the edge prediction output out into different arrays (one per batch)
        _, counts = torch.unique(data[0][:,3], return_counts=True)
        vids  = np.concatenate([np.arange(n.item()) for n in counts])
        cids  = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        node_pred = [node_pred[b] for b in bcids]
        edge_pred    = [edge_pred[b] for b in beids]
        edge_index   = [cids[edge_index[:,b]].T for b in beids]
        split_clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        # Get the predicted group ids of each of the clusters (no overlap between batches)
        split_group_ids = []
        if self.group_pred == 'threshold':
            for b in range(len(counts)):
                split_group_ids.append(node_assignment(edge_index[b], np.argmax(edge_pred[b].detach().cpu().numpy(), axis=1), len(split_clusts[b])))
        elif self.group_pred == 'score':
            for b in range(len(counts)):
                if len(split_clusts[b]):
                    split_group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(split_clusts[b])))
                else:
                    split_group_ids.append(np.array([], dtype = np.int64))

        result.update(dict(
            shower_fragments = [split_clusts],
            edge_index = [edge_index],
            node_pred = [node_pred],
            edge_pred = [edge_pred],
            group_pred = [split_group_ids]
        ))
        return result

    def forward(self, data):

        # Pass the input data through UResNet+PPN (semantic segmentation + point prediction)
        result = self.uresnet_lonely([data[0]])
        ppn_input = {}
        ppn_input.update(result)
        ppn_input['ppn_feature_enc'] = ppn_input['ppn_feature_enc'][0]
        ppn_input['ppn_feature_dec'] = ppn_input['ppn_feature_dec'][0]
        if 'ghost' in ppn_input:
            ppn_input['ghost'] = ppn_input['ghost'][0]
        ppn_output = self.ppn(ppn_input)
        result.update(ppn_output)

        # Update input based on deghosting results
        deghost = result['ghost'][0].argmax(dim=1) == 0
        data[0] = data[0][deghost]
        if self.input_features > 1:
            data[0] = data[0][:, :-self.input_features+1]

        segmentation, points = result['segmentation'][0].clone(), result['points'][0].clone()

        deghost_result = {}
        deghost_result.update(result)
        deghost_result['segmentation'][0] = result['segmentation'][0][deghost]
        deghost_result['points'][0] = result['points'][0][deghost]
        # Run the rest of the full chain

        full_chain_result = self.full_chain(data, deghost_result)

        result.update(full_chain_result)
        result['segmentation'][0] = segmentation
        result['points'][0] = points

        return result


class GhostChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(GhostChainLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.ppn_loss     = PPNLoss(cfg)
        self.gnn_loss    = GNNLoss(cfg['grappa_loss'])

    def full_chain_loss(self, result, clust_label):
        if 'shower_fragments' in result:
            result['clusts'] = result['shower_fragments']
        gnn_loss = self.gnn_loss(result, clust_label)
        return gnn_loss

    def forward(self, result, sem_label, particles, clust_label):
        loss = {}
        uresnet_loss = self.uresnet_loss(result, sem_label)
        ppn_loss = self.ppn_loss(result, sem_label, particles)

        # Adapt to ghost points
        clust_label = adapt_labels(result, sem_label, clust_label)

        gnn_loss = self.full_chain_loss(result, clust_label)

        loss.update(uresnet_loss)
        loss.update(ppn_loss)
        loss.update(gnn_loss)
        loss['seg_loss'] = uresnet_loss['loss']
        loss['seg_accuracy'] = uresnet_loss['accuracy']
        loss['ppn_accuracy'] = ppn_loss['ppn_acc']
        loss['loss'] = uresnet_loss['loss'] + ppn_loss['ppn_loss'] + gnn_loss['loss']
        loss['accuracy'] = (uresnet_loss['accuracy'] + ppn_loss['ppn_acc'] + gnn_loss['accuracy'])/3
        return loss
