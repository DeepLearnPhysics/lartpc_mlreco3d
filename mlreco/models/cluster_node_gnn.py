# GNN that attempts to predict primary clusters
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import node_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.dbscan import DBScanClusts2
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label
from mlreco.utils.gnn.network import loop_graph, complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features

class ClustNodeGNN(torch.nn.Module):
    """
    Driver class for cluster node prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          network         : <type of network: 'empty', 'loop', 'complete', 'delaunay', 'mst' or 'bipartite' (default 'complete')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          edge_dist_numpy : <use numpy to compute inter cluster distance (default False)>
        dbscan:
          <dictionary of dbscan parameters>
        node_encoder:
          name: <name of the node encoder>
          <dictionary of arguments to pass to the encoder>
        edge_encoder:
          name: <name of the edge encoder>
          <dictionary of arguments to pass to the encoder>
        edge_model:
          name: <name of the edge model>
          <dictionary of arguments to pass to the model>
          model_path      : <path to the model weights>
    """

    MODULES = ['chain', 'dbscan', 'node_encoder', 'edge_encoder', 'edge_model', 'node_model']

    def __init__(self, cfg):
        super(ClustNodeGNN, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # Choose what type of node to use
        self.node_type = chain_config.get('node_type', 0)
        self.node_min_size = chain_config.get('node_min_size', -1)

        # Choose what type of network to use
        self.network = chain_config.get('network', 'complete')
        self.edge_max_dist = chain_config.get('edge_max_dist', -1)
        self.edge_dist_metric = chain_config.get('edge_dist_metric','set')
        self.edge_dist_numpy = chain_config.get('edge_dist_numpy',False)
        self.num_edge_feats =  cfg['node_model'].get('edge_feats')

        # If requested, use DBSCAN to form clusters from semantics
        self.do_dbscan = False
        if 'dbscan' in cfg:
            self.do_dbscan = True
            self.dbscan = DBScanClusts2(cfg)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)

        # Construct the model
        self.node_predictor = node_model_construct(cfg)

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            data ([torch.tensor]): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            dict:
                'node_pred' (torch.tensor): (E,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'batch_ids' (np.ndarray)  : (C) Cluster batch ids
                'edge_index' (np.ndarray)  : (2,0) Empty incidence matrix
        """
        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        data = data[0]
        device = data.device
        if self.do_dbscan:
            clusts = self.dbscan(data, onehot=False)
            if self.node_type > -1:
                clusts = clusts[self.node_type]
            else:
                clusts = np.concatenate(clusts).tolist()
        else:
            if self.node_type > -1:
                mask = torch.nonzero(data[:,7] == self.node_type).flatten()
                clusts = form_clusters(data[mask], self.node_min_size)
                clusts = [mask[c].cpu().numpy() for c in clusts]
            else:
                clusts = form_clusters(data, self.node_min_size)

        if not len(clusts):
            return {}

        # Get the batch id for each cluster
        batch_ids = get_cluster_batch(data, clusts)

        # Compute the cluster distance matrix, if necessary
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst':
            dist_mat = inter_cluster_distance(data[:,:3], clusts, batch_ids, self.edge_dist_metric, self.edge_dist_numpy)

        # Form the requested network
        if self.network == 'empty':
            edge_index = np.empty((2,0))
        elif self.network == 'loop':
            edge_index = loop_graph(len(clusts))
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(data, clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'bipartite':
            group_ids = get_cluster_label(data, clusts, column=6)
            primary_ids = get_cluster_primary(clust_ids, group_ids)
            edge_index = bipartite_graph(batch_ids, primary_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # Obtain node and edge features
        x = self.node_encoder(data, clusts)
        e = torch.empty((0, self.num_edge_feats), device=device)
        if edge_index.shape[1]:
            e = self.edge_encoder(data, clusts, edge_index)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, get output (long edge_index)
        out = self.node_predictor(x, index, e, xbatch)
        node_pred = out['node_pred'][0]

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(data[:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        node_pred = [node_pred[b] for b in bcids]
        clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]
        if edge_index.shape[1]:
            cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
            beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]
            edge_index = [cids[edge_index[:,b]].T for b in beids]
        else:
            edge_index = [np.empty((2,0)) for b in range(len(counts))]

        return {'node_pred':[node_pred],
                'edge_index':[edge_index],
                'clusts':[clusts]}


class NodeChannelLoss(torch.nn.Module):
    """
    Takes the output of EdgeModel and computes the channel-loss.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          high_purity     : <only penalize loss on groups with a primary (default False)>
    """
    def __init__(self, cfg, name='chain'):
        super(NodeChannelLoss, self).__init__()

        # Get the chain input parameters
        chain_config = cfg[name]

        # Set the loss
        self.loss = chain_config.get('loss', 'CE')
        self.reduction = chain_config.get('reduction', 'sum')
        self.balance_classes = chain_config.get('balance_classes', False)
        self.high_purity = chain_config.get('high_purity', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        n_clusts = 0
        for i in range(len(clusters)):

            # If the input did not have any node, proceed
            if 'node_pred' not in out:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = clusters[i][:,3]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = clusters[i][batches==j]

                # Use the primary information to determine the true node assignment
                node_pred = out['node_pred'][i][j]
                if not node_pred.shape[0]:
                    continue
                clusts = out['clusts'][i][j]
                clust_ids = get_cluster_label(labels, clusts)
                group_ids = get_cluster_label(labels, clusts, column=6)
                if self.high_purity:
                    purity_mask = np.zeros(len(clusts), dtype=bool)
                    for g in np.unique(group_ids):
                        group_mask = group_ids == g
                        if np.sum(group_mask) > 1 and g in clust_ids[group_mask]:
                            purity_mask[group_mask] = np.ones(np.sum(group_mask))
                    clusts    = clusts[purity_mask]
                    clust_ids = clust_ids[purity_mask]
                    group_ids = group_ids[purity_mask]
                    node_pred = node_pred[np.where(purity_mask)[0]]
                    if not len(clusts):
                        continue

                # If the majority cluster ID agrees with the majority group ID, assign as primary
                node_assn = torch.tensor(clust_ids == group_ids, dtype=torch.long, device=node_pred.device, requires_grad=False)

                # Increment the loss, balance classes if requested
                if self.balance_classes:
                    vals, counts = torch.unique(node_assn, return_counts=True)
                    weights = np.array([float(counts[k])/len(node_assn) for k in range(len(vals))])
                    for k, v in enumerate(vals):
                        total_loss += (1./weights[k])*self.lossfn(node_pred[node_assn==v], node_assn[node_assn==v])
                else:
                    total_loss += self.lossfn(node_pred, node_assn)

                # Compute accuracy of assignment (fraction of correctly assigned nodes)
                total_acc += torch.sum(torch.argmax(node_pred, dim=1) == node_assn).float()

                # Increment the number of events
                n_clusts += len(clusts)

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=clusters[0].device),
                'n_clusts': n_clusts
            }

        return {
            'accuracy': total_acc/n_clusts,
            'loss': total_loss/n_clusts,
            'n_clusts': n_clusts
        }


class NodeTypeLoss(torch.nn.Module):
    """
    Takes the output of EdgeModel and computes the channel-loss.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          high_purity     : <only penalize loss on groups with a primary (default False)>
    """
    def __init__(self, cfg, name='chain'):
        super(NodeTypeLoss, self).__init__()

        # Get the chain input parameters
        chain_config = cfg[name]

        # Set the loss
        self.loss = chain_config.get('loss', 'CE')
        self.reduction = chain_config.get('reduction', 'sum')
        self.balance_classes = chain_config.get('balance_classes', False)
        self.high_purity = chain_config.get('high_purity', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, types):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            types ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        n_clusts = 0
        for i in range(len(types)):

            # If the input did not have any node, proceed
            if 'node_pred' not in out:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = types[i][:,3]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]

                # Use the primary information to determine the true node assignment
                node_pred = out['node_pred'][i][j]
                if not node_pred.shape[0]:
                    continue
                clusts = out['clusts'][i][j]
                clust_ids = get_cluster_label(labels, clusts)
                group_ids = get_cluster_label(labels, clusts, column=6)
                if self.high_purity:
                    purity_mask = np.zeros(len(clusts), dtype=bool)
                    for g in np.unique(group_ids):
                        group_mask = group_ids == g
                        if np.sum(group_mask) > 1 and g in clust_ids[group_mask]:
                            purity_mask[group_mask] = np.ones(np.sum(group_mask))
                    clusts    = clusts[purity_mask]
                    clust_ids = clust_ids[purity_mask]
                    group_ids = group_ids[purity_mask]
                    node_pred = node_pred[np.where(purity_mask)[0]]
                    if not len(clusts):
                        continue

                # If the majority cluster ID agrees with the majority group ID, assign as primary
                node_assn = torch.tensor(clust_ids == group_ids, dtype=torch.long, device=node_pred.device, requires_grad=False)

                # Increment the loss, balance classes if requested
                if self.balance_classes:
                    vals, counts = torch.unique(node_assn, return_counts=True)
                    weights = np.array([float(counts[k])/len(node_assn) for k in range(len(vals))])
                    for k, v in enumerate(vals):
                        total_loss += (1./weights[k])*self.lossfn(node_pred[node_assn==v], node_assn[node_assn==v])
                else:
                    total_loss += self.lossfn(node_pred, node_assn)

                # Compute accuracy of assignment (fraction of correctly assigned nodes)
                total_acc += torch.sum(torch.argmax(node_pred, dim=1) == node_assn).float()

                # Increment the number of events
                n_clusts += len(clusts)

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=types[0].device),
                'n_clusts': n_clusts
            }

        return {
            'accuracy': total_acc/n_clusts,
            'loss': total_loss/n_clusts,
            'n_clusts': n_clusts
        }
