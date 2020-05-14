# GNN that selects edges iteratively until there are no edges left to select
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import edge_model_construct
from mlreco.utils.gnn.cluster import form_clusters, reform_clusters, get_cluster_batch, get_cluster_label
from mlreco.utils.gnn.network import bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph, clustering_metrics

class IterativeEdgeModel(torch.nn.Module):
    """
    Driver class for iterative edge prediction, assumed to be a GNN model.
    The model essentially operates as follows:
      1) Start with set of primary nodes
      2) Builds a bipartite graph
      3) Predicts edge strength
      4) Adds nodes that are strongly connected to the list of primaries
      5) Repeat 3, 4, 5 until no more edge is added

    For use in config:
    model:
      name: cluster_iter_gnn
      modules:
        edge_model:
          name: <name of the edge model>
          model_cfg:
            <dictionary of arguments to pass to the model>
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          node_encoder    : <node feature encoding: 'basic' or 'cnn' (default 'basic')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          maxiter         : <maximum number of iterations (default 10)>
          thresh          : <threshold edge score to consider a secondary to be matched>
          model_path      : <path to the model weights>
    """

    MODULES = ['iter_edge_model']

    def __init__(self, cfg):
        super(IterativeEdgeModel, self).__init__()

        # Get the model input parameters
        self.model_config = cfg['iter_edge_model']

        # Choose what type of node to use
        self.node_type = self.model_config.get('node_type', 0)
        self.node_min_size = self.model_config.get('node_min_size', -1)
        self.node_encoder = self.model_config.get('node_encoder', 'basic')

        # Choose what type of network to use
        self.network = self.model_config.get('network', 'complete')
        self.edge_max_dist = self.model_config.get('edge_max_dist', -1)
        self.edge_dist_metric = self.model_config.get('edge_dist_metric','set')

        # Extract the model to use
        edge_model = edge_model_construct(self.model_config.get('name', {'edge_model': {}}))

        # Construct the model
        self.edge_predictor = edge_model#(self.model_config.get('model_cfg', {}))

        # Parse the iterative model parameters
        self.maxiter = self.model_config.get('maxiter', 10)
        self.thresh = self.model_config.get('thresh', 0.9)

    @staticmethod
    def default_return(device):
        """
        Default return when no valid node is found in the input data.

        Args:
            device (torch.device): Device on which the input is stored
        Returns:
            dict:
                'edge_pred' (torch.tensor): (0,2) Empty two-channel edge predictions
                'clust_ids' (np.ndarray)  : (0) Empty cluster ids
                'batch_ids' (np.ndarray)  : (0) Empty cluster batch ids
                'primary_ids' (np.ndarray): (0) Empty primary cluster ids
                'edge_index' (np.ndarray) : (2,0) Empty incidence matrix
                'matched' (np.ndarray)    : (0) Empty cluster group predictions
                'counter' (np.ndarray)    : (1) Zero counter
        """
        xg = torch.empty((0,2), requires_grad=True, device=device)
        x  = np.empty(0)
        e  = np.empty((2,0))
        return {'edge_pred':[xg], 'clust_ids':[x], 'batch_ids':[x], 'primary_ids':[x], 'edge_index':[e], 'matched':[x], 'counter':[np.array([0])]}

    @staticmethod
    def assign_clusters(edge_index, edge_pred, others, matched, thresh):
        """
        Given edge predictions, assign secondaries to the primary they
        are most likely to be conneced to.

        Args:
            edge_index (np.ndarray) : (2,E) Incidence matrix
            edge_pred (torch.tensor): (E,2) Two-channel edge predictions
            others (np.ndarray)     : (O) Secondary cluster ids
            matched (np.ndarray)    : (C) Current cluster group predictions
            thresh (double)         : Threshold to add a node into an existing group
        Returns:
            matched (np.ndarray): (C) Updated cluster group predictions
            found_match (bool)  : True if a secondary was added to the matched list
        """
        found_match = False
        scores = torch.nn.functional.softmax(edge_pred,dim=1)[:,1]
        for i in others:
            inds = np.where(edge_index[1,:] == i)[0]
            if not len(inds):
                continue
            indmax = torch.argmax(scores[inds])
            ei = inds[indmax]
            if scores[ei] > thresh:
                found_match = True
                j = edge_index[0, ei]
                matched[i] = matched[j]

        return matched, found_match

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the iterative GNN model.

        Args:
            data ([torch.tensor]): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            dict:
                'edge_pred' ([torch.tensor]): List of (E,2) Two-channel edge predictions
                'matched' ([np.ndarray])    : List of (C) Cluster group predictions
                'edge_index' ([np.ndarray]) : List of (2,E) Incidence matrix
                'counter' (np.ndarray)      : (1) Iteration counter
                'clust_ids' (np.ndarray)    : (C) Cluster ids
                'batch_ids' (np.ndarray)    : (C) Cluster batch ids
                'primary_ids' (np.ndarray)  : (P) Primary cluster ids
        """
        # Get original device, bring data to CPU (for data preparation)
        device = data[0].device
        cluster_label = data[0].detach().cpu().numpy()

        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        if self.node_type > -1:
            mask = np.where(cluster_label[:,-1] == 0)[0]
            clusts = form_clusters(cluster_label[mask], self.node_min_size)
            clusts = np.array([mask[c] for c in clusts])
        else:
            clusts = form_clusters(cluster_label, self.node_min_size)

        if not len(clusts):
            return self.default_return(device)

        # Get the batch, cluster and group id of each cluster
        batch_ids = get_cluster_batch(cluster_label, clusts)
        clust_ids = get_cluster_label(cluster_label, clusts)

        # Identify the primary clusters
        group_ids = get_cluster_group(cluster_label, clusts)
        primary_ids = np.where(clust_ids == group_ids)[0]

        # Obtain node features
        if self.node_encoder == 'basic':
            x = torch.tensor(cluster_vtx_features(cluster_label, clusts), device=device, dtype=torch.float)
        elif self.node_encoder == 'cnn':
            raise NotImplementedError('CNN encoder not yet implemented...')
        else:
            raise ValueError('Node encoder not recognized: '+self.node_encoding)

        # If the maximum length of edges is to restricted,
        # compute the intercluster distance matrix
        dist_mat = None
        if self.edge_max_dist > 0:
            dist_mat = inter_cluster_distance(cluster_label[:,:3], clusts, self.edge_dist_metric)

        # Keep track of who is matched. -1 is not matched
        xbatch = torch.tensor(batch_ids, device=device)
        matched = np.repeat(-1, len(clusts))
        matched[primary_ids] = primary_ids

        counter = 0
        edge_index = []
        edge_pred = []
        found_match = True

        while (-1 in matched) and (counter < self.maxiter) and found_match:
            # Continue until either:
            # 1. Everything is matched
            # 2. We have exceeded the max number of iterations
            # 3. We didn't find any matches
            counter += 1

            # Get matched indices
            assigned = np.where(matched >  -1)[0]
            others   = np.where(matched == -1)[0]

            # Form a bipartite graph between assigned clusters and others
            edges = bipartite_graph(batch_ids, assigned, dist_mat, self.edge_max_dist)

            # Check if there are any edges to predict also batch norm will fail
            # on only 1 edge, so break if this is the case
            if edges.shape[1] < 2:
                counter -= 1
                break

            # Obtain edge features
            e = torch.tensor(cluster_edge_features(cluster_label, clusts, edges), device=device, dtype=torch.float)

            # Pass through the model, get output
            index = torch.tensor(edges, device=device, dtype=torch.long)
            out = self.edge_predictor(x, index, e, xbatch)

            # Predictions for this edge set.
            pred = out['edge_pred'][0]
            edge_pred.append(pred)
            edge_index.append(edges)

            # Assign group ids to new clusters
            matched, found_match = self.assign_clusters(edges, pred, others, matched, self.thresh)

        return {'edge_pred':[edge_pred],
                'matched':[matched],
                'edge_index':[edge_index],
                'counter':[np.array([counter])],
                'clust_ids':[clust_ids],
                'batch_ids':[batch_ids],
                'primary_ids':[primary_ids]}


class IterEdgeChannelLoss(torch.nn.Module):
    """
    Takes the output of IterEdgeModel and computes the channel-loss.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        edge_model:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target_photons  : <use true photon connections as basis for loss (default False)>
    """
    def __init__(self, cfg):
        super(IterEdgeChannelLoss, self).__init__()

        # Get the model input parameters
        self.model_config = cfg

        # Set the loss
        self.loss = self.model_config.get('loss', 'CE')
        self.reduction = self.model_config.get('reduction', 'mean')
        self.balance_classes = self.model_config.get('balance_classes', False)
        self.target_photons = self.model_config.get('target_photons', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters, graph):
        """
        Applies the requested loss on the edge prediction.

        Args:
            out (dict):
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clust_ids' (np.ndarray)  : (C) Cluster ids
                'batch_ids' (np.ndarray)  : (C) Cluster batch ids
                'edge_index' (np.ndarray) : (2,E) Incidence matrix
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
            graph ([torch.tensor])        : (N,3) True edges
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        total_iter = 0
        ngpus = len(clusters)
        out['group_ids'] = []
        for i in range(ngpus):

            # If the input did not have any node, proceed
            if not len(out['clust_ids'][i]):
                if ngpus == 1:
                    total_loss = torch.tensor(0., requires_grad=True, device=out['edge_pred'][i].device)
                ngpus = max(1, ngpus-1)
                continue

            # Get list of IDs of points contained in each cluster
            cluster_label = clusters[i].detach().cpu().numpy()
            clust_ids = out['clust_ids'][i]
            batch_ids = out['batch_ids'][i]
            clusts = reform_clusters(cluster_label, clust_ids, batch_ids)

            # Append the number of iterations
            niter = out['counter'][i][0]
            total_iter += niter

            # Loop over iterations and add loss at each step based
            # on the graph formed at that iteration.
            group_ids = get_cluster_group(cluster_label, clusts)
            out['group_ids'].append(group_ids)
            graph = graph[i].detach().cpu().numpy()
            true_edge_index = get_fragment_edges(graph, clust_ids, batch_ids)
            for j in range(niter):
                # Use group information or particle tree to determine the true edge assigment
                edge_pred = out['edge_pred'][i][j]
                edge_index = out['edge_index'][i][j]
                if not self.target_photons:
                    edge_assn = edge_assignment(edge_index, batch_ids, group_ids)
                else:
                    edge_assn = edge_assignment_from_graph(edge_index, true_edge_index)

                edge_assn = torch.tensor(edge_assn, device=edge_pred.device, dtype=torch.long, requires_grad=False).view(-1)

                # Increment the loss, balance classes if requested
                if self.balance_classes:
                    counts = torch.unique(edge_assn, return_counts=True)[1]
                    weights = np.array([float(counts[k])/len(edge_assn) for k in range(2)])
                    for k in range(2):
                        total_loss += (1./weights[k])*self.lossfn(edge_pred[edge_assn==k], edge_assn[edge_assn==k])
                else:
                    total_loss += self.lossfn(edge_pred, edge_assn)

                # Increment accuracy of assignment (fraction of correctly assigned edges)
                total_acc += torch.sum(torch.argmax(edge_pred, dim=1) == edge_assn).float()/edge_assn.shape[0]

        return {
            'accuracy': total_acc/max(1,total_iter),
            'loss': total_loss/ngpus,
            'n_iter': total_iter
        }
