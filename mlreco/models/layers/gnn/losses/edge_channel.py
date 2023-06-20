import torch
import numpy as np

from mlreco.utils.globals import CLUST_COL, GROUP_COL, PART_COL, PSHOW_COL
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph, edge_purity_mask
from mlreco.models.experimental.bayes.evidential import EVDLoss

class EdgeChannelLoss(torch.nn.Module):
    """
    Takes the two-channel edge output of the GNN and optimizes
    edge-wise scores such that edges that connect nodes that belong
    to common instance are given a high score.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        grappa_loss:
          edge_loss:
            name:           : channel
            target_col      : <column in the label data that specifies the target group ids of each voxel (default 6)>
            batch_col       : <column in the label data that specifies the batch ids of each voxel (default 3)>
            loss            : <loss function: 'CE' or 'MM' (default 'CE')>
            reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
            balance_classes : <balance loss per class: True or False (default False)>
            target          : <type of target adjacency matrix: 'group', 'forest', 'particle_forest' (default 'group')>
            high_purity     : <only penalize loss on groups with a primary (default False)>
    """

    RETURNS = {
        'loss': ['scalar'],
        'accuracy': ['scalar'],
        'n_edges': ['scalar']
    }

    def __init__(self, loss_config, batch_col=0, coords_col=(1, 4)):
        super(EdgeChannelLoss, self).__init__()

        # Set the source and target for the loss
        self.batch_col = batch_col
        self.coords_col = coords_col

        self.target_col = loss_config.get('target_col', GROUP_COL)
        self.primary_col = loss_config.get('primary_col', PSHOW_COL)
        self.particle_col = loss_config.get('particle_col', PART_COL)

        # Set the loss
        self.loss = loss_config.get('loss', 'CE')
        self.reduction = loss_config.get('reduction', 'sum')
        self.balance_classes = loss_config.get('balance_classes', False)
        self.target = loss_config.get('target', 'group')
        self.high_purity = loss_config.get('high_purity', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        elif self.loss == 'EVD':
            evd_loss_name = loss_config.get('evd_loss_name', 'evd_nll')
            T = loss_config.get('T', 50000)
            self.lossfn = EVDLoss(evd_loss_name, reduction=self.reduction,T=T, num_classes=2, mode='evidence')
        else:
            raise ValueError('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters, graph=None):
        """
        Applies the requested loss on the edge prediction.

        Args:
            out (dict):
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
            (graph ([torch.tensor])       : (N,3) True edges, optional)
        Returns:
            double: loss, accuracy, edge count
        """
        total_loss, total_acc = 0., 0.
        n_edges = 0
        for i in range(len(clusters)):

            # If this batch did not have any node, proceed
            if 'edge_pred' not in out:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = clusters[i][:, self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = clusters[i][batches == j]
                if not labels.shape[0]:
                    continue
                # Get the output of the forward function
                edge_pred = out['edge_pred'][i][j]
                if not edge_pred.shape[0]:
                    continue
                edge_index = out['edge_index'][i][j]
                clusts     = out['clusts'][i][j]
                group_ids  = get_cluster_label(labels, clusts, self.target_col)
                part_ids   = get_cluster_label(labels, clusts, self.particle_col)

                # If a cluster target is -1, none of its edges contribute to the loss
                valid_clust_mask = group_ids > -1
                valid_mask = np.all(valid_clust_mask[edge_index], axis = -1)

                # If high purity is requested, remove edges in groups without a single primary
                if self.high_purity:
                    primary_ids  = get_cluster_label(labels, clusts, self.primary_col)
                    valid_mask  &= edge_purity_mask(edge_index, part_ids, group_ids, primary_ids)

                # Apply valid mask to edges and their predictions
                if not valid_mask.any(): continue
                edge_index = edge_index[valid_mask]
                edge_pred  = edge_pred[np.where(valid_mask)[0]]

                # Use group information or particle tree to determine the true edge assigment
                if self.target == 'group':
                    edge_assn = edge_assignment(edge_index, group_ids)
                elif self.target == 'forest':
                    # For each group, find the most likely spanning tree, label the edges in the
                    # tree as 1. For all other edges, apply loss only if in separate group.
                    # If undirected, also assign symmetric path to 1.
                    from scipy.sparse.csgraph import minimum_spanning_tree
                    edge_assn     = edge_assignment(edge_index, group_ids)
                    off_scores    = torch.softmax(edge_pred, dim=1)[:,0].detach().cpu().numpy()
                    score_mat     = np.full((len(clusts), len(clusts)), 2.0)
                    score_mat[tuple(edge_index.T)] = off_scores
                    new_edges = np.empty((0,2))
                    for g in np.unique(group_ids):
                        clust_ids = np.where(group_ids == g)[0]
                        if len(clust_ids) < 2:
                            continue

                        mst_mat = minimum_spanning_tree(score_mat[np.ix_(clust_ids,clust_ids)]+1e-6).toarray().astype(float)
                        inds = np.where(mst_mat.flatten() > 0.)[0]
                        ind_pairs = np.array(np.unravel_index(inds, mst_mat.shape)).T
                        edges = np.array([[clust_ids[i], clust_ids[j]] for i, j in ind_pairs])
                        edges = np.concatenate((edges, np.flip(edges, axis=1))) # reciprocal connections
                        new_edges = np.concatenate((new_edges, edges))

                    edge_assn_max = np.zeros(len(edge_assn))
                    for e in new_edges:
                        edge_id = np.where([(e == ei).all() for ei in edge_index])[0]
                        edge_assn_max[edge_id] = 1.

                    max_mask = edge_assn == edge_assn_max
                    edge_assn = edge_assn_max[max_mask]
                    edge_pred = edge_pred[np.where(max_mask)[0]]
                    if not len(edge_pred):
                        continue
                elif 'particle_forest' in self.target:
                    part_ids = get_cluster_label(labels, clusts, self.particle_col)
                    subgraph = graph[i][graph[i][:, self.batch_col] == j, self.coords_col[0]:self.coords_col[0]+2]
                    true_edge_index = get_fragment_edges(subgraph, part_ids)
                    edge_assn = edge_assignment_from_graph(edge_index, true_edge_index)
                else:
                    raise ValueError('Prediction target not recognized:', self.target)

                edge_assn = torch.tensor(edge_assn, device=edge_pred.device, dtype=torch.long, requires_grad=False).view(-1)

                # Increment the loss, balance classes if requested
                if self.balance_classes:
                    vals, counts = torch.unique(edge_assn, return_counts=True)
                    weights = len(edge_assn)/len(counts)/counts
                    for k, v in enumerate(vals):
                        total_loss += weights[k] * self.lossfn(edge_pred[edge_assn==v], edge_assn[edge_assn==v])
                else:
                    total_loss += self.lossfn(edge_pred, edge_assn)

                # Compute accuracy of assignment (fraction of correctly assigned edges)
                total_acc += torch.sum(torch.argmax(edge_pred, dim=1) == edge_assn).float()

                # Increment the number of edges
                n_edges += len(edge_pred)

        return {
            'accuracy': total_acc/n_edges if n_edges else 1.,
            'loss': total_loss/n_edges if n_edges else torch.tensor(0., requires_grad=True, device=clusters[0].device),
            'n_edges': n_edges
        }
