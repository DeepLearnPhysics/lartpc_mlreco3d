import torch
import numpy as np

from mlreco.utils.globals import GROUP_COL, PSHOW_COL
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.gnn.evaluation import node_assignment, node_assignment_score, node_purity_mask

class NodePrimaryLoss(torch.nn.Module):
    """
    Takes the two-channel node output of the GNN and optimizes
    node-wise scores such that nodes that initiate a particle
    cascade are given a high score (typically for showers).

    For use in config:
    model:
      name: cluster_gnn
      modules:
        grappa_loss:
          node_loss:
            name:           : primary
            batch_col       : <column in the label data that specifies the batch ids of each voxel (default 3)>
            loss            : <loss function: 'CE' or 'MM' (default 'CE')>
            reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
            balance_classes : <balance loss per class: True or False (default False)>
            high_purity     : <only penalize loss on groups with a single primary (default False)>
            use_group_pred  : <redifines group ids according to edge predictions (default False)>
            group_pred_alg  : <algorithm used to predict cluster labels: 'threshold' or 'score' (default 'score')>
    """

    RETURNS = {
        'loss': ['scalar'],
        'accuracy': ['scalar'],
        'n_clusts': ['scalar']
    }

    def __init__(self, loss_config, batch_col=0, coords_col=(1, 4)):
        super(NodePrimaryLoss, self).__init__()

        # Set the loss
        self.batch_col = batch_col
        self.coords_col = coords_col
        self.group_col = loss_config.get('group_col', GROUP_COL)
        self.primary_col = loss_config.get('primary_col', PSHOW_COL)

        self.loss = loss_config.get('loss', 'CE')
        self.reduction = loss_config.get('reduction', 'sum')
        self.balance_classes = loss_config.get('balance_classes', False)
        self.high_purity = loss_config.get('high_purity', False)
        self.use_group_pred = loss_config.get('use_group_pred', False)
        self.group_pred_alg = loss_config.get('group_pred_alg', 'score')

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise ValueError('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor) : (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])    : [(N_0), (N_1), ..., (N_C)] Cluster ids
                ('edge_pred' (torch.tensor): (C,2) Two-channel edge predictions, optional)
                ('edge_index' (np.ndarray) : (E,2) Incidence matrix, optional)
            clusters ([torch.tensor])      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            double: loss, accuracy, cluster count
        """
        total_loss, total_acc = 0., 0.
        n_clusts = 0
        for i in range(len(clusters)):

            # If the input did not have any node, proceed
            if 'node_pred' not in out:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = clusters[i][:,self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the label tensor and other predictions to the batch at hand
                labels = clusters[i][batches==j]
                if not labels.shape[0]:
                    continue
                node_pred = out['node_pred'][i][j]
                if not node_pred.shape[0]:
                    continue
                clusts = out['clusts'][i][j]
                group_ids = get_cluster_label(labels, clusts, column=self.group_col)
                primary_ids = get_cluster_label(labels, clusts, column=self.primary_col)

                # If requested, relabel the group ids in the batch according to the group predictions
                if self.use_group_pred:
                    if self.group_pred_alg == 'threshold':
                        group_ids = node_assignment(out['edge_index'][i][j], np.argmax(out['edge_pred'][i][j].detach().cpu().numpy(), axis=1), len(clusts))
                    elif self.group_pred_alg == 'score':
                        group_ids = node_assignment_score(out['edge_index'][i][j], out['edge_pred'][i][j].detach().cpu().numpy(), len(clusts))
                    else:
                        raise ValueError('Group prediction algorithm not recognized: '+self.group_pred_alg)

                # If a cluster target is -1, ignore the loss associated with it
                valid_mask = primary_ids > -1

                # If requested, remove groups that do not contain exactly one primary from the loss
                if self.high_purity:
                    valid_mask &= node_purity_mask(group_ids, primary_ids)

                # Apply valid mask to nodes and their predictions
                if not valid_mask.any(): continue
                clusts      = clusts[valid_mask]
                primary_ids = primary_ids[valid_mask]
                node_pred   = node_pred[np.where(valid_mask)[0]]

                # If the majority cluster ID agrees with the majority group ID, assign as primary
                node_assn = torch.tensor(primary_ids, dtype=torch.long, device=node_pred.device, requires_grad=False)

                # Increment the loss, balance classes if requested
                if self.balance_classes:
                    vals, counts = torch.unique(node_assn, return_counts=True)
                    weights = len(node_assn)/len(counts)/counts
                    for k, v in enumerate(vals):
                        total_loss += weights[k] * self.lossfn(node_pred[node_assn==v], node_assn[node_assn==v])
                else:
                    total_loss += self.lossfn(node_pred, node_assn)

                # Compute accuracy of assignment (fraction of correctly assigned nodes)
                total_acc += torch.sum(torch.argmax(node_pred, dim=1) == node_assn).float()
                # print(i, j, torch.sum(torch.argmax(node_pred, dim=1) == node_assn).float()/len(node_assn))
                # Increment the number of nodes
                n_clusts += len(clusts)

        return {
            'accuracy': total_acc/n_clusts if n_clusts else 1.,
            'loss': total_loss/n_clusts if n_clusts else torch.tensor(0., requires_grad=True, device=clusters[0].device),
            'n_clusts': n_clusts
        }
