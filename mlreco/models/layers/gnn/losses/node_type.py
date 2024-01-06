import torch
import numpy as np
from mlreco.utils.globals import *
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.models.experimental.bayes.evidential import EVDLoss

class NodeTypeLoss(torch.nn.Module):
    """
    Takes the c-channel node output of the GNN and optimizes
    node-wise scores such that the score corresponding to the
    correct class is maximized.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        grappa_loss:
          node_loss:
            name:           : type
            batch_col       : <column in the label data that specifies the batch ids of each voxel (default 3)>
            target_col      : <column in the label data that specifies the target node class of each voxel (default 7)>
            loss            : <loss function: 'CE' or 'MM' (default 'CE')>
            reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
            balance_classes : <balance loss per class: True or False (default False)>
    """

    RETURNS = {
        'loss': ['scalar'],
        'accuracy': ['scalar'],
        'n_clusts': ['scalar']
    }

    def __init__(self, loss_config, batch_col=0, coords_col=(1, 4)):
        super(NodeTypeLoss, self).__init__()

        # Set the target for the loss
        self.batch_col = batch_col
        self.coords_col = coords_col

        self.group_col = loss_config.get('group_col', GROUP_COL)
        self.target_col = loss_config.get('target_col', INTER_COL)

        # Set the loss
        self.loss = loss_config.get('loss', 'CE')
        self.reduction = loss_config.get('reduction', 'sum')
        self.high_purity = loss_config.get('high_purity', False)
        self.balance_classes = loss_config.get('balance_classes', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=-1)
        elif self.loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        elif self.loss == 'EVD':
            evd_loss_name = loss_config['evd_loss_name']
            T = loss_config.get('T', 50000)
            self.lossfn = EVDLoss(evd_loss_name, reduction=self.reduction, T=T, num_classes=5, mode='evidence')
        else:
            raise ValueError('Loss not recognized: ' + self.loss)

    def forward(self, out, types):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            types ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, pdg]
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
            batches = types[i][:, self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]

                # Get the class labels from the specified column
                node_pred = out['node_pred'][i][j]
                if not node_pred.shape[0]:
                    continue
                clusts = out['clusts'][i][j]
                node_assn = get_cluster_label(labels, clusts, column=self.target_col)

                # Do not apply loss to nodes labeled -1 (unknown class)
                valid_mask = node_assn > -1

                # Do not apply loss if the logit corresponding to the true class is -inf (forbidden)
                # Not a problem is node_assn_type is -1, as these rows are excluded by previous mask
                valid_mask &= (node_pred[np.arange(len(node_assn)),node_assn] != -float('inf')).detach().cpu().numpy()

                # If high purity is requested, do not include broken particle in the loss
                if self.high_purity:
                    group_ids    = get_cluster_label(labels, clusts, column=self.group_col)
                    _, inv, cnts = np.unique(group_ids, return_inverse=True, return_counts=True)
                    valid_mask &= (cnts[inv] == 1)
                valid_mask = np.where(valid_mask)[0]

                # Compute loss
                if len(valid_mask):
                    node_pred = node_pred[valid_mask]
                    node_assn = torch.tensor(node_assn[valid_mask], dtype=torch.long, device=node_pred.device, requires_grad=False)

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

                    # Increment the number of nodes
                    n_clusts += len(valid_mask)

        return {
            'accuracy': total_acc/n_clusts if n_clusts else 1.,
            'loss': total_loss/n_clusts if n_clusts else torch.tensor(0., requires_grad=True, device=types[0].device),
            'n_clusts': n_clusts
        }
