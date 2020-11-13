import torch
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_label

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
    def __init__(self, loss_config):
        super(NodeTypeLoss, self).__init__()

        # Set the target for the loss
        self.batch_col = loss_config.get('batch_col', 3)
        self.target_col = loss_config.get('target_col', 7)

        # Set the loss
        self.loss = loss_config.get('loss', 'CE')
        self.reduction = loss_config.get('reduction', 'sum')
        self.balance_classes = loss_config.get('balance_classes', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=-1)
        elif self.loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
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
            batches = types[i][:,self.batch_col]
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
                node_assn = torch.tensor(node_assn, dtype=torch.long, device=node_pred.device, requires_grad=False)

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

                # Increment the number of nodes
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
