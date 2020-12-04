import torch
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label

class LogRMSE(torch.nn.modules.loss._Loss):

    def __init__(self, reduction='none', eps=1e-7):
        super(LogRMSE, self).__init__()
        self.reduction = reduction
        self.mseloss = torch.nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, inputs, targets):
        x = torch.log(inputs + self.eps)
        y = torch.log(targets + self.eps)
        out = self.mseloss(x, y)
        out = torch.sqrt(out + self.eps)
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            return out


class BerHuLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction='none'):
        super(BerHuLoss, self).__init__()
        self.reduction == reduction

    def forward(self, inputs, targets):
        norm = torch.abs(inputs - targets)
        c = norm.max() * 0.20
        out = torch.where(norm <= c, norm, (norm**2 + c**2) / (2.0 * c))
        if self.reduction == 'sum':
            return out.sum()
        elif self.reduction == 'mean':
            return out.mean()
        else:
            return out


class NodeKinematicsLoss(torch.nn.Module):
    """
    Takes the n-features node output of the GNN and optimizes
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
            type_col        : <column in the label data that specifies the target node class (default 7)>
            momentum_col    : <column in the label data that specifies the target node momentum (default 8)>
            loss            : <loss function: 'CE' or 'MM' (default 'CE')>
            reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
            balance_classes : <balance loss per class: True or False (default False)>
    """
    def __init__(self, loss_config):
        super(NodeKinematicsLoss, self).__init__()

        # Set the target for the loss
        self.batch_col = loss_config.get('batch_col', 3)
        self.type_col = loss_config.get('target_col', 7)
        self.momentum_col = loss_config.get('target_col', 8)

        # Set the losses
        self.type_loss = loss_config.get('type_loss', 'CE')
        self.reg_loss = loss_config.get('reg_loss', 'l2')
        self.reduction = loss_config.get('reduction', 'sum')
        self.balance_classes = loss_config.get('balance_classes', False)

        if self.type_loss == 'CE':
            self.type_lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=-1)
        elif self.type_loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.type_lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise ValueError('Type loss not recognized: ' + self.type_loss)

        if self.reg_loss == 'l2':
            self.reg_lossfn = torch.nn.MSELoss(reduction=self.reduction)
        elif self.reg_loss == 'l1':
            self.reg_lossfn = torch.nn.L1Loss(reduction=self.reduction)
        elif self.reg_loss == 'log_rmse':
            self.reg_lossfn = LogRMSE(reduction=self.reduction)
        elif self.reg_loss == 'huber':
            self.reg_lossfn = torch.nn.SmoothL1Loss(reduction=self.reduction)
        elif self.reg_loss == 'berhu':
            self.reg_lossfn = BerHuLoss(reduction=self.reduction)
        else:
            raise ValueError('Regression loss not recognized: ' + self.reg_loss)


    def forward(self, out, types):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            types ([torch.tensor])     : (N,9) [x, y, z, batchid, value, id, groupid, pdg, p]
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        type_loss, p_loss, type_acc, p_acc = 0., 0., 0., 0.
        n_clusts = 0
        for i in range(len(types)):

            # If the input did not have any node, proceed
            if 'node_pred_type' not in out or 'node_pred_p' not in out:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = types[i][:,self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]

                # Get the class labels and true momenta from the specified columns
                node_pred_type = out['node_pred_type'][i][j]
                node_pred_p = out['node_pred_p'][i][j]
                if not node_pred_type.shape[0] or not node_pred_p.shape[0]:
                    continue
                clusts = out['clusts'][i][j]
                node_assn_type = get_cluster_label(labels, clusts, column=self.type_col)
                node_assn_type = torch.tensor(node_assn_type, dtype=torch.long, device=node_pred_type.device, requires_grad=False)
                node_assn_p = get_momenta_label(labels, clusts, column=self.momentum_col)

                # Increment the type loss, balance classes if requested
                if self.balance_classes:
                    vals, counts = torch.unique(node_assn_type, return_counts=True)
                    weights = np.array([float(counts[k])/len(node_assn_type) for k in range(len(vals))])
                    for k, v in enumerate(vals):
                        loss = (1./weights[k])*self.type_lossfn(node_pred_type[node_assn==v], node_assn[node_assn_type==v])
                        total_loss += loss
                        type_loss += float(loss)
                else:
                    loss = self.type_lossfn(node_pred_type, node_assn_type)
                    total_loss += loss
                    type_loss += float(loss)

                # Increment the momentum loss
                loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p)
                total_loss += loss
                p_loss += float(loss)

                # Compute the accuracy of assignment (fraction of correctly assigned nodes)
                # and the accuracy of momentum estimation (RMS relative residual)
                type_acc += float(torch.sum(torch.argmax(node_pred_type, dim=1) == node_assn_type))
                p_acc += float(torch.sum(1.- torch.abs(node_pred_p.squeeze()-node_assn_p)/node_assn_p)) # 1-MAPE

                # Increment the number of nodes
                n_clusts += len(clusts)

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=types[0].device),
                'type_loss': 0.,
                'p_loss': 0.,
                'n_clusts': n_clusts
            }

        return {
            'accuracy': (type_acc+p_acc)/(2*n_clusts),
            'loss': total_loss/n_clusts,
            'type_accuracy': type_acc/n_clusts,
            'p_accuracy': p_acc/n_clusts,
            'type_loss': type_loss/n_clusts,
            'p_loss': p_loss/n_clusts,
            'n_clusts': n_clusts
        }
