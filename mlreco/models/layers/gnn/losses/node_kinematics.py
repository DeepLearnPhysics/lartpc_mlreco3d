import torch
import numpy as np
from mlreco.utils.globals import *
from mlreco.utils.metrics import unique_label
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label
from mlreco.models.experimental.bayes.evidential import EDLRegressionLoss, EVDLoss
from torch_scatter import scatter

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

    RETURNS = {
        'loss': ['scalar'],
        'type_loss': ['scalar'],
        'p_loss': ['scalar'],
        'vtx_score_loss': ['scalar'],
        'vtx_position_loss': ['scalar'],
        'accuracy': ['scalar'],
        'type_accuracy': ['scalar'],
        'p_accuracy': ['scalar'],
        'vtx_score_accuracy': ['scalar'],
        'vtx_position_accuracy': ['scalar'],
        'n_clusts_momentum': ['scalar'],
        'n_clusts_type': ['scalar'],
        'n_clusts_vtx': ['scalar'],
        'n_clusts_vtx_positives': ['scalar'],
        'vtx_labels': ['tensor', None, True],
        'vtx_labels': ['tensor', None, True]
    }

    def __init__(self, loss_config, batch_col=0, coords_col=(1, 4)):
        super(NodeKinematicsLoss, self).__init__()

        # Set the target for the loss
        self.batch_col = batch_col
        self.coords_col = coords_col

        self.group_col = loss_config.get('cluster_col', GROUP_COL)
        self.type_col = loss_config.get('type_col', PID_COL)
        self.vtx_col = loss_config.get('vtx_col', VTX_COLS[0])
        self.vtx_positives_col = loss_config.get('vtx_positives_col', PGRP_COL)

        # Set the losses
        self.type_loss = loss_config.get('type_loss', 'CE')
        self.reg_loss = loss_config.get('reg_loss', 'l2')
        self.reduction = loss_config.get('reduction', 'sum')
        self.balance_classes = loss_config.get('balance_classes', False)
        self.spatial_size = loss_config.get('spatial_size', 768)

        if self.type_loss == 'CE':
            self.type_lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=-1)
        elif self.type_loss == 'EVD':
            evd_loss_name = loss_config.get('evd_loss_name', 'evd_nll')
            T = loss_config.get('T', 50000)
            self.type_lossfn = EVDLoss(evd_loss_name, reduction=self.reduction,T=T, one_hot=False, mode='evidence')
        elif self.type_loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.type_lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise ValueError('Type loss not recognized: ' + self.type_loss)

        if self.reg_loss == 'l2':
            self.reg_lossfn = torch.nn.MSELoss(reduction=self.reduction)
        elif self.reg_loss == 'edl':
            w = loss_config.get('kld_weight', 0.0)
            self.reg_lossfn = EDLRegressionLoss(reduction=self.reduction, w=w)
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

        self.vtx_position_loss = torch.nn.L1Loss(reduction='none')
        self.vtx_score_loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        self.normalize_vtx_label = loss_config.get('normalize_vtx_label', True)
        self.use_anchor_points = loss_config.get('use_anchor_points', False)
        self.max_vertex_distance = loss_config.get('max_vertex_distance', 50)
        self.type_num_classes = loss_config.get('type_num_classes', 5)
        self.type_loss_weight = loss_config.get('type_loss_weight', 1.0)
        self.type_high_purity = loss_config.get('type_high_purity', True)
        self.momentum_high_purity = loss_config.get('momentum_high_purity', True)
        self.vtx_high_purity  = loss_config.get('vtx_high_purity', True)

        self.compute_momentum_switch = loss_config.get('compute_momentum', True)

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
        vtx_position_loss, vtx_score_loss, vtx_position_acc, vtx_score_acc = 0., 0., 0., 0.
        n_clusts_type, n_clusts_momentum, n_clusts_vtx, n_clusts_vtx_pos = 0, 0, 0, 0

        compute_type = 'node_pred_type' in out
        compute_momentum = 'node_pred_p' in out
        if not self.compute_momentum_switch:
            compute_momentum = False
        compute_vtx = 'node_pred_vtx' in out
        compute_vtx_pos = False # TODO: make this cleaner

        vtx_anchors, vtx_labels = [], []

        for i in range(len(types)):

            # If the input did not have any node, proceed
            if not compute_type and not compute_momentum and not compute_vtx:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = types[i][:, self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]
                if not labels.shape[0]:
                    continue
                clusts = out['clusts'][i][j]

                # Increment the type loss, balance classes if requested
                if compute_type and out['node_pred_type'][i][j].shape[0]:
                    # Get the type predictions and true types from the specified columns
                    node_pred_type = out['node_pred_type'][i][j]
                    node_assn_type = get_cluster_label(labels, clusts, column=self.type_col)

                    # Set the labels for classes above the max number to -1
                    node_assn_type[node_assn_type >= self.type_num_classes] = -1

                    # Do not apply loss to nodes labeled -1 (unknown class)
                    valid_mask_type = node_assn_type > -1

                    # Do not apply loss if the logit corresponding to the true class is -inf (forbidden prediction)
                    # Not a problem is node_assn_type is -1, as these rows will already be excluded by previous mask
                    valid_mask_type &= (node_pred_type[np.arange(len(node_assn_type)), node_assn_type] != -float('inf')).detach().cpu().numpy()

                    # If high purity is requested, do not include broken particle in the loss
                    if self.type_high_purity:
                        group_ids    = get_cluster_label(labels, clusts, column=self.group_col)
                        _, inv, cnts = np.unique(group_ids, return_inverse=True, return_counts=True)
                        valid_mask_type  &= (cnts[inv] == 1)
                    valid_mask_type = np.where(valid_mask_type)[0]

                    # Compute loss
                    if len(valid_mask_type):
                        node_pred_type = node_pred_type[valid_mask_type]
                        node_assn_type = torch.tensor(node_assn_type[valid_mask_type], dtype=torch.long, device=node_pred_type.device, requires_grad=False)
                        if self.balance_classes:
                            vals, counts = torch.unique(node_assn_type, return_counts=True)
                            weights = len(node_assn_type)/len(counts)/counts
                            for k, v in enumerate(vals):
                                loss = weights[k] * self.type_lossfn(node_pred_type[node_assn_type==v],
                                                                     node_assn_type[node_assn_type==v])
                                total_loss += self.type_loss_weight * loss
                                type_loss += self.type_loss_weight * float(loss)
                        else:
                            loss = self.type_lossfn(node_pred_type, node_assn_type)
                            total_loss += self.type_loss_weight * loss
                            type_loss += self.type_loss_weight * float(loss)

                        # Increment the number of nodes
                        n_clusts_type += len(valid_mask_type)

                # Increment the momentum loss
                if compute_momentum and out['node_pred_p'][i][j].shape[0]:
                    # Get the momentum predictions and true momenta from the specified columns
                    node_pred_p = out['node_pred_p'][i][j]
                    node_assn_p = get_momenta_label(labels, clusts)

                    # Do not apply loss to nodes labeled -1 (unknown class)
                    valid_mask_p = node_assn_p.detach().cpu().numpy() > -1

                    # If high purity is requested, do not include broken particle in the loss
                    if self.momentum_high_purity:
                        group_ids    = get_cluster_label(labels, clusts, column=self.group_col)
                        _, inv, cnts = np.unique(group_ids, return_inverse=True, return_counts=True)
                        valid_mask_p  &= (cnts[inv] == 1)
                    valid_mask_p = np.where(valid_mask_p)[0]

                    # Compute loss
                    if len(valid_mask_p):
                        node_pred_p = node_pred_p[valid_mask_p]
                        node_assn_p = node_assn_p[valid_mask_p]

                        loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p.float())
                        total_loss += loss
                        p_loss += float(loss)

                        # Increment the number of nodes
                        n_clusts_momentum += len(clusts)

                if compute_vtx and out['node_pred_vtx'][i][j].shape[0]:
                    # Get the vertex predictions, node features and true vertices from the specified columns
                    node_pred_vtx = out['node_pred_vtx'][i][j]
                    node_features = out['node_features'][i][j]
                    node_assn_vtx     = np.stack([get_cluster_label(labels, clusts, column=c) for c in range(self.vtx_col, self.vtx_col+3)], axis=1)
                    node_assn_vtx_pos = get_cluster_label(labels, clusts, column=self.vtx_positives_col)
                    compute_vtx_pos   = node_pred_vtx.shape[-1] == 5

                    # Do not apply loss to nodes labeled -1 or nodes with vertices outside of volume (TODO: this is weak if the volume is not a cube)
                    valid_mask_vtx = (node_assn_vtx >= 0.).all(axis=1) & (node_assn_vtx <= self.spatial_size).all(axis=1) & (node_assn_vtx_pos > -1)

                    # If high purity is requested, do not include broken particle in the loss
                    if self.vtx_high_purity:
                        group_ids       = get_cluster_label(labels, clusts, column=self.group_col)
                        _, inv, cnts    = np.unique(group_ids, return_inverse=True, return_counts=True)
                        valid_mask_vtx &= cnts[inv] == 1
                    valid_mask_vtx = np.where(valid_mask_vtx)[0]

                    # Compute the losses only if there is at least > 1 true positive node 
                    pos_mask_vtx = np.where(node_assn_vtx_pos[valid_mask_vtx])[0]
                    if len(pos_mask_vtx):
                        # Compute the primary score loss on all valid nodes
                        node_pred_vtx     = node_pred_vtx[valid_mask_vtx]
                        node_assn_vtx_pos = torch.tensor(node_assn_vtx_pos[valid_mask_vtx], dtype=torch.long, device=node_pred_vtx.device)
                        if not compute_vtx_pos:
                            loss1 = self.vtx_score_loss(node_pred_vtx, node_assn_vtx_pos)
                            vtx_score_loss += float(loss1)
                            total_loss += loss1
                            n_clusts_vtx += len(valid_mask_vtx)
                        else:
                            loss1 = self.vtx_score_loss(node_pred_vtx[:, 3:], node_assn_vtx_pos)

                            # Compute the vertex position loss on positive nodes only
                            vtx_label = torch.tensor(node_assn_vtx[valid_mask_vtx][pos_mask_vtx], dtype=node_pred_vtx.dtype, device=node_pred_vtx.device)
                            if self.normalize_vtx_label: # If requested, bring vertex labels in the range [0,1 ]
                                vtx_label = vtx_label/self.spatial_size
                            vtx_labels.append(vtx_label.detach().cpu().numpy())

                            vtx_pred = node_pred_vtx[pos_mask_vtx,:3]
                            if self.use_anchor_points: # If requested, predict positions with respect to anchor points (end points of particles)
                                end_points = node_features[valid_mask_vtx,19:25][pos_mask_vtx].view(-1, 2, 3)
                                dist_to_anchor = torch.norm(vtx_pred.view(-1, 1, 3) - end_points, dim=2).view(-1, 2)
                                min_dist = torch.argmin(dist_to_anchor, dim=1)
                                range_index = torch.arange(end_points.shape[0]).to(device=end_points.device).long()
                                anchors = end_points[range_index, min_dist, :]
                                vtx_anchors.append(anchors.detach().cpu().numpy())
                                vtx_pred = vtx_pred + anchors

                            loss2 = torch.mean(torch.clamp(torch.sum(self.vtx_position_loss(vtx_pred, vtx_label), dim=1),
                                                           max=self.max_vertex_distance**2))

                            # Combine losses
                            total_loss += loss1 + loss2
                            vtx_score_loss += float(loss1)
                            vtx_position_loss += float(loss2)

                            # Increment the number of nodes
                            n_clusts_vtx += len(valid_mask_vtx)
                            n_clusts_vtx_pos += len(pos_mask_vtx)
                    else:
                        vtx_labels.append(np.empty((0,3), dtype=np.float32))
                        if self.use_anchor_points: anchors.append(np.empty((0,3)))

                # Compute the accuracy of assignment (fraction of correctly assigned nodes)
                # and the accuracy of momentum estimation (RMS relative residual)
                if compute_type and out['node_pred_type'][i][j].shape[0] and len(valid_mask_type):
                    type_acc += float(torch.sum(torch.argmax(node_pred_type, dim=1) == node_assn_type))

                if compute_momentum and out['node_pred_p'][i][j].shape[0] and len(valid_mask_p):
                    p_acc += float(torch.sum(1.- torch.abs(node_pred_p.squeeze()-node_assn_p)/node_assn_p)) # 1-MAPE

                if compute_vtx and out['node_pred_vtx'][i][j].shape[0] and len(pos_mask_vtx):
                    compute_vtx_pos = node_pred_vtx.shape[-1] == 5
                    if not compute_vtx_pos:
                        vtx_score_acc += float(torch.sum(torch.argmax(node_pred_vtx, dim=1) == node_assn_vtx_pos))
                    else:
                        vtx_score_acc += float(torch.sum(torch.argmax(node_pred_vtx[:,3:], dim=1) == node_assn_vtx_pos))
                        vtx_position_acc += float(torch.sum(1. - torch.abs(vtx_pred - vtx_label)/(torch.abs(vtx_pred) + torch.abs(vtx_label))))/3.

        n_clusts = n_clusts_type + n_clusts_momentum + n_clusts_vtx + n_clusts_vtx_pos

        result = {
            'accuracy': (type_acc + p_acc + vtx_position_acc + vtx_score_acc)/n_clusts if n_clusts else 1.,
            'loss': total_loss/n_clusts if n_clusts else torch.tensor(0., requires_grad=True, device=types[0].device, dtype=torch.float),
            'n_clusts_momentum': n_clusts_momentum,
            'n_clusts_type': n_clusts_type,
            'n_clusts_vtx': n_clusts_vtx,
            'n_clusts_vtx_positives': n_clusts_vtx_pos
        }

        if compute_type:
            result.update({
                'type_accuracy': type_acc/n_clusts_type if n_clusts_type else 1.,
                'type_loss': type_loss/n_clusts_type if n_clusts_type else 0.
            })
        if compute_momentum:
            result.update({
                'p_accuracy': p_acc/n_clusts_momentum if n_clusts_momentum else 1.,
                'p_loss': p_loss/n_clusts_momentum if p_loss else 0.
            })
        if compute_vtx:
            result.update({
                'vtx_score_loss': vtx_score_loss/n_clusts_vtx if n_clusts_vtx else 0.,
                'vtx_score_accuracy': vtx_score_acc/n_clusts_vtx if n_clusts_vtx else 1.,
                'vtx_position_loss': vtx_position_loss/n_clusts_vtx_pos if n_clusts_vtx_pos else 0.,
                'vtx_position_accuracy': vtx_position_acc/n_clusts_vtx_pos if n_clusts_vtx_pos else 1.
            })
            if compute_vtx_pos: result['vtx_labels'] = vtx_labels,
            if self.use_anchor_points: result['vtx_anchors'] = vtx_anchors

        return result



class NodeEvidentialKinematicsLoss(NodeKinematicsLoss):

    def __init__(self, loss_config, **kwargs):
        super(NodeEvidentialKinematicsLoss, self).__init__(loss_config, **kwargs)
        evd_loss_name = loss_config.get('evd_loss_name', 'evd_nll')
        T = loss_config.get('T', 50000)
        self.type_lossfn = EVDLoss(evd_loss_name,
                                   reduction='sum',
                                   T=T,
                                   one_hot=False,
                                   mode='evidence')
        w = loss_config.get('kld_weight', 0.0)
        if self.reg_loss == 'l2':
            self.reg_lossfn = torch.nn.MSELoss(reduction=self.reduction)
        elif self.reg_loss == 'edl':
            w = loss_config.get('kld_weight', 0.0)
            kl_mode = loss_config.get('kl_mode', 'evd')
            logspace = loss_config.get('logspace', False)
            print("logspace = ", logspace)
            self.reg_lossfn = EDLRegressionLoss(reduction=self.reduction, w=w, kl_mode=kl_mode, logspace=logspace)
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

    def compute_type(self, node_pred_type, labels, clusts, iteration):
        '''

        '''
        # assert False
        total_loss, n_clusts_type, type_loss = 0, 0, 0

        node_assn_type = get_cluster_label(labels, clusts, column=self.type_col)
        node_assn_type = torch.tensor(node_assn_type, dtype=torch.long,
                                                      device=node_pred_type.device,
                                                      requires_grad=False)

        # Do not apply loss to nodes labeled -1 (unknown class)
        node_mask = torch.nonzero(node_assn_type > -1, as_tuple=True)[0]
        if len(node_mask):
            node_pred_type = node_pred_type[node_mask]
            node_assn_type = node_assn_type[node_mask]

            if self.balance_classes:
                vals, counts = torch.unique(node_assn_type, return_counts=True)
                weights = len(node_assn_type)/len(counts)/counts
                for k, v in enumerate(vals):
                    loss = weights[k] * self.type_lossfn(node_pred_type[node_assn_type==v],
                                                         node_assn_type[node_assn_type==v],
                                                         T=iteration)
                    total_loss += loss
                    type_loss += float(loss)
            else:
                loss = self.type_lossfn(node_pred_type, node_assn_type)
                total_loss += torch.clamp(loss, min=0)
                type_loss += float(torch.clamp(loss, min=0))

            # Increment the number of nodes
            n_clusts_type += len(node_mask)

        with torch.no_grad():
            type_acc = float(torch.sum(torch.argmax(node_pred_type, dim=1) == node_assn_type))

        # print("TYPE: {}, {}, {}, {}".format(
        #     float(total_loss), float(type_loss), float(n_clusts_type), float(type_acc)))

        return total_loss, type_loss, n_clusts_type, type_acc


    def compute_momentum(self, node_pred_p, labels, clusts, iteration=None):

        node_assn_p = get_momenta_label(labels, clusts)
        with torch.no_grad():
            p_acc = torch.pow(node_pred_p[:, 0]-node_assn_p, 2).sum()
        n_clusts_momentum = len(clusts)

        if self.reg_loss == 'edl':
            loss, nll_loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p.float())
            nll_loss = torch.clamp(nll_loss, min=0).detach().cpu().numpy()
            p_acc = np.exp(-nll_loss)
            loss, nll_loss, p_acc = loss.nansum(), float(np.nansum(nll_loss)), np.sum(p_acc)
            # print("Momentum: {}, {}, {}, {}".format(
            #     float(loss), float(nll_loss), float(n_clusts_momentum), float(p_acc)))
            return loss, float(nll_loss), n_clusts_momentum, p_acc
        else:
            loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p.float())
            return loss, float(loss), n_clusts_momentum, p_acc


    def compute_vertex(self, node_pred_vtx, labels, clusts):
        node_x_vtx = get_cluster_label(labels, clusts, column=self.vtx_col)
        node_y_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+1)
        node_z_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+2)

        node_assn_vtx = torch.tensor(np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1),
                                    dtype=torch.float, device=node_pred_vtx.device, requires_grad=False)
        node_assn_vtx = node_assn_vtx/self.spatial_size

        # Exclude vertex that is outside of the volume
        good_index = torch.all(torch.abs(node_assn_vtx) <= 1., dim=1)

        #positives = get_cluster_label(labels, clusts, column=self.vtx_positives_col)
        # Take the max for each cluster - e.g. for a shower, the primary fragment only
        # is marked as primary particle, so taking majority count would eliminate the shower
        # from primary particles for vertex identification purpose.
        positives = []
        for c in clusts:
            positives.append(labels[c, self.vtx_positives_col].max().item())
        positives = np.array(positives)

        positives = torch.tensor(positives, dtype=torch.long,
                                            device=node_pred_vtx.device,
                                            requires_grad=False)
        # for now only sum losses, they get averaged below in results dictionary
        loss2 = self.vtx_score_loss(node_pred_vtx[good_index, 3:], positives[good_index])
        loss1 = torch.sum(torch.mean(
            self.vtx_position_loss(
                node_pred_vtx[good_index & positives.bool(), :3],
                node_assn_vtx[good_index & positives.bool()]), dim=1))

        loss = loss1 + loss2

        vtx_position_loss = float(loss1)
        vtx_score_loss = float(loss2)

        n_clusts_vtx = (good_index).sum().item()
        n_clusts_vtx_positives = (good_index & positives.bool()).sum().item()

        if node_pred_vtx[good_index].shape[0]:
            # print(node_pred_vtx[good_index & positives.bool(), :3], node_assn_vtx[good_index & positives.bool()])
            vtx_position_acc = float(torch.sum(1. - torch.abs(node_pred_vtx[good_index & positives.bool(), :3] - \
                                                              node_assn_vtx[good_index & positives.bool()]) / \
                                    (torch.abs(node_assn_vtx[good_index & positives.bool()]) + \
                                     torch.abs(node_pred_vtx[good_index & positives.bool(), :3]))))/3.
            vtx_score_acc = float(torch.sum(
                torch.argmax(node_pred_vtx[good_index, 3:], dim=1) == positives[good_index]))

        out =  (loss,
                vtx_position_loss,
                vtx_score_loss,
                n_clusts_vtx,
                n_clusts_vtx_positives,
                vtx_position_acc,
                vtx_score_acc)

        return out


    def forward(self, out, types, iteration=None):
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
        vtx_position_loss, vtx_score_loss, vtx_position_acc, vtx_score_acc = 0., 0., 0., 0.
        n_clusts_type, n_clusts_momentum, n_clusts_vtx, n_clusts_vtx_positives = 0, 0, 0, 0

        compute_type = 'node_pred_type' in out
        compute_momentum = 'node_pred_p' in out
        if not self.compute_momentum_switch:
            compute_momentum = False
        compute_vtx = 'node_pred_vtx' in out

        for i in range(len(types)):

            # If the input did not have any node, proceed
            if not compute_type and not compute_momentum and not compute_vtx:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = types[i][:,self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]

                clusts = out['clusts'][i][j]

                # Increment the type loss, balance classes if requested
                if compute_type:
                    # Get the class labels and true momenta from the specified columns
                    node_pred_type = out['node_pred_type'][i][j]
                    if not node_pred_type.shape[0]:
                        continue
                    type_data = self.compute_type(node_pred_type, labels, clusts, iteration)

                    total_loss    += type_data[0]
                    type_loss     += type_data[1]
                    n_clusts_type += type_data[2]
                    type_acc      += type_data[3]

                # Increment the momentum loss
                if compute_momentum:
                    # Get the class labels and true momenta from the specified columns
                    node_pred_p = out['node_pred_p'][i][j]
                    if not node_pred_p.shape[0]:
                        continue
                    p_data = self.compute_momentum(node_pred_p, labels, clusts, iteration)


                    total_loss        += p_data[0]
                    p_loss            += float(p_data[1])
                    n_clusts_momentum += p_data[2]
                    p_acc             += p_data[3] # Convert NLL loss to Model Evidence
                if compute_vtx:
                    node_pred_vtx = out['node_pred_vtx'][i][j]
                    if not node_pred_vtx.shape[0]:
                        continue

                    vtx_data = self.compute_vertex(node_pred_vtx, labels, clusts)

                    total_loss             += vtx_data[0]
                    vtx_position_loss      += vtx_data[1]
                    vtx_score_loss         += vtx_data[2]
                    n_clusts_vtx           += vtx_data[3]
                    n_clusts_vtx_positives += vtx_data[4]
                    vtx_position_acc       += vtx_data[5]
                    vtx_score_loss         += vtx_data[6]

                # Compute the accuracy of assignment (fraction of correctly assigned nodes)
                # and the accuracy of momentum estimation (RMS relative residual)

        n_clusts = n_clusts_type + n_clusts_momentum + n_clusts_vtx + n_clusts_vtx_positives

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            result = {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True,
                                         device=types[0].device,
                                         dtype=torch.float),
                'n_clusts_momentum': n_clusts_momentum,
                'n_clusts_type': n_clusts_type,
                'n_clusts_vtx': n_clusts_vtx,
                'n_clusts_vtx_positives': n_clusts_vtx_positives
            }
            if compute_type:
                result.update({
                    'type_loss': 0.,
                    'type_accuracy': 0.,
                })
            if compute_momentum:
                result.update({
                    'p_loss': 0.,
                    'p_accuracy': 0.,
                })
            if compute_vtx:
                result.update({
                    'vtx_position_loss': 0.,
                    'vtx_score_loss': 0.,
                    'vtx_position_accurary': 0.,
                    'vtx_score_accuracy': 0.,
                })
            return result

        result = {
            'accuracy': (type_acc+p_acc+vtx_position_acc+vtx_score_acc)/n_clusts,
            'loss': total_loss/n_clusts,
            'n_clusts_momentum': n_clusts_momentum,
            'n_clusts_type': n_clusts_type,
            'n_clusts_vtx': n_clusts_vtx,
            'n_clusts_vtx_positives': n_clusts_vtx_positives
        }

        if compute_type:
            result.update({
                'type_accuracy': 0. if not n_clusts_type else type_acc/n_clusts_type,
                'type_loss': 0. if not n_clusts_type else type_loss/n_clusts_type,
            })
        if compute_momentum:
            result.update({
                'p_accuracy': 0. if not n_clusts_momentum else p_acc/n_clusts_momentum,
                'p_loss': 0. if not n_clusts_momentum else p_loss/n_clusts_momentum,
            })
        if compute_vtx:
            result.update({
                'vtx_score_loss': 0. if not n_clusts_vtx else vtx_score_loss/n_clusts_vtx,
                'vtx_score_accurary': 0. if not n_clusts_vtx else vtx_score_acc/n_clusts_vtx,
                'vtx_position_loss': 0. if not n_clusts_vtx_positives else vtx_position_loss/n_clusts_vtx_positives,
                'vtx_position_accuray': 0. if not n_clusts_vtx_positives else vtx_position_acc/n_clusts_vtx_positives,
            })

        return result


class NodeTransformerLoss(NodeEvidentialKinematicsLoss):
    '''
    Vertex Loss Override for Transformer-like vertex net.
    '''
    def __init__(self, loss_config, **kwargs):
        super(NodeEvidentialKinematicsLoss, self).__init__(loss_config, **kwargs)
        self.type_lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=-1)
        self.use_primaries_for_vtx = loss_config.get('use_primaries_for_vtx', False)
        self.avg_pool_loss = loss_config.get('avg_pool_loss', False)
        self.normalize_vtx_label = loss_config.get('normalize_vtx_label', True)

    def compute_type(self, node_pred_type, labels, clusts, iteration=None):
        '''
        Compute particle classification type loss.
        '''
        # assert False
        total_loss, n_clusts_type, type_loss = 0, 0, 0

        node_assn_type = get_cluster_label(labels, clusts, column=self.type_col)
        node_assn_type = torch.tensor(node_assn_type, dtype=torch.long,
                                                      device=node_pred_type.device,
                                                      requires_grad=False)

        # Do not apply loss to nodes labeled -1 (unknown class)
        node_mask = torch.nonzero(node_assn_type > -1, as_tuple=True)[0]
        if len(node_mask):
            node_pred_type = node_pred_type[node_mask]
            node_assn_type = node_assn_type[node_mask]

            if self.balance_classes:
                vals, counts = torch.unique(node_assn_type, return_counts=True)
                weights = len(node_assn_type)/len(counts)/counts
                for k, v in enumerate(vals):
                    loss = weights[k] * self.type_lossfn(node_pred_type[node_assn_type==v],
                                                         node_assn_type[node_assn_type==v])
                    total_loss += loss
                    type_loss += float(loss)
            else:
                loss = self.type_lossfn(node_pred_type, node_assn_type)
                total_loss += torch.clamp(loss, min=0)
                type_loss += float(torch.clamp(loss, min=0))

            # Increment the number of nodes
            n_clusts_type += len(node_mask)

        with torch.no_grad():
            type_acc = float(torch.sum(torch.argmax(node_pred_type, dim=1) == node_assn_type))

        # print("TYPE: {}, {}, {}, {}".format(
        #     float(total_loss), float(type_loss), float(n_clusts_type), float(type_acc)))

        return total_loss, type_loss, n_clusts_type, type_acc

    def compute_vertex(self, node_pred_vtx, labels, clusts):

        node_x_vtx = get_cluster_label(labels, clusts, column=self.vtx_col)
        node_y_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+1)
        node_z_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+2)

        node_assn_vtx = torch.tensor(np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1),
                                    dtype=torch.float, device=node_pred_vtx.device, requires_grad=False)

        vtx_position_loss = 0
        vtx_score_loss = 0
        loss = 0
        n_clusts_vtx = 0
        n_clusts_vtx_positives = 0
        vtx_position_acc = 0
        vtx_score_acc = 0

        # Select primaries for vertex regression
        select_primaries = get_cluster_label(labels, clusts,
                                             column=self.vtx_positives_col)

        select_primaries = select_primaries.astype(bool)

        positives = []
        for c in clusts:
            positives.append(labels[c, self.vtx_positives_col].max().item())
        positives = np.array(positives)

        positives = torch.tensor(positives, dtype=torch.long,
                                            device=node_pred_vtx.device,
                                            requires_grad=False)

        # Use only primaries when computing vertex loss
        if self.use_primaries_for_vtx:
            # print("Before Primaries = ", node_pred_vtx.shape)
            nodes_vtx_reg = node_pred_vtx[select_primaries]
            positives_vtx = positives[select_primaries]
            node_assn_vtx = node_assn_vtx[select_primaries]
        else:
            nodes_vtx_reg = node_pred_vtx
            positives_vtx = positives

        # print("After Primaries = ", nodes_vtx_reg.shape)

        if nodes_vtx_reg.shape[0] < 1:
            out =  (loss,
                    vtx_position_loss,
                    vtx_score_loss,
                    n_clusts_vtx,
                    n_clusts_vtx_positives,
                    vtx_position_acc,
                    vtx_score_acc)
            return out

        # positives = get_cluster_label(labels, clusts, column=self.vtx_positives_col)
        # Take the max for each cluster - e.g. for a shower, the primary fragment only
        # is marked as primary particle, so taking majority count would eliminate the shower
        # from primary particles for vertex identification purpose.

        # print("Normalize Vertex Label = ", self.normalize_vtx_label)

        if self.normalize_vtx_label:
            vtx_scatter_label = node_assn_vtx/self.spatial_size
            good_index_vtx = torch.all((0 <= vtx_scatter_label) & (vtx_scatter_label <= 1), dim=1)
        else:
            vtx_scatter_label = node_assn_vtx
            good_index_vtx = torch.all((0 <= vtx_scatter_label) & (vtx_scatter_label <= self.spatial_size), dim=1)

        vtx_scatter = nodes_vtx_reg

        vtx_score_acc = float(torch.sum(
            torch.argmax(node_pred_vtx[:, 3:], dim=1) == positives))

        if len(vtx_scatter):

            # for now only sum losses, they get averaged below in results dictionary
            loss2 = self.vtx_score_loss(node_pred_vtx[:, 3:],
                                        positives)
            loss1 = torch.sum(torch.mean(
                self.vtx_position_loss(vtx_scatter[good_index_vtx, :3],
                                       vtx_scatter_label[good_index_vtx]), dim=1))

            # print("Position Loss = ", loss1, vtx_scatter[good_index_vtx].shape)
            # print("Particle Primary Loss = ", loss2)

            loss = loss1 + loss2

            vtx_position_loss = float(loss1)
            vtx_score_loss = float(loss2)

            n_clusts_vtx += (good_index_vtx).sum().item()
            n_clusts_vtx_positives += (good_index_vtx & positives_vtx.bool()).sum().item()

        if vtx_scatter[good_index_vtx].shape[0]:
            # print(node_pred_vtx[good_index & positives.bool(), :3], node_assn_vtx[good_index & positives.bool()])
            vtx_position_acc = float(torch.sum(1. - torch.abs(vtx_scatter[good_index_vtx, :3] - \
                                                              vtx_scatter_label[good_index_vtx]) / \
                                    (torch.abs(vtx_scatter_label[good_index_vtx]) + \
                                     torch.abs(vtx_scatter[good_index_vtx, :3]))))/3.

        out =  (loss,
                vtx_position_loss,
                vtx_score_loss,
                n_clusts_vtx,
                n_clusts_vtx_positives,
                vtx_position_acc,
                vtx_score_acc)

        return out
