import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

# from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.gnn.losses.edge_channel import EdgeChannelLoss

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.message_passing.nnconv import *
from mlreco.models.gnn.encoders.cnn import *
from mlreco.models.layers.cnn_encoder import *
from mlreco.utils.gnn.cluster import *
from mlreco.utils.gnn.network import complete_graph
# from torch_geometric.utils import subgraph

from collections import defaultdict, Counter

from mlreco.models.gnn.normalizations import BatchNorm
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool

import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat

from mlreco.models.cluster_cnn.losses.lovasz import lovasz_softmax_flat


def get_edge_features(nodes, batch_idx, edge_net):
    '''
    Compile Fully Connected Edge Features from nodes and batch indices.

    INPUTS:
        - nodes (N x d Tensor): list of node features
        - batch_idx (N x 1 Tensor): list of batch indices for nodes
        - edge_net: nn.Module that taks two vectors and returns edge feature vector.

    RETURNS:
        - edge_features: list of edges features
        - edge_indices: list of edge indices (i->j)
        - edge_batch_indices: list of batch indices (0 to B)
    '''
    unique_batch = batch_idx.unique()
    edge_index = []
    edge_features = []
    for bidx in unique_batch:
        mask = bidx == batch_idx
        clust_ids = torch.nonzero(mask).flatten()
        nodes_batch = nodes[mask]
        subindex = torch.arange(nodes_batch.shape[0])
        N = nodes_batch.shape[0]
        for i, row in enumerate(nodes_batch):
            submask = subindex != i
            edge_idx = [[clust_ids[i].item(), clust_ids[j].item()] for j in subindex[submask]]
            edge_index.extend(edge_idx)
            others = nodes_batch[submask]
            ei2j = edge_net(row.expand_as(others), others)
            edge_features.extend(ei2j)

    edge_index = np.vstack(edge_index)
    edge_features = torch.stack(edge_features, dim=0)

    return edge_index.T, edge_features


class EdgeFeatureNet(nn.Module):
    '''
    Small MLP for extracting input edge features from two node features.

    USAGE:
        net = EdgeFeatureNet(16, 16)
        node_x = torch.randn(16, 5)
        node_y = torch.randn(16, 5)
        edge_feature_x2y = net(node_x, node_y) # (16, 5)
    '''
    def __init__(self, num_input, num_output, num_hidden=128):
        super(EdgeFeatureNet, self).__init__()
        self.linear1 = nn.Linear(num_input * 2, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.ELU()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.linear1(x)
        if x.shape[0] > 1:
            x = self.elu(self.norm1(x))
        x = self.linear2(x)
        if x.shape[0] > 1:
            x = self.elu(self.norm2(x))
        x = self.linear3(x)
        return x


class MomentumNet(nn.Module):
    '''
    Small MLP for extracting input edge features from two node features.

    USAGE:
        net = EdgeFeatureNet(16, 16)
        node_x = torch.randn(16, 5)
        node_y = torch.randn(16, 5)
        edge_feature_x2y = net(node_x, node_y) # (16, 5)
    '''
    def __init__(self, num_input, num_output=3, num_hidden=128):
        super(MomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.ELU()

    def forward(self, x):
        if x.shape[0] > 1:
            self.norm1(x)
        x = self.linear1(x)
        x = self.elu(x)
        if x.shape[0] > 1:
            x = self.norm2(x)
        x = self.linear2(x)
        x = self.elu(x)
        x = self.linear3(x)
        x = x / torch.norm(x, dim=1).view(-1, 1)
        return x


class GraphEncoder(nn.Module):
    '''
    Graph Encoder Module, using the first half of GraphUNet implementation in
    Pytorch Geometric.
    '''
    def __init__(self, in_channels, hidden_channels, out_channels,
                 depth, pool_ratio=0.5, sum_res=True, act=F.elu):
        super(GraphEncoder, self).__init__()
        assert depth > 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratio = pool_ratio
        self.act = act
        self.sum_res = sum_res
        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.norms.append(BatchNorm(in_channels))
        self.down_convs.append(GCNConv(in_channels, channels[0], improved=True))
        for i in range(1, depth):
            self.pools.append(TopKPooling(channels[i-1], self.pool_ratio))
            self.norms.append(BatchNorm(channels[i-1]))
            self.down_convs.append(GCNConv(channels[i-1], channels[i], improved=True))

        self.reset_parameters()


    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.norms[0](x)
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]

        for i in range(1, self.depth):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch=batch)
            x = self.norms[i](x)
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        x = global_mean_pool(x, batch)

        return x


from torch_geometric.utils.num_nodes import maybe_num_nodes

def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False,
             num_nodes=None):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset
        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        n_mask[subset.to(dtype=torch.bool)] = 1
        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    # print(edge_index[0])
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr


class ParticleFlowModel(nn.Module):
    '''
    Particle Flow Prediction and Particle Classification using induced subgraph
    encoder module.
    '''

    MODULES = [('node_encoder', {'network_base': {}, 'uresnet_encoder': {}, 'res_encoder': {}}),
                'edge_model', 'particle_flow', 'subgraph_loss', 'chain']

    def __init__(self, cfg, name='particle_flow'):
        super(ParticleFlowModel, self).__init__()
        self.model_config = cfg[name]
        self.cnn_encoder = ClustCNNNodeEncoder2(cfg['node_encoder'])
        self.gnn = NNConvModel(cfg['edge_model'])

        self.num_node_features = self.model_config.get('num_node_features', 32)
        self.num_edge_features = self.model_config.get('num_edge_features', 32)
        self.node_type = self.model_config.get('node_type', -1)
        self.node_min_size = self.model_config.get('node_min_size', -1)
        self.source_col = self.model_config.get('source_col', 5)
        self.edge_net = EdgeFeatureNet(
            self.num_node_features, self.num_edge_features)
        self.graph_encoder = GraphEncoder(128, [256, 512, 1024], 512, 3)
        self.particle_pred = nn.Linear(1024, 5)


    def forward(self, input):
        device = input[0].device

        if self.node_type > -1:
            mask = torch.nonzero(input[0][:,-1] == self.node_type).flatten()
            clusts = form_clusters(input[0][mask], self.node_min_size, self.source_col)
            groups = form_clusters(input[0][mask], self.node_min_size, 6)
            clusts = [mask[c].cpu().numpy() for c in clusts]
            groups = [mask[c].cpu().numpy() for c in groups]
        else:
            clusts = form_clusters(input[0], self.node_min_size, self.source_col)
            clusts = [c.cpu().numpy() for c in clusts]
            groups = form_clusters(input[0], self.node_min_size, 6)
            groups = [c.cpu().numpy() for c in groups]

        if not len(clusts):
            return {}
        x = self.cnn_encoder(input[0], clusts)
        # print(x.shape)
        batch_ids = get_cluster_batch(input[0], clusts)
        # fragment_ids = get_cluster_label(input[0], clusts, column=5)
        # print("Fragments = ", fragment_ids, fragment_ids.shape)
        group_ids = get_cluster_label(input[0], clusts, column=6)
        # print("Groups = ", group_ids, group_ids.shape)
        xbatch = torch.tensor(batch_ids, device=device, dtype=torch.long)
        # edge_index = complete_graph(batch_ids)
        # e = self.edge_encoder(input[0], clusts, edge_index)
        edge_index, e = get_edge_features(x, xbatch, self.edge_net)
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        res = self.gnn(x, index, e, xbatch)

        node_pred = res['node_pred'][0]
        edge_pred = res['edge_pred'][0]
        edge_assn = torch.argmax(edge_pred, dim=1).detach().cpu().numpy().astype(bool)

        pruned_index = edge_index.T
        pruned_index = torch.from_numpy(pruned_index[edge_assn].T).to(device, dtype=torch.long)
        pruned_edge_features = res['edge_features'][0]
        edge_assn = torch.argmax(edge_pred, dim=1).to(dtype=torch.bool)
        pruned_edge_features = pruned_edge_features[edge_assn]
        node_features = res['node_features'][0]

        pgraph_xbatch = []
        pgraph_nodes = []
        pgraph_edges = []
        pgraph_eindex = []

        xbatch_count = 0

        # print(node_pred.shape)

        for b in np.unique(batch_ids):
            group_batch = group_ids[batch_ids == b]
            for g in np.unique(group_batch):
                particle_mask = np.logical_and(batch_ids == b, group_ids == g)
                particle_mask = torch.from_numpy(particle_mask).to(device, dtype=torch.bool)
                sg_nodes = node_features[particle_mask]
                # print(torch.sum(particle_mask))
                # print(sg_nodes.shape)
                # print(torch.sum(particle_mask))
                # if torch.sum(particle_mask) > 3:
                sg_index, sg_attr = subgraph(particle_mask,
                    pruned_index,
                    edge_attr=pruned_edge_features)
                    # print(sg_idnex, sg_attr)
                # print(torch.sum(particle_mask).item(), sg_index, sg_attr.shape)
                pgraph_xbatch.append(torch.zeros(sg_nodes.shape[0]).to(device, dtype=torch.long) + xbatch_count)
                pgraph_nodes.append(sg_nodes)
                pgraph_edges.append(sg_attr)
                pgraph_eindex.append(sg_index)
                xbatch_count += 1

        pgraph_xbatch = torch.cat(pgraph_xbatch)
        pgraph_nodes = torch.cat(pgraph_nodes, dim=0)
        pgraph_edges = torch.cat(pgraph_edges, dim=0)
        pgraph_eindex = torch.cat(pgraph_eindex, dim=1)

        # print(xbatch, xbatch.shape)
        # print(pgraph_xbatch, pgraph_xbatch.shape)
        # print(pgraph_nodes, pgraph_nodes.shape)
        # print(pgraph_edges, pgraph_edges.shape)
        # print(pgraph_eindex, pgraph_eindex.shape)

        particles = self.graph_encoder(pgraph_nodes, pgraph_eindex, batch=pgraph_xbatch)
        particles_pred = self.particle_pred(particles)

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]
        # print(beids)

        node_pred = [node_pred[b] for b in bcids]
        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        res = {'node_pred': [node_pred],
               'edge_pred': [edge_pred],
               'edge_index': [edge_index],
            #    'momenta_pred': [momenta_pred],
               'clusts': [clusts],
               'particles_pred': [particles_pred],
               'groups': [groups]
        }

        return res


class ChainLoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of ClustHierarchyGNN and computes the total loss
    coming from the edge model and the node model.

    For use in config:
    model:
      name: cluster_hierachy_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target_photons  : <use true photon connections as basis for loss (default False)>
    """
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.type_loss = SubgraphLoss(cfg)
        self.edge_loss = EdgeChannelLoss(cfg)

    def forward(self, result, cluster_labels, graph, kinematics):
        loss = {}
        type_loss = self.type_loss(result, kinematics)
        edge_loss = self.edge_loss(result, cluster_labels, graph)
        loss.update(type_loss)
        # print(edge_loss['loss'])
        loss.update(edge_loss)
        print("Type Loss: {:.4f}, Accuracy = {:.4f}".format(loss['type_loss'], loss['type_accuracy']))
        # print("Momentum Loss: {:.4f}, Accuracy = {:.4f}".format(loss['loss_momenta'], loss['acc_momenta']))
        print("Edge Loss: {:.4f}, Accuracy = {:.4f}".format(edge_loss['loss'], edge_loss['accuracy']))
        loss['loss'] = edge_loss['loss']
        # print(edge_loss['loss'])
        loss['loss'] += loss['type_loss']
        # loss['node_accuracy'] = node_loss['accuracy']
        loss['accuracy'] = (edge_loss['accuracy'] + type_loss['type_accuracy']) / 2

        return loss


class SubgraphLoss(nn.Module):
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
    def __init__(self, cfg, name='subgraph_loss'):
        super(SubgraphLoss, self).__init__()

        # Get the chain input parameters
        chain_config = cfg[name]
        # print(chain_config)
        # Set the loss
        self.loss = chain_config.get('type_loss', 'CE')
        self.reduction = chain_config.get('reduction', 'sum')
        self.type_weight = chain_config.get('type_weight', 1.0)
        self.momentum_weight = chain_config.get('momentum_weight', 1.0)
        self.balance_classes = chain_config.get('balance_classes', False)
        # self.high_purity = chain_config.get('high_purity', False)

        if self.loss == 'CE':
            # self.ce_weight = torch.Tensor([2, 1, 1.5, 5, 3])
            self.ce_weight = torch.Tensor([1, 1, 1, 1, 1])
            self.lossfn = torch.nn.CrossEntropyLoss(weight=self.ce_weight, reduction=self.reduction)
        elif self.loss == 'lovasz-softmax':
            self.softmax = nn.Softmax(dim=1)
            def lossfn(logits, labels):
                probs = self.softmax(logits)
                loss = lovasz_softmax_flat(probs, labels)
                return loss
            self.lossfn = lossfn
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, kinematics):
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
        # print(kinematics[0][:, 4:])
        total_loss, total_acc = 0., 0.
        loss_type, acc_type = 0., 0.
        loss_momenta, acc_momenta = 0., 0.
        acc_per_type = defaultdict(list)
        n_clusts = 0

        for i in range(len(kinematics)):

            # If the input did not have any node, proceed
            if 'node_pred' not in out:
                continue

            # true_labels = []
            # # Get the list of batch ids, loop over individual batches
            # batches = kinematics[i][:,3]
            # group_ids = kinematics[i][:, 6]
            # nbatches = len(batches.unique())
            # for b in range(nbatches):
            #     batch_mask = batches == b
            #     # print(batch_mask)
            #     group_batch = kinematics[i][batch_mask][:, 6]
            #     for g in group_batch.unique():
            #         particle_mask = batch_mask & (group_ids == g)
            #         # print(particle_mask)
            #         vals, counts = kinematics[i][particle_mask][:, 11].to(
            #             dtype=torch.long).unique(return_counts=True)
            #         if len(vals) > 1:
            #             print("Two PDGs in one group: {}".format(str(vals)))
            #         idx = torch.argmax(counts)
            #         true_labels.append(vals[idx].item())
            #         # particle_mask = torch.from_numpy(particle_mask).to(device, dtype=torch.bool)
            #         # print(kinematics[i][particle_mask][:, 11].to(dtype=torch.long).unique())
            #         # print(kinematics[i][batch_mask][group_batch == g][:, 11].to(dtype=torch.long).unique())
            # print(np.asarray(true_labels))
            groups = out['groups'][i]
            ngroups = len(groups)
            pdg_ids = get_cluster_label(kinematics[i], groups, column=11)
            particles_logits = out['particles_pred'][i]
            print(particles_logits)
            # print(particles_logits.shape)
            print(pdg_ids, pdg_ids.shape)
            # print(Counter(pdg_ids))

            # If the majority cluster ID agrees with the majority group ID, assign as primary
            node_assn = torch.tensor(pdg_ids,
                dtype=torch.long,
                device=particles_logits.device,
                requires_grad=False)

            # Increment the loss, balance classes if requested
            if self.balance_classes:
                vals, counts = torch.unique(node_assn, return_counts=True)
                weights = np.array(
                    [float(counts[k])/len(node_assn) for k in range(len(vals))])
                for k, v in enumerate(vals):
                    loss_type += (1./weights[k])*self.lossfn(
                        particles_logits[node_assn==v], node_assn[node_assn==v])
            else:
                loss_type += self.lossfn(particles_logits, node_assn)
            # cs = self.cosine_sim(momenta_pred, momenta_true)
            # loss_momenta += torch.sum(1.0 - cs)
            # print(torch.max(particles_logits, dim=1))
            print(torch.argmax(particles_logits, dim=1).detach().cpu().numpy())
            # Compute accuracy of assignment (fraction of correctly assigned nodes)
            acc_type += float(torch.sum(
                torch.argmax(particles_logits, dim=1) == node_assn))

            for pdg in node_assn.unique():
                correct = torch.argmax(particles_logits, dim=1) == node_assn
                acc = float(correct[node_assn == pdg].sum()) / \
                    float(torch.sum(node_assn == pdg))
                acc_per_type['acc_type_{}'.format(int(pdg))].append(acc)

            n_clusts += ngroups

        acc_log = {}

        for pdg, lvals in acc_per_type.items():
            acc_log[pdg] = sum(lvals) / len(lvals)

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            return {
                'type_accuracy': 0.,
                'type_loss': torch.tensor(0., requires_grad=True,
                    device=clusters[0].device),
                'n_clusts': n_clusts,
                'acc_type': 0.,
                # 'acc_momenta': 0.,
                'acc_type_0': 0.,
                'acc_type_1': 0.,
                'acc_type_2': 0.,
                'acc_type_3': 0.,
                'acc_type_4': 0.
            }

        loss_type /= n_clusts
        # loss_momenta /= n_clusts
        total_loss = self.type_weight * loss_type
        # total_loss = (self.type_weight * loss_type + self.momentum_weight * loss_momenta) / 2
        acc_type /= n_clusts
        # acc_momenta /= n_clusts
        total_acc = acc_type
        # total_acc = (acc_type + acc_momenta) / 2

        res = {
            'type_accuracy': total_acc,
            'type_loss': total_loss,
            'n_clusts': n_clusts,
            'acc_type': acc_type,
            # 'acc_momenta': acc_momenta,
            'loss_type': self.type_weight * float(loss_type)
            # 'loss_momenta': self.momentum_weight * float(loss_momenta)
        }

        res.update(acc_log)

        TYPE_LABELS = {
            'acc_type_0': 'photon',  # photon
            'acc_type_1': 'e',  # e-
            'acc_type_2': 'mu', # e+
            'acc_type_3': 'pi',  # mu-
            'acc_type_4': 'p', # mu+
        }

        for key, val in acc_log.items():
            print("{}: {:.4f}".format(TYPE_LABELS[key], val))

        return res


# class FullNodeLoss(torch.nn.Module):
#     """
#     Takes the output of EdgeModel and computes the channel-loss.

#     For use in config:
#     model:
#       name: cluster_gnn
#       modules:
#         chain:
#           loss            : <loss function: 'CE' or 'MM' (default 'CE')>
#           reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
#           balance_classes : <balance loss per class: True or False (default False)>
#           high_purity     : <only penalize loss on groups with a primary (default False)>
#     """
#     def __init__(self, cfg, name='full_node_loss'):
#         super(FullNodeLoss, self).__init__()

#         # Get the chain input parameters
#         chain_config = cfg[name]
#         # print(chain_config)
#         # Set the loss
#         self.loss = chain_config.get('type_loss', 'CE')
#         self.reduction = chain_config.get('reduction', 'sum')
#         self.type_weight = chain_config.get('type_weight', 1.0)
#         self.momentum_weight = chain_config.get('momentum_weight', 1.0)
#         self.balance_classes = chain_config.get('balance_classes', False)
#         # self.high_purity = chain_config.get('high_purity', False)

#         if self.loss == 'CE':
#             self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
#         elif self.loss == 'MM':
#             p = chain_config.get('p', 1)
#             margin = chain_config.get('margin', 1.0)
#             self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
#         else:
#             raise Exception('Loss not recognized: ' + self.loss)

#         self.cosine_sim = torch.nn.CosineSimilarity()

#     def forward(self, out, kinematics):
#         """
#         Applies the requested loss on the node prediction.

#         Args:
#             out (dict):
#                 'node_pred' (torch.tensor): (C,2) Two-channel node predictions
#                 'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
#             clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
#         Returns:
#             double: loss, accuracy, clustering metrics
#         """
#         # print(kinematics[0][:, 4:])
#         total_loss, total_acc = 0., 0.
#         loss_type, loss_momenta = 0., 0.
#         acc_type, acc_momenta = 0., 0.
#         acc_per_type = defaultdict(list)
#         n_clusts = 0
#         for i in range(len(kinematics)):

#             # If the input did not have any node, proceed
#             if 'node_pred' not in out:
#                 continue

#             # Get the list of batch ids, loop over individual batches
#             batches = kinematics[i][:,3]
#             nbatches = len(batches.unique())
#             for j in range(nbatches):

#                 # Narrow down the tensor to the rows in the batch
#                 labels = kinematics[i][batches==j]

#                 # Use the primary information to determine the true node assignment
#                 node_pred = out['node_pred'][i][j]
#                 # momenta_pred = out['momenta_pred'][i][j]

#                 if not node_pred.shape[0]:
#                     continue
#                 clusts = out['clusts'][i][j]
#                 clust_ids = get_cluster_label(labels, clusts)
#                 group_ids = get_cluster_label(labels, clusts, column=6)
#                 pdg_ids = get_cluster_label(labels, clusts, column=11)
#                 # print(group_ids)
#                 # momenta_true = get_momenta_labels(labels, clusts, columns=[7, 8, 9])

#                 # if self.high_purity:
#                 #     purity_mask = np.zeros(len(clusts), dtype=bool)
#                 #     for g in np.unique(pdg_ids):
#                 #         group_mask = pdg_ids == g
#                 #         if np.sum(group_mask) > 1 and g in clust_ids[group_mask]:
#                 #             purity_mask[group_mask] = np.ones(np.sum(group_mask))
#                 #     clusts    = clusts[purity_mask]
#                 #     clust_ids = clust_ids[purity_mask]
#                 #     pdg_ids = pdg_ids[purity_mask]
#                 #     node_pred = node_pred[np.where(purity_mask)[0]]
#                     # if not len(clusts):
#                     #     continue

#                 # If the majority cluster ID agrees with the majority group ID, assign as primary
#                 node_assn = torch.tensor(pdg_ids, dtype=torch.long, device=node_pred.device, requires_grad=False)

#                 # Increment the loss, balance classes if requested
#                 if self.balance_classes:
#                     vals, counts = torch.unique(node_assn, return_counts=True)
#                     weights = np.array([float(counts[k])/len(node_assn) for k in range(len(vals))])
#                     for k, v in enumerate(vals):
#                         loss_type += (1./weights[k])*self.lossfn(node_pred[node_assn==v], node_assn[node_assn==v])
#                 else:
#                     loss_type += self.lossfn(node_pred, node_assn)
#                 # cs = self.cosine_sim(momenta_pred, momenta_true)
#                 # loss_momenta += torch.sum(1.0 - cs)

#                 # Compute accuracy of assignment (fraction of correctly assigned nodes)
#                 acc_type += float(torch.sum(torch.argmax(node_pred, dim=1) == node_assn))

#                 for pdg in node_assn.unique():
#                     correct = torch.argmax(node_pred, dim=1) == node_assn
#                     acc = float(correct[node_assn == pdg].sum()) / float(torch.sum(node_assn == pdg))
#                     acc_per_type['acc_type_{}'.format(int(pdg))].append(acc)
#                 # with torch.no_grad():
#                     # acc_momenta += float(torch.sum((1.0 + cs) / 2))

#                 # Increment the number of events
#                 n_clusts += len(clusts)

#         acc_log = {}

#         for pdg, lvals in acc_per_type.items():
#             acc_log[pdg] = sum(lvals) / len(lvals)

#         # Handle the case where no cluster/edge were found
#         if not n_clusts:
#             return {
#                 'node_accuracy': 0.,
#                 'node_loss': torch.tensor(0., requires_grad=True, device=clusters[0].device),
#                 'n_clusts': n_clusts,
#                 'acc_type': 0.,
#                 'loss_type': 0.,
#                 # 'acc_momenta': 0.,
#                 'acc_type_0': 0.,
#                 'acc_type_1': 0.,
#                 'acc_type_2': 0.,
#                 'acc_type_3': 0.,
#                 'acc_type_4': 0.
#             }

#         loss_type /= n_clusts
#         # loss_momenta /= n_clusts
#         total_loss = self.type_weight * loss_type
#         # total_loss = (self.type_weight * loss_type + self.momentum_weight * loss_momenta) / 2
#         acc_type /= n_clusts
#         # acc_momenta /= n_clusts
#         total_acc = acc_type
#         # total_acc = (acc_type + acc_momenta) / 2

#         res = {
#             'node_accuracy': total_acc,
#             'node_loss': total_loss,
#             'n_clusts': n_clusts,
#             'acc_type': acc_type,
#             # 'acc_momenta': acc_momenta,
#             'loss_type': self.type_weight * float(loss_type)
#             # 'loss_momenta': self.momentum_weight * float(loss_momenta)
#         }

#         res.update(acc_log)

#         TYPE_LABELS = {
#             'acc_type_0': 'photon',  # photon
#             'acc_type_1': 'e',  # e-
#             'acc_type_2': 'mu', # e+
#             'acc_type_3': 'pi',  # mu-
#             'acc_type_4': 'p', # mu+
#         }

#         for key, val in acc_log.items():
#             print("{}: {:.4f}".format(TYPE_LABELS[key], val))

#         return res
