import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.modular_nnconv import *
from mlreco.models.gnn.cluster_cnn_encoder import *
from mlreco.models.layers.cnn_encoder import *
from mlreco.utils.gnn.cluster import *
from mlreco.utils.gnn.network import complete_graph
from .gnn import edge_model_construct, node_encoder_construct, edge_encoder_construct

class ParticleFlowModel(nn.Module):

    def __init__(self, cfg, name='particle_flow'):
        super(ParticleFlowModel, self).__init__()
        self.model_config = cfg[name]
        self.cnn_encoder = ClustCNNNodeEncoder2(cfg)
        self.edge_encoder = ClustCNNEdgeEncoder2(cfg)
        self.gnn = NNConvModel(cfg)

        self.node_type = self.model_config.get('node_type', 0)
        self.node_min_size = self.model_config.get('node_min_size', -1)
        self.source_col = self.model_config.get('source_col', 5)

    def forward(self, input):
        device = input[0].device

        if self.node_type > -1:
            mask = torch.nonzero(input[0][:,-1] == self.node_type).flatten()
            clusts = form_clusters(input[0][mask], self.node_min_size, self.source_col)
            clusts = [mask[c].cpu().numpy() for c in clusts]
        else:
            clusts = form_clusters(input[0], self.node_min_size, self.source_col)
            clusts = [c.cpu().numpy() for c in clusts]

        if not len(clusts):
            return {}
        x = self.cnn_encoder(input[0], clusts)
        batch_ids = get_cluster_batch(input[0], clusts)
        edge_index = complete_graph(batch_ids)
        e = self.edge_encoder(input[0], clusts, edge_index)
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device, dtype=torch.long)
        res = self.gnn(x, index, e, xbatch)

        node_pred = res['node_pred'][0]
        edge_pred = res['edge_pred'][0]

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        res = {'node_pred': [node_pred],
               'edge_pred': [edge_pred],
               'edge_index': [index],
               'clusts': [clusts]
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
        # self.node_loss = NodeChannelLoss(cfg)
        self.edge_loss = EdgeChannelLoss(cfg)

    def forward(self, result, cluster_labels, graph):
        loss = {}
        # node_loss = self.node_loss(result, cluster_labels)
        edge_loss = self.edge_loss(result, cluster_labels, graph)
        # loss.update(node_loss)
        loss.update(edge_loss)
        # loss['node_loss'] = node_loss['loss']
        loss['loss'] = edge_loss['loss']
        # loss['node_accuracy'] = node_loss['accuracy']
        loss['accuracy'] = edge_loss['accuracy']
        return loss