# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from mlreco.models.grappa import GNN, GNNLoss
from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.utils.deghosting import adapt_labels


class GhostClustFullGNN(torch.nn.Module):
    """
    Deghosting + GNN clustering class

    If DBSCAN is used, use the semantic label tensor (last column) as an input.

    For use in config:
    model:
      name: ghost_cluster_full_gnn
      modules:
        uresnet_lonely:
          (...)
        chain:
          (...)
    See ClustFullGNN in mlreco/models/cluster_full_gnn.py for the rest of the config modules.
    """

    MODULES = ['chain', 'dbscan', 'node_encoder', 'edge_encoder', 'node_model', 'edge_model', 'uresnet_lonely']

    def __init__(self, cfg):
        super(GhostClustFullGNN, self).__init__()
        self.chain = GNN(cfg['grappa'])
        self.uresnet_lonely = UResNet(cfg)
        self.features = cfg['uresnet_lonely'].get('features', 1)

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            data ([torch.tensor]): (N,5-6) [x, y, z, batchid, (value,) id]
        Returns:
            dict:
                'node_pred' (torch.tensor): (N,2) Two-channel node predictions
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
        """
        point_cloud = data[0]
        result1 = self.uresnet_lonely((point_cloud,))
        # Extract predicted non-ghost points
        # Add semantic class in last column (expected by ClustFullGNN)
        # + Keep only 1 data feature
        predicted_nonghost = (result1['ghost'][0].argmax(dim=1) == 0)
        new_point_cloud = point_cloud[predicted_nonghost]
        new_point_cloud = torch.cat([new_point_cloud[:, :-self.features], result1['segmentation'][0][predicted_nonghost].argmax(dim=1).double().view((-1, 1))], dim=1)

        new_input = (new_point_cloud,)
        if len(data) > 1:
            new_input += (data[1],)
        result2 = self.chain(new_input)

        result = {}
        result.update(result1)
        result.update(result2)
        return result


class ChainLoss(torch.nn.modules.loss._Loss):
    """
    Uses SegmentationLoss and GNNLoss
    """
    def __init__(self, cfg, name='chain'):
        super(ChainLoss, self).__init__()
        self.gnn_loss = GNNLoss(cfg['grappa_loss'])
        self.uresnet_loss = SegmentationLoss(cfg)
        self._num_classes = cfg['uresnet_lonely'].get('num_classes', 5)

    def forward(self, result, clust_label, seg_label):
        uresnet_res = self.uresnet_loss(result, seg_label)
        # Make adapted labels to include ghost points
        print(seg_label[0].shape, (seg_label[0][:, -1] < 5).sum(), clust_label[0].shape, result['segmentation'][0].shape, (result['ghost'][0].argmax(dim=1) == 0).sum())

        clust_label = adapt_labels(result, seg_label, clust_label)
        seg_label = [seg_label[i][result['ghost'][i].argmax(dim=1) == 0] for i in range(len(seg_label))]
        # Now we can apply the GNN loss without risking size mismatches
        gnn_res = self.gnn_loss(result, clust_label)

        result = {}
        for key in uresnet_res:
            result['uresnet_' + key] = uresnet_res[key]
        for key in gnn_res:
            result['gnn_' + key] = gnn_res[key]
        result['loss'] = uresnet_res['loss'] + gnn_res['loss']
        result['accuracy'] = gnn_res['accuracy']
        return result
