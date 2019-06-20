from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.layers.dbscan import DBScan, DBScanClusts
from mlreco.models.uresnet_ppn import PPNUResNet, SegmentationLoss
from mlreco.models.attention_gnn import BasicAttentionModel

# chain UResNet + PPN + DBSCAN + GNN for showers

class Chain(torch.nn.Module):
    """
    Chain of Networks
    1) UResNet - for voxel labels
    2) PPN - for particle locations
    3) DBSCAN - to form cluster
    4) GNN - to assign EM shower groups
    
    INPUT DATA:
        "input_data", "particles_label"
    """
    def __init__(self, model_config):
        super(Chain, self).__init__()
        self.dbscan = DBScanClusts(model_config['modules']['dbscan'])
        self.uresnet_ppn = PPNUResNet(model_config)
        if 'attention_gnn' in model_config['modules']:
            self.gnn = BasicAttentionModel(model_config)
        else:
            raise ValueError('No Valid GNN model provided')
        # self.keys = {'clusters': 5, 'segmentation': 3, 'points': 0}

    def forward(self, data):
        x = self.uresnet_ppn(data)
        #print(input[0].shape)
        #print(x[3][0].shape)
        new_input = torch.cat([data[0].double(), x[3][0].double()], dim=1)
        clusters = self.dbscan(new_input)
        return x, clusters


class ChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.loss = SegmentationLoss(cfg)

    def forward(self, segmentation, label, particles, clusters):
        # print(len(segmentation), len(segmentation[0]))
        # print(clusters[0].shape, label[0].shape)
        # assert len(segmentation[0]) == len(label)
        # assert len(particles) == len(label)
        # batch_ids = [d[:, -2] for d in label]
        # for i in range(len(label)):
        #     event_particles = particles[i]
        #     for b in batch_ids[i].unique():
        #         batch_index = batch_ids[i] == b
        #         event_data = label[i][batch_index][:, :-2]  # (N, 3)
        return self.loss(segmentation, label, particles)