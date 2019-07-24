from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.layers.dbscan import DBScan, DBScanClusts
from mlreco.models.uresnet_ppn import PPNUResNet, SegmentationLoss
# from mlreco.models.attention_gnn import BasicAttentionModel
from mlreco.utils.ppn import uresnet_ppn_point_selector

# chain UResNet + PPN + DBSCAN + GNN for showers

class Chain(torch.nn.Module):
    """
    Chain of Networks
    1) UResNet - for voxel labels
    2) PPN - for particle locations
    3) DBSCAN - to form cluster
    4) GNN - to assign EM shower groups

    INPUT DATA:
        just energy deposision data
        "input_data": ["parse_sparse3d_scn", "sparse3d_data"],
    """
    MODULES = ['dbscan', 'uresnet_ppn', 'attention_gnn']

    def __init__(self, model_config):
        pass
#         super(Chain, self).__init__()
#         self.dbscan = DBScanClusts(model_config)
#         self.uresnet_ppn = PPNUResNet(model_config)
#         if 'attention_gnn' in model_config['modules']:
#             self.gnn = BasicAttentionModel(model_config)
#         else:
#             raise ValueError('No Valid GNN model provided')
        # self.keys = {'clusters': 5, 'segmentation': 3, 'points': 0}

    def forward(self, data):
        pass
#         x = self.uresnet_ppn(data)
#         onehot_types = torch.argmax(x[3][0], 1).view(-1,1)
#         # get predicted 5-types data
#         pred_types = torch.cat([data[0][:,:4].double(), onehot_types.view(-1,1).double()], dim=1)
#         # cluster on 5-types data
#         clusters = self.dbscan(pred_types, onehot=False)

#         # point selector from uresnet+ppn
#         ppn_pts = uresnet_ppn_point_selector(pred_types, x)
#         em_sel = ppn_pts[:,-1] > 1 # select em points

#         # pass into gnn
#         # gnn_data = [five_types_data, particle_data]
#         gnn_data = [pred_types, torch.tensor(ppn_pts[em_sel], dtype=torch.float)]

#         gnn_out = self.gnn(gnn_data)

#         return x, clusters, ppn_pts, gnn_data, gnn_out


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
