from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from .layers.dbscan import DBScan
from .uresnet_ppn import PPNUResNet, SegmentationLoss


class Chain(torch.nn.Module):
    """
    Extracts tracks clusters
    """
    def __init__(self, model_config):
        super(Chain, self).__init__()
        self.dbscan = DBScan(model_config['modules']['dbscan'])
        self.uresnet_ppn = PPNUResNet(model_config['modules']['uresnet_ppn'])
        # self.keys = {'clusters': 5, 'segmentation': 3, 'points': 0}

    def forward(self, input):
        x = self.uresnet_ppn(input)
        #print(input[0].shape)
        #print(x[3][0].shape)
        new_input = torch.cat([input[0].double(), x[3][0].double()], dim=1)
        #print(new_input[:10])
        clusters = self.dbscan(new_input)
        #c = torch.cat(clusters, dim=0)
        final = []
        i = 0
        for cluster in clusters:
            if cluster[0, -1] == 0 or cluster[0, -1] == 1:
                cluster = torch.nn.functional.pad(cluster, (0, 1, 0, 0), mode='constant', value=i)
                final.append(cluster)
                i += 1
        print(len(clusters), len(final))
        if len(final) > 0:
            final = torch.cat(final, dim=0)
        return x + [[final]]


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
