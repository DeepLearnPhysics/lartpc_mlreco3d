from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.layers.dbscan import DBScan2
# from mlreco.models.uresnet_ppn import PPNUResNet, SegmentationLoss
from mlreco.models.uresnet_ppn_type import PPNUResNet, SegmentationLoss


class Chain(torch.nn.Module):
    """
    Extracts tracks clusters
    """
    MODULES = ['dbscan', 'uresnet_ppn_type']
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,), (3, 1)],
        ["parse_particle_points", (int,), (3, 1)]
    ]

    def __init__(self, model_config):
        super(Chain, self).__init__()
        self.dbscan = DBScan2(model_config)
        self.uresnet_ppn_type = PPNUResNet(model_config)
        self._num_classes = model_config['modules']['uresnet_ppn_type'].get('num_classes', 5)
        # self.keys = {'clusters': 5, 'segmentation': 3, 'points': 0}

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        No multi-GPU! (We select index 0 of x['segmentation'])
        """
        x = self.uresnet_ppn_type(input)
        new_input = torch.cat([input[0].double(), x['segmentation'][0].double()], dim=1)
        one_hot = torch.cat([new_input[:, :-5], torch.nn.functional.one_hot(torch.argmax(new_input[:, -self._num_classes:], dim=1), num_classes=self._num_classes).double()], dim=1)
        clusters = self.dbscan(one_hot)
        #c = torch.cat(clusters, dim=0)
        final = []
        i = 0
        # print("clusters", clusters[:10])
        for cluster in clusters:
            if cluster[0, -1] == 0 or cluster[0, -1] == 1:
                cluster = torch.nn.functional.pad(cluster, (0, 1, 0, 0), mode='constant', value=i)
                final.append(cluster)
                i += 1
        # print(len(clusters), len(final))
        if len(final) > 0:
            final = torch.cat(final, dim=0)
        # print(len(x))
        x['final'] = [final]
        return x


class ChainLoss(torch.nn.modules.loss._Loss):
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (int,), (3, 1)],
        ["parse_particle_points", (int,), (3, 1)],
        ["parse_cluster3d_clean", (int,), (3, 1)]
    ]

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
