from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
#from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.clustercnn_se import ClusterCNN, ClusteringLoss


class GhostChain(torch.nn.Module):
    """
    Run UResNet and use its encoding/decoding feature maps for PPN layers
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (float,), (3, 1)],
    #     ["parse_particle_points", (int, int), (3, 2)]
    # ]
    # MODULES = ['ppn', 'uresnet_lonely']

    def __init__(self, model_config):
        super(GhostChain, self).__init__()
        self.uresnet_lonely = UResNet(model_config)
        self.spatial_embeddings = ClusterCNN(model_config)
        # self._freeze_uresnet = model_config['uresnet_lonely'].get('freeze', False)
        #
        # if self._freeze_uresnet:
        #     for param in self.uresnet_lonely.parameters():
        #         param.requires_grad = False

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        """
        point_cloud = input[0]
        result1 = self.uresnet_lonely((point_cloud,))
        #print((result1['ghost'][0].argmax(dim=1) == 1).sum(), (result1['ghost'][0].argmax(dim=1) == 0).sum())
        new_point_cloud = point_cloud[result1['ghost'][0].argmax(dim=1) == 0]
        #print(new_point_cloud.size())
        result2 = self.spatial_embeddings((new_point_cloud,))
        result = {}
        result.update(result1)
        result.update(result2)
        result['batch_idx'] = [new_point_cloud[:, 3]]
        return result


class GhostChainLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(GhostChainLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.clustering_loss = ClusteringLoss(cfg)
        self._num_classes = cfg['uresnet_lonely']['num_classes']

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, result, label_seg, label_clustering):
        #print("label_seg ", label_seg[0].size(), "label_clustering ", label_clustering[0].size())
        #print("no ghost", (label_seg[0][:, -1] < 5).sum())
        #print((label_seg[0][label_seg[0][:, -1] < 5][:, :4] != label_clustering[0][:, :4]).sum())
        uresnet_res = self.uresnet_loss(result, label_seg)
        # label_pred = result['segmentation'][0][result['ghost'][0].argmax(dim=1) == 1].argmax(dim=1)
        # print(label_pred.size())
        complete_label_clustering = []
        for i in range(len(label_seg)):
            coords = label_seg[i][:, :4]
            label_c = []
            for batch_id in coords[:, -1].unique():
                batch_mask = coords[:, -1] == batch_id
                batch_coords = coords[batch_mask]
                batch_clustering = label_clustering[i][label_clustering[0][:, 3] == batch_id]
                nonghost_mask = (result['ghost'][i][batch_mask].argmax(dim=1) == 0)
                # Select voxels predicted as nonghost, but true ghosts
                mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == self._num_classes)
                #true_mask = (batch_clustering[:, -1] < self._num_classes)
                # Assign them to closest cluster
                #print("mask", mask.size(), "true mask", true_mask.size(), "coords", coords.size())
                d = self.distances(batch_coords[mask, :3], batch_clustering[:, :3]).argmin(dim=1)
                #print("d", d.size(), "batch_clustering", batch_clustering.size(), "batch_coords[mask]", batch_coords[mask].size())
                #print("select", batch_clustering[d, 4:].size())
                additional_label_clustering = torch.cat([batch_coords[mask], batch_clustering[d, 4:]], dim=1)
                new_label_clustering = -1. * torch.ones((batch_coords.size(0), batch_clustering.size(1))).double()
                if torch.cuda.is_available():
                    new_label_clustering = new_label_clustering.cuda()
                #print(new_label_clustering.type(), additional_label_clustering.type())
                new_label_clustering[mask] = additional_label_clustering
                new_label_clustering[label_seg[i][batch_mask, -1] < self._num_classes] = batch_clustering
                #print(new_label_clustering.unique())
                #new_label_clustering = torch.cat([batch_clustering, additional_label_clustering], dim=0)
                label_c.append(new_label_clustering[nonghost_mask])
            label_c = torch.cat(label_c, dim=0)
            complete_label_clustering.append(label_c)
        #print("label_c", label_c.size(), "embeddings", result['embeddings'][0].size())
        clustering_res = self.clustering_loss(result, complete_label_clustering)

        result = {}
        result.update(uresnet_res)
        result.update(clustering_res)

        # Don't forget to sum all losses
        result['uresnet_loss'] = uresnet_res['loss'].float()
        result['uresnet_accuracy'] = uresnet_res['accuracy']
        result['clustering_loss'] = clustering_res['loss'].float()
        result['clustering_accuracy'] = clustering_res['accuracy']
        result['loss'] = clustering_res['loss'].float() + uresnet_res['loss'].float()

        return result
