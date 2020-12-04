from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
#from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.clustercnn_se import ClusterCNN, ClusteringLoss
from mlreco.models.layers.dbscan import distances
from mlreco.utils.deghosting import adapt_labels


class GhostSpatialEmbeddings(torch.nn.Module):
    """
    Run UResNet deghosting + Spatial Embeddings (CNN clustering)
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (float,), (3, 1)],
    # ]
    MODULES = ['spatial_embeddings', 'uresnet_lonely'] + ClusterCNN.MODULES

    def __init__(self, model_config):
        super(GhostSpatialEmbeddings, self).__init__()
        self.uresnet_lonely = UResNet(model_config)
        self.spatial_embeddings = ClusterCNN(model_config)
        self.input_features = model_config['uresnet_lonely'].get('features', 1)
        # self._freeze_uresnet = model_config['uresnet_lonely'].get('freeze', False)
        #
        # if self._freeze_uresnet:
        #     for param in self.uresnet_lonely.parameters():
        #         param.requires_grad = False

    def forward(self, input):
        """
        Input can have several features, but only the 1st one will be passed
        to the CNN clustering step.
        """
        point_cloud = input[0]
        result1 = self.uresnet_lonely((point_cloud,))
        #print((result1['ghost'][0].argmax(dim=1) == 1).sum(), (result1['ghost'][0].argmax(dim=1) == 0).sum())
        new_point_cloud = point_cloud[result1['ghost'][0].argmax(dim=1) == 0]
        if self.input_features > 1:
            new_point_cloud = new_point_cloud[:, :-self.input_features+1]
        #print(new_point_cloud.size())
        result2 = self.spatial_embeddings((new_point_cloud,))
        result = {}
        result.update(result1)
        result.update(result2)
        #result['batch_idx'] = [new_point_cloud[:, 3]]
        return result


class GhostSpatialEmbeddingsLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + CNN spatial embeddings chain
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(GhostSpatialEmbeddingsLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.spice_loss = ClusteringLoss(cfg)
        self._num_classes = cfg['uresnet_lonely'].get('num_classes', 5)

    def forward(self, result, label_seg, label_clustering):
        uresnet_res = self.uresnet_loss(result, label_seg)
        complete_label_clustering = adapt_labels(result, label_seg, label_clustering)
        clustering_res = self.spice_loss(result, complete_label_clustering)

        result = {}
        result.update(uresnet_res)
        result.update(clustering_res)

        # Don't forget to sum all losses
        result['uresnet_loss'] = uresnet_res['loss'].float()
        result['uresnet_accuracy'] = uresnet_res['accuracy']
        result['spice_loss'] = clustering_res['loss'].float()
        result['clustering_accuracy'] = clustering_res['accuracy']
        result['loss'] = clustering_res['loss'].float() + uresnet_res['loss'].float()

        return result
