
import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from mlreco.models.uresnet import SegmentationLoss
from collections import defaultdict
from mlreco.models.uresnet import UResNet_Chain
from mlreco.models.layers.common.vertex_ppn import VertexPPN, VertexPPNLoss
from mlreco.models.experimental.layers.pointnet import PointNetEncoder

from mlreco.utils.gnn.data import split_clusts
from mlreco.utils.globals import INTER_COL, BATCH_COL, VTX_COLS, NU_COL
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label
from torch_geometric.data import Batch, Data

class VertexPPNChain(nn.Module):
    """
    Experimental model for PPN-like vertex prediction
    """
    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    def __init__(self, cfg):
        super(VertexPPNChain, self).__init__()
        self.model_config = cfg
        self.backbone = UResNet_Chain(cfg)
        self.vertex_ppn = VertexPPN(cfg)
        self.num_classes = self.backbone.num_classes
        self.num_filters = self.backbone.F
        self.segmentation = ME.MinkowskiLinear(
            self.num_filters, self.num_classes)

    def forward(self, input):

        primary_labels = None
        if self.training:
            assert(len(input) == 2)
            primary_labels = input[1][:, -2]
            segment_labels = input[1][:, -1]

        input_tensors = [input[0][:, :5]]

        out = defaultdict(list)

        for igpu, x in enumerate(input_tensors):
            # input_data = x[:, :5]
            res = self.backbone([x])
            input_sparse_tensor = res['encoderTensors'][0][0]
            segmentation = self.segmentation(res['decoderTensors'][igpu][-1])
            res_vertex = self.vertex_ppn(res['finalTensor'][igpu],
                               res['decoderTensors'][igpu],
                               input_sparse_tensor=input_sparse_tensor,
                               primary_labels=primary_labels,
                               segment_labels=segment_labels)
            out['segmentation'].append(segmentation.F)
            out.update(res_vertex)
        return out


class UResNetVertexLoss(nn.Module):
    """
    See Also
    --------
    mlreco.models.uresnet.SegmentationLoss, mlreco.models.layers.common.ppnplus.PPNLonelyLoss
    """
    def __init__(self, cfg):
        super(UResNetVertexLoss, self).__init__()
        self.vertex_loss = VertexPPNLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)

    def forward(self, outputs, kinematics_label):

        res_segmentation = self.segmentation_loss(outputs, kinematics_label)

        res_vertex = self.vertex_loss(outputs, kinematics_label)

        res = {
            'loss': res_segmentation['loss'] + res_vertex['vertex_loss'],
            'accuracy': (res_segmentation['accuracy'] + res_vertex['vertex_acc']) / 2.0,
            'reg_loss': res_vertex['vertex_reg_loss']
        }
        return res

class VertexPointNet(nn.Module):

    def __init__(self, cfg, name='vertex_pointnet'):
        super(VertexPointNet, self).__init__()
        self.encoder = PointNetEncoder(cfg)
        self.D = cfg[name].get('D', 3)
        self.final_layer = nn.Sequential(
            nn.Linear(self.encoder.latent_size, self.D),
            nn.Softplus())

    def split_input(self, point_cloud, clusts=None):
        point_cloud_cpu  = point_cloud.detach().cpu().numpy()
        batches, bcounts = np.unique(point_cloud_cpu[:, BATCH_COL], return_counts=True)
        if clusts is None:
            clusts = form_clusters(point_cloud_cpu, column=INTER_COL)
        if not len(clusts):
            return Batch()
        
        data_list = []
        for i, c in enumerate(clusts):
            x = point_cloud[c, 4].view(-1, 1)
            pos = point_cloud[c, 1:4]
            data = Data(x=x, pos=pos)
            data_list.append(data)
        
        split_data = Batch.from_data_list(data_list)
        return split_data, clusts   

    def forward(self, input, clusts=None):
        res = {}
        point_cloud, = input
        batch, clusts = self.split_input(point_cloud, clusts)

        interactions = torch.unique(batch.batch)
        centroids = torch.vstack([batch.pos[batch.batch == b].mean(dim=0) for b in interactions])

        out = self.encoder(batch)
        out = self.final_layer(out)
        res['clusts'] = [clusts]
        res['vertex_pred'] = [centroids + out]
        return res
    

class VertexPointNetLoss(nn.Module):

    def __init__(self, cfg, name='vertex_pointnet_loss'):
        super(VertexPointNetLoss, self).__init__()
        self.spatial_size = cfg[name].get('spatial_size', 6144)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, res, cluster_label):

        clusts = res['clusts'][0]
        vertex_pred = res['vertex_pred'][0]

        device = cluster_label[0].device

        vtx_x = get_cluster_label(cluster_label[0], clusts, column=VTX_COLS[0])
        vtx_y = get_cluster_label(cluster_label[0], clusts, column=VTX_COLS[1])
        vtx_z = get_cluster_label(cluster_label[0], clusts, column=VTX_COLS[2])

        nu_label = get_cluster_label(cluster_label[0], clusts, column=NU_COL)
        nu_mask = torch.Tensor(nu_label == 1).bool().to(device)

        vtx_label = torch.cat([torch.Tensor(vtx_x.reshape(-1, 1)).to(device), 
                               torch.Tensor(vtx_y.reshape(-1, 1)).to(device), 
                               torch.Tensor(vtx_z.reshape(-1, 1)).to(device)], dim=1)

        mask = nu_mask & (vtx_label >= 0).all(dim=1) & (vtx_label < self.spatial_size).all(dim=1)
        loss = self.loss_fn(vertex_pred[mask], vtx_label[mask]).sum(dim=1).mean()

        result = {
            'loss': loss,
            'accuracy': loss
        }

        return result