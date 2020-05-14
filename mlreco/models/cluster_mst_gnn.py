# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label
from mlreco.utils.gnn.primary import assign_primaries, analyze_primaries
from mlreco.utils.gnn.data import edge_features, edge_assignment
from mlreco.utils.gnn.features.utils import edge_labels_to_node_labels
from mlreco.utils.groups import process_group_data
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency
from .gnn import edge_model_construct

from mlreco.utils.gnn.features.core import generate_graph

from topologylayer.nn import DelaunayLayer, BarcodePolyFeature
from .layers.uresnet import UResNet

class MSTEdgeModel(torch.nn.Module):
    """
    Driver for edge prediction, assumed to be with PyTorch GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model

    for use in config
    model:
        modules:
            edge_model:
                name: <name of edge model>
                model_cfg:
                    <dictionary of arguments to pass to model>
                remove_compton: <True/False to remove compton clusters> (default True)
                balance_classes: <True/False for loss computation> (default True)
    """
    def __init__(self, cfg):
        super(MSTEdgeModel, self).__init__()

        self.model_config = cfg['mst_edge_model']

        # only use points with EM segmentation label
        self.em_only = self.model_config.get('em_only', False)

        # Optional UResNet
        uresnet_cfg = self.model_config.get('uresnet_cfg')
        if uresnet_cfg is None:
            self.transform_input = False
        else:
            self.transform_input = True
            self.transform = UResNet(uresnet_cfg)
            uresnet_path = self.model_config.get('uresnet_path')
            if uresnet_path:
                # load pre-trained weights
                with open(uresnet_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu')
                    # Edit checkpoint variable names
                    for name in self.transform.state_dict():
                        other_name = 'module.' + name
                        if other_name in checkpoint['state_dict']:
                            checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(other_name)

                    bad_keys = self.transform.load_state_dict(checkpoint['state_dict'], strict=False)

                    print(bad_keys)
            # freezes uresnet model
            if self.model_config.get('uresnet_freeze', True):
                for param in self.transform.parameters():
                    param.requires_grad = False

        # regularization on connected components
        self.reg_ph0 = self.model_config.get('reg_ph0', 0.0)
        # regularization on cycles
        self.reg_ph1 = self.model_config.get('reg_ph1', 0.0)
        if self.reg_ph1 > 0:
            self.hdim = 1
            self.cpxdim = 2
            self.halg = 'hom'
        else:
            self.hdim = 0
            self.cpxdim = 1
            self.halg = 'union_find'


        # extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))

        # construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))



    def forward(self, data):
        """
        inputs data:
            data[0] - energy deposition data
            data[1] - 5-types labels
        output:
        dictionary, with
            'edge_pred': torch.tensor with edge prediction weights
            'complex'  : simplicial complex of Freudenthal triangulation
            'edges'    : torch tensor with edges used
        """
        # get device
        dev = data[0].device

        data0 = data[0]

        if self.em_only:
            # select voxels with 5-types classification > 1
            sel = data[1][:,-1] > 1
            data0 = data0[sel,:]

        # first get data
        voxels = data0[:,:3]
        xbatch = data0[:,3]
        energy = data0[:,-1]


        # optionally pass data through transformation (scn UResNet)
        if self.transform_input:
            x = self.transform(data0)
        else:
            x = energy.float()

        # construct graph from Delaunay triangulation
        vox_np = voxels.detach().cpu().numpy()
        X = DelaunayLayer(vox_np, self.cpxdim, alg=self.halg, extension='flag', sublevel=False, maxdim=self.hdim)
        edges = X.Edges()



        # construct edge features
        edge_index = torch.tensor(edges.T, dtype=torch.long).to(dev)
        e = edge_features(data[0], edge_index, cuda=False, device=dev)

        # get output
        out = self.edge_predictor(x, edge_index, e, xbatch)

        return {
            'complex' : X,
            'edges'   : edge_index,
            'edge_pred' : out
        }


class MSTEdgeChannelLoss(torch.nn.Module):
    """
    Edge loss based on two channel output
    """
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(MSTEdgeChannelLoss, self).__init__()
        self.model_config = cfg['mst_edge_model']

        # only use points with EM segmentation label
        self.em_only = self.model_config.get('em_only', False)

        # use MST only for edges
        self.mst = self.model_config.get('MST', True)

        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')

        # regularization on connected components
        self.reg_ph0 = self.model_config.get('reg_ph0', 0.0)
        # regularization on cycles
        self.reg_ph1 = self.model_config.get('reg_ph1', 0.0)
        self.reg = False
        if self.reg_ph0 > 0:
            self.reg = True
            self.regh0 = BarcodePolyFeature(0, 1, 0) # sum of lengths of finite H0
        if self.reg_ph1 > 0:
            self.reg = True
            self.regh1 = BarcodePolyFeature(1, 1, 0) # sum of lengths of finite H1

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('unrecognized loss: ' + self.loss)


    def forward(self, out, data0, data1):
        """
        out : dictionary, with
            'edge_pred': torch.tensor with edge prediction weights
            'complex'  : simplicial complex of Freudenthal triangulation
            'edges'    : torch tensor with edges used
        data:
            data0 - groups data
            data1 - 5-types data
        """
        dev = data0[0].device

        data_grps = data0[0]
        data_seg = data1[0]
        edge_pred = out['edge_pred'][0][0]
        X = out['complex']
        edge_index = out['edges']

        # 1. compute MST on edge weights edge_pred[:,1] - edge_pred[:,0]
        ft = edge_pred[:,1] - edge_pred[:,0]
        # if using MST in loss
        if self.mst:
            # loss is only on MST
            ce_inds = X.CriticalEdgeInds(ft) # critical edges form MST
            active_edge_index = edge_index[:,ce_inds]
            active_edge_pred = edge_pred[ce_inds,:]
        else:
            # loss is on all edges
            active_edge_index = edge_index
            active_edge_pred = edge_pred

        # 2. get edge labels
        """
        inputs:
        edge_index: torch tensor of edges
        batches: torch tensor of batch id for each node
        groups: torch tensor of group ids for each node
        """
        if self.em_only:
            # select voxels with 5-types classification > 1
            sel = data_seg[:,-1] > 1
            data_grps = data_grps[sel,:]

        batch = data_grps[:,-2] # get batch from data
        group = data_grps[:,-1] # get gouprs from data
        edge_assn = edge_assignment(active_edge_index, batch, group, cuda=False, dtype=torch.long, device=dev)

        # 3. compute loss, only on critical edges
        # extract critical edges
        loss = self.lossfn(active_edge_pred, edge_assn)

        loss_terms = {'loss_raw': loss.detach().cpu().item()}

        # 3a. add regularization (optional)
        if self.reg:
            ph_out = X(ft)
            if self.reg_ph0 > 0:
                penh0 = self.reg_ph0 * self.regh0(ph_out)
                loss_terms['reg_ph0'] = penh0.detach().cpu().item()
                loss = loss + penh0
            if self.reg_ph1 > 0:
                penh1 = self.reg_ph1 * self.regh1(ph_out)
                loss_terms['reg_ph1'] = penh1.detach().cpu().item()
                loss = loss + penh1


        # 4. compute predicted clustering with some threhsold (0?)
        clusts = X.GetClusters(ft, 0.0) # clusters based on edge being more likely than not
        clusts = np.array(clusts)

        # print(clusts)
        # 5. compute clustering metrics vs. group id.
        group = group.cpu().detach().numpy()
        # print(group)
        sbd = SBD(clusts, group)
        ami = AMI(clusts, group)
        ari = ARI(clusts, group)
        pur, eff = purity_efficiency(clusts, group)

        return {
            'SBD' : sbd,
            'AMI' : ami,
            'ARI' : ari,
            'purity': pur,
            'efficiency': eff,
            'accuracy': ari,
            'loss': loss,
            **loss_terms
        }
