# GNN that selects edges iteratively until there are no edges left to select
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries, analyze_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment, cluster_vtx_features_old
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency
from mlreco.utils.groups import process_group_data
from .gnn import edge_model_construct

class IterativeEdgeModel(torch.nn.Module):
    """
    GNN that applies an edge model iteratively to select edges until there are no edges left to select
    
    for use in config:
    model:
        modules:
            iter_gnn:
                edge_model: <config for edge gnn model>
    """
    def __init__(self, cfg):
        super(IterativeEdgeModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['iter_edge_model']
        else:
            self.model_config = cfg
            
        if 'remove_compton' in self.model_config:
            self.remove_compton = self.model_config['remove_compton']
        else:
            self.remove_compton = True
            
        if 'name' in self.model_config:
            # extract the actual model to use
            model = edge_model_construct(self.model_config['name'])
        else:
            model = edge_model_construct('basic_attention')
            
        if 'model_cfg' in self.model_config:
            # construct with model parameters
            self.edge_predictor = model(self.model_config['model_cfg'])
        else:
            self.edge_predictor = model({})
            
        # maximum number of iterations
        if 'maxiter' in self.model_config:
            self.maxiter = self.model_config['maxiter']
        else:
            self.maxiter = 5
            
            
    def assign_clusters(self, edge_index, edge_pred, others, matched, thresh=0.5):
        """
        assigns clusters that have not been assigned to clusters that have been assigned
        """
        for i in others:
            inds = edge_index[1,:] == i
            if sum(inds) == 0:
                continue
            indmax = torch.argmax(edge_pred[inds])
            ei = np.where(inds.cpu().detach().numpy())[0][indmax]
            if edge_pred[ei] > thresh:
                # we make an assignment
                j = edge_index[0, ei]
                matched[i] = matched[j]
        return matched
        
        
    def forward(self, data):
        """
        input data:
            data[0] - dbscan data
            data[1] - primary data
        output data:
            dictionary with following keys:
                edges     : list of edge_index tensors used for edge prediction
                edge_pred : list of torch tensors with edge prediction weights
                edge_cont : list of edge contractions at each step
            each list is of length k, where k is the number of times the iterative network is applied
        """
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove compton clusters
        # if no cluster fits this condition, return
        if self.remove_compton:
            selection = filter_compton(clusts) # non-compton looking clusters
            if not len(selection):
                e = torch.tensor([], requires_grad=True)
                if data[0].is_cuda:
                    e.cuda()
                return e
            clusts = clusts[selection]
        

        #others = np.array([(i not in primaries) for i in range(n)])
        batch = get_cluster_batch(data[0], clusts)
        # get x batch
        xbatch = torch.tensor(batch).cuda()
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data[1], clusts, data[0])
        # keep track of who is matched. -1 is not matched
        matched = np.repeat(-1, len(clusts))
        matched[primaries] = primaries
        # print(matched)
        
        edges = []
        edge_pred = []
        
        counter = 0
        
        while (-1 in matched) and (counter < self.maxiter):
            
            print('iter ', counter)
            counter = counter + 1
            
            # get matched indices
            assigned = np.where(matched >  -1)[0]
            # print(assigned)
            others   = np.where(matched == -1)[0]
            
            edge_index = primary_bipartite_incidence(batch, assigned, cuda=True)
            
            # obtain vertex features
            x = cluster_vtx_features(data[0], clusts, cuda=True)
            # obtain edge features
            e = cluster_edge_features(data[0], clusts, edge_index, cuda=True)
            # print(x.shape)
            # print(torch.max(edge_index))
            # print(torch.min(edge_index))
        

            out = self.edge_predictor(x, edge_index, e, xbatch)
            
            # predictions for this edge set.
            edge_pred.append(out['edge_pred'])
            edges.append(edge_index)
            
            matched = self.assign_clusters(edge_index, out['edge_pred'], others, matched)
            
            print(edges)
            print(edge_pred)
            
        return {
            'edges': edges,
            'edge_pred': edge_pred,
            'matched': matched
        }
    
    
class IterEdgeLabelLoss(torch.nn.Module):
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(IterEdgeLabelLoss, self).__init__()
        self.model_config = cfg['modules']['iter_edge_model']

        if 'loss' in self.model_config:
            if self.model_config['loss'] == 'L1':
                self.lossfn = torch.nn.L1Loss(reduction='sum')
            elif self.model_config['loss'] == 'L2':
                self.lossfn = torch.nn.MSELoss(reduction='sum')
        else:
            self.lossfn = torch.nn.L1Loss(reduction='sum')
            
        if 'remove_compton' in self.model_config:
            self.remove_compton = self.model_config['remove_compton']
        else:
            self.remove_compton = True

        if 'balance_classes' in self.model_config:
            self.balance = self.model_config['balance_classes']
        else:
            # default behavior
            self.balance = True
            
    def balance_classes(self, edge_assn, edge_pred):
        # weight edges so that 0/1 labels appear equally often
        ind0 = edge_assn == 0
        ind1 = edge_assn == 1
        # number in each class
        n0 = torch.sum(ind0).float()
        n1 = torch.sum(ind1).float()
        #print("n0 = ", n0, " n1 = ", n1)
        # weights to balance classes
        w0 = n1 / (n0 + n1)
        w1 = n0 / (n0 + n1)
        #print("w0 = ", w0, " w1 = ", w1)
        edge_assn[ind0] = w0 * edge_assn[ind0]
        edge_assn[ind1] = w1 * edge_assn[ind1]
        edge_pred = edge_pred.clone()
        edge_pred[ind0] = w0 * edge_pred[ind0]
        edge_pred[ind1] = w1 * edge_pred[ind1]
        return edge_assn, edge_pred
        
        
    def forward(self, out, data0, data1, data2):
        """
        out:
            dictionary output from GNN Model
            keys:
                'edge_pred': predicted edge weights from model forward
        data:
            data[0] - 5 types data
            data[1] - groups data
            data[2] - primary data
        """
        data0 = data0[0]
        data1 = data1[0]
        data2 = data2[0]

        clusts = form_clusters_new(data0)

        # remove compton clusters
        # if no cluster fits this condition, return
        if self.remove_compton:
            selection = filter_compton(clusts) # non-compton looking clusters
            if not len(selection):
                edge_pred = out['edge_pred'][0]
                total_loss = self.lossfn(edge_pred, edge_pred)
                return {
                    'accuracy': 1.,
                    'loss_seg': total_loss
                }

        clusts = clusts[selection]

        # process group data
        data_grp = data1

        # form primary/secondary bipartite graph
        primaries = assign_primaries(data2, clusts, data0)
        batch = get_cluster_batch(data0, clusts)
        edge_index = primary_bipartite_incidence(batch, primaries)
        group = get_cluster_label(data_grp, clusts)

        primaries_true = assign_primaries(data2, clusts, data1, use_labels=True)
        primary_fdr, primary_tdr, primary_acc = analyze_primaries(primaries, primaries_true)
        
        # determine true assignments
        edge_index = out['edges'][0]
        edge_assn = edge_assignment(edge_index, batch, group, cuda=True)

        edge_pred = out['edge_pred'][0]
        print(edge_pred)
        
        print(edge_assn.shape)
        print(edge_pred.shape)
        edge_assn = edge_assn.view(-1)
        edge_pred = edge_pred.view(-1)
        print(edge_assn.shape)
        print(edge_pred.shape)

        if self.balance:
            edge_assn, edge_pred = self.balance_classes(edge_assn, edge_pred)

        total_loss = self.lossfn(edge_pred, edge_assn)

        # compute accuracy of assignment
        # need to multiply by batch size to be accurate
        total_acc = (np.max(batch) + 1) * torch.tensor(secondary_matching_vox_efficiency(edge_index, edge_assn, edge_pred, primaries, clusts, len(clusts)))

        return {
            'primary_fdr': primary_fdr,
            'primary_acc': primary_acc,
            'accuracy': total_acc,
            'loss_seg': total_loss
        }