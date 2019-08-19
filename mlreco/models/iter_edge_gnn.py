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
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency2
from mlreco.utils.gnn.evaluation import DBSCAN_cluster_metrics2
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
            
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
            
        # extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))
            
        # construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))
            
        # maximum number of iterations
        self.maxiter = self.model_config.get('maxiter', np.inf)
        
        # threshold for matching
        self.thresh = self.model_config.get('thresh', 0.9)
        
        # check if primaries assignment should be thresholded
        self.pmd = self.model_config.get('primary_max_dist', None)
            
    
    @staticmethod
    def assign_clusters(edge_index, edge_pred, others, matched, thresh=0.5):
        """
        assigns clusters that have not been assigned to clusters that have been assigned
        
        assume 2-channel output to edge_pred
        """
        found_match = False
        for i in others:
            inds = edge_index[1,:] == i
            if sum(inds) == 0:
                continue
            indmax = torch.argmax(edge_pred[inds])
            ei = np.where(inds.cpu().detach().numpy())[0][indmax]
            if edge_pred[ei] > thresh:
                found_match = True
                # we make an assignment
                j = edge_index[0, ei]
                matched[i] = matched[j]
        return matched, found_match
        
        
    def forward(self, data):
        """
        input data:
            data[0] - dbscan data
            data[1] - primary data
        output data:
            dictionary with following keys:
                edges     : list of edge_index tensors used for edge prediction
                edge_pred : list of torch tensors with edge prediction weights
                matched   : numpy array of group for each cluster (identified by primary index)
                n_iter    : number of iterations taken
            each list is of length k, where k is the number of times the iterative network is applied
        """
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove compton clusters
        # if no cluster fits this condition, return
        if self.remove_compton:
            selection = filter_compton(clusts, self.compton_thresh) # non-compton looking clusters
            if not len(selection):
                e = torch.tensor([], requires_grad=True)
                if data[0].is_cuda:
                    e = e.cuda()
                return e

            clusts = clusts[selection]
        

        #others = np.array([(i not in primaries) for i in range(n)])
        batch = get_cluster_batch(data[0], clusts)
        # get x batch
        xbatch = torch.tensor(batch).cuda()
        
        primaries = assign_primaries(data[1], clusts, data[0], max_dist=self.pmd)
        # keep track of who is matched. -1 is not matched
        matched = np.repeat(-1, len(clusts))
        matched[primaries] = primaries
        # print(matched)
        
        edges = []
        edge_pred = []
        
        counter = 0
        found_match = True
        
        while (-1 in matched) and (counter < self.maxiter) and found_match:
            # continue until either:
            # 1. everything is matched
            # 2. we have exceeded the max number of iterations
            # 3. we didn't find any matches
            
            #print('iter ', counter)
            counter = counter + 1
            
            # get matched indices
            assigned = np.where(matched >  -1)[0]
            # print(assigned)
            others   = np.where(matched == -1)[0]
            
            edge_index = primary_bipartite_incidence(batch, assigned, cuda=True)
            # check if there are any edges to predict
            # also batch norm will fail on only 1 edge, so break if this is the case
            if edge_index.shape[1] < 2:
                counter -= 1
                break
            
            # obtain vertex features
            x = cluster_vtx_features(data[0], clusts, cuda=True)
            # obtain edge features
            e = cluster_edge_features(data[0], clusts, edge_index, cuda=True)
            # print(x.shape)
            # print(torch.max(edge_index))
            # print(torch.min(edge_index))
        
            out = self.edge_predictor(x, edge_index, e, xbatch)
            
            # predictions for this edge set.
            edge_pred.append(out[0][0])
            edges.append(edge_index)
            
            #print(out[0][0].shape)

            matched, found_match = self.assign_clusters(edge_index,
                                                        out[0][0][:,1] - out[0][0][:,0],
                                                        others,
                                                        matched,
                                                        self.thresh)

            
            # print(edges)
            # print(edge_pred)

        #print('num iterations: ', counter)

        matched = torch.tensor(matched)
        counter = torch.tensor([counter])            
        if data[0].is_cuda:
            matched = matched.cuda()
            counter = counter.cuda()

        return {'edges':[edges],
                'edge_pred':[edge_pred],
                'matched':[matched],
                'counter':[counter]}

    
class IterEdgeChannelLoss(torch.nn.Module):
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(IterEdgeChannelLoss, self).__init__()
        self.model_config = cfg['modules']['iter_edge_model']

        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
        self.pmd = self.model_config.get('primary_max_dist')
        
        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')
        
        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('unrecognized loss: ' + self.loss)
        
        
    def forward(self, out, clusters, groups, primary):
        """
        out:
            array output from the DataParallel gather function
            out[0] - n_gpus tensors of edge indexes
            out[1] - n_gpus tensors of predicted edge weights from model forward
            out[2] - n_gpus arrays of group ids for each cluster
            out[3] - n_gpus number of iterations
        data:
            cluster_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, cluster_id)
            group_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, group_id) 
            em_primaries - n_gpus tensor of (x, y, z) coordinates of origins of EM primaries
        """
        total_loss, total_acc, total_primary_fdr, total_primary_acc, total_iter = 0., 0., 0., 0., 0
        total_ari, total_ami, total_sbd, total_pur, total_eff = 0., 0., 0., 0., 0.
        ngpus = len(clusters)
        for i in range(ngpus):
            data0 = clusters[i]
            data1 = groups[i]
            data2 = primary[i]

            clusts = form_clusters_new(data0)

            # remove compton clusters
            # if no cluster fits this condition, return
            if self.remove_compton:
                selection = filter_compton(clusts) # non-compton looking clusters
                if not len(selection):
                    edge_pred = out[1][i][0]
                    total_loss += self.lossfn(edge_pred, edge_pred)
                    total_acc += 1.

            clusts = clusts[selection]

            # process group data
            data_grp = data1

            # form primary/secondary bipartite graph
            primaries = assign_primaries(data2, clusts, data0)
            batch = get_cluster_batch(data0, clusts)
            # edge_index = primary_bipartite_incidence(batch, primaries)
            group = get_cluster_label(data_grp, clusts)

            primaries_true = assign_primaries(data2, clusts, data1, use_labels=True)
            primary_fdr, primary_tdr, primary_acc = analyze_primaries(primaries, primaries_true)
            total_primary_fdr += primary_fdr
            total_primary_acc += primary_acc

            niter = out[3][i][0] # number of iterations
            total_iter += niter

            # loop over iterations and add loss at each iter.
            for j in range(niter):
                # determine true assignments
                edge_index = out[0][i][j]
                edge_assn = edge_assignment(edge_index, batch, group, cuda=True, dtype=torch.long)

                # get edge predictions (2 channels)
                edge_pred = out[1][i][j]

                edge_assn = edge_assn.view(-1)

                total_loss += self.lossfn(edge_pred, edge_assn)

            # compute accuracy of assignment
            total_acc += secondary_matching_vox_efficiency2(
                    out[2][i],
                    group,
                    primaries,
                    clusts
                )

            # get clustering metrics
            #print(out[2][i].shape)
            ari, ami, sbd, pur, eff = DBSCAN_cluster_metrics2(
                out[2][i].cpu().numpy(),
                clusts,
                group
            )
            total_ari += ari
            total_ami += ami
            total_sbd += sbd
            total_pur += pur
            total_eff += eff

        return {
            'primary_fdr': total_primary_fdr/ngpus,
            'primary_acc': total_primary_acc/ngpus,
            'ARI': ari/ngpus,
            'AMI': ami/ngpus,
            'SBD': sbd/ngpus,
            'purity': pur/ngpus,
            'efficiency': eff/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'n_iter': total_iter
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
            
        self.remove_compton = self.model_config.get('remove_compton', True)

        self.balance = self.model_config.get('balance_classes', True)
        
            
    @staticmethod
    def balance_classes(edge_assn, edge_pred):
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
        
        
    def forward(self, out, clusters, groups, primary):
        """
        out:
            array output from the DataParallel gather function
            out[0] - n_gpus tensors of edge indexes
            out[1] - n_gpus tensors of predicted edge weights from model forward
            out[2] - n_gpus arrays of group ids for each cluster
            out[3] - n_gpus number of iterations
        data:
            cluster_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, cluster_id)
            group_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, group_id) 
            em_primaries - n_gpus tensor of (x, y, z) coordinates of origins of EM primaries
        """
        total_loss, total_acc, total_primary_fdr, total_primary_acc, total_iter = 0., 0., 0., 0., 0
        ngpus = len(clusters)
        for i in range(ngpus):
            data0 = clusters[i]
            data1 = groups[i]
            data2 = primary[i]

            clusts = form_clusters_new(data0)

            # remove compton clusters
            # if no cluster fits this condition, return
            if self.remove_compton:
                selection = filter_compton(clusts) # non-compton looking clusters
                if not len(selection):
                    edge_pred = out[1][i]
                    total_loss += self.lossfn(edge_pred, edge_pred)
                    total_acc += 1.
                    continue

            clusts = clusts[selection]

            # process group data
            data_grp = data1

            # form primary/secondary bipartite graph
            primaries = assign_primaries(data2, clusts, data0)
            batch = get_cluster_batch(data0, clusts)
            # edge_index = primary_bipartite_incidence(batch, primaries)
            group = get_cluster_label(data_grp, clusts)

            primaries_true = assign_primaries(data2, clusts, data1, use_labels=True)
            primary_fdr, primary_tdr, primary_acc = analyze_primaries(primaries, primaries_true)
            total_primary_fdr += primary_fdr
            total_primary_acc += primary_acc

            niter = out[3][i][0] # number of iterations
            total_iter += niter
            for j in range(niter):
                # determine true assignments
                edge_index = out[0][i][j]
                edge_assn = edge_assignment(edge_index, batch, group, cuda=True)

                edge_pred = out[1][i][j]
                # print(edge_pred)

                # print(edge_assn.shape)
                # print(edge_pred.shape)
                edge_assn = edge_assn.view(-1)
                edge_pred = edge_pred.view(-1)
                # print(edge_assn.shape)
                # print(edge_pred.shape)

                if self.balance:
                    edge_assn, edge_pred = self.balance_classes(edge_assn, edge_pred)

                total_loss += self.lossfn(edge_pred, edge_assn)

            # compute accuracy of assignment
            # need to multiply by batch size to be accurate
            #total_acc = (np.max(batch) + 1) * torch.tensor(secondary_matching_vox_efficiency(edge_index, edge_assn, edge_pred, primaries, clusts, len(clusts)))
            # use out['matched']
            total_acc += torch.tensor(
                secondary_matching_vox_efficiency2(
                    out[2][i],
                    group,
                    primaries,
                    clusts
                )
            )

        return {
            'primary_fdr': total_primary_fdr/ngpus,
            'primary_acc': total_primary_acc/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'n_iter': total_iter
        }
