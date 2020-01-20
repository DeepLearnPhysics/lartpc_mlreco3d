# GNN that selects edges iteratively until there are no edges left to select
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, get_cluster_group, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries, analyze_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence, get_fragment_edges
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
        
        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['iter_edge_model']
        else:
            self.model_config = cfg
            
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
            
        # Extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))
            
        # Construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))
            
        # Maximum number of iterations
        self.maxiter = self.model_config.get('maxiter', np.inf)
        
        # Threshold for matching
        self.thresh = self.model_config.get('thresh', 0.9)
        
    @staticmethod
    def default_return(device):
        """
        Default forward return if the graph is empty (no node)
        """
        xg = torch.tensor([], requires_grad=True)
        x  = torch.tensor([])
        x.to(device)
        return {'edge_pred':[xg], 'clust_ids':[x], 'group_ids':[x], 'batch_ids':[x], 'primary_ids':[x], 'edge_index':[x], 'matched':[x], 'counter':[x]}

    @staticmethod
    def assign_clusters(edge_index, edge_pred, others, matched, thresh=0.5):
        """
        Assigns clusters that have not been assigned to clusters that have been assigned
        
        Assume 2-channel output to edge_pred
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
            data[0]: (Nx8) Cluster tensor with row (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type)
        output data:
            dictionary with following keys:
                'edge_index': list of edge_index tensors used for edge prediction
                'edge_pred' : list of torch tensors with edge prediction weights
                'matched'   : numpy array of group for each cluster (identified by primary index)
                'n_iter'    : number of iterations taken
                'clust_ids': torch.tensor with cluster ids
                'group_ids': torch.tensor with cluster group ids
                'batch_ids': torch.tensor with cluster batch ids
            each list is of length k, where k is the number of times the iterative network is applied
        """
        # Get device
        cluster_label = data[0]
        device = cluster_label.device

        # Mask out the energy depositions that are not EM
        em_mask = np.where(cluster_label[:,-1] == 0)[0]

        # Find index of points that belong to the same EM clusters
        clusts = form_clusters_new(cluster_label[em_mask])
        clusts = np.array([em_mask[c] for c in clusts])
        
        # If requested, remove clusters below a certain size threshold
        if self.remove_compton:
            selection = np.where(filter_compton(clusts, self.compton_thresh))[0]
            if not len(selection):
                return self.default_return(device)
            clusts = clusts[selection]

        # Get the cluster id of each cluster
        clust_ids = get_cluster_label(cluster_label, clusts)

        # Get the group id of each cluster
        group_ids = get_cluster_group(cluster_label, clusts)

        # Get the batch id of each cluster
        batch_ids = get_cluster_batch(cluster_label, clusts)

        # Form binary/secondary bipartite incidence graph 
        primary_ids = np.where(clust_ids == group_ids)[0]

        # Keep track of who is matched. -1 is not matched
        matched = np.repeat(-1, len(clusts))
        matched[primary_ids] = primary_ids
        
        edges = []
        edge_pred = []
        
        counter = 0
        found_match = True
        
        while (-1 in matched) and (counter < self.maxiter) and found_match:
            # Continue until either:
            # 1. Everything is matched
            # 2. We have exceeded the max number of iterations
            # 3. We didn't find any matches
            counter += 1 
            
            # Get matched indices
            assigned = np.where(matched >  -1)[0]
            others   = np.where(matched == -1)[0]

            # Form a bipartite graph between assigned clusters  and others 
            edge_index = primary_bipartite_incidence(batch_ids, assigned, device=device)

            # Check if there are any edges to predict also batch norm will fail
            # on only 1 edge, so break if this is the case
            if edge_index.shape[1] < 2:
                counter -= 1
                break
            
            # Obtain vertex features
            x = cluster_vtx_features(data[0], clusts, device=device)

            # Obtain edge features
            e = cluster_edge_features(data[0], clusts, edge_index, device=device)

            # Pass through the model, get output
            out = self.edge_predictor(x, edge_index, e, torch.tensor(batch_ids))
            
            # Predictions for this edge set.
            pred = out['edge_pred'][0]
            edge_pred.append(pred)
            edges.append(edge_index)
            
            # Assign group ids to new clusters  
            matched, found_match = self.assign_clusters(edge_index,
                                                        pred[:,1] - pred[:,0],
                                                        others,
                                                        matched,
                                                        self.thresh)
 
        return {'edge_index':[edges],
                'edge_pred':[edge_pred],
                'matched':[torch.tensor(matched).to(device)],
                'counter':[torch.tensor(counter).to(device)],
                'clust_ids':[torch.tensor(clust_ids).to(device)],
                'group_ids':[torch.tensor(group_ids).to(device)],
                'batch_ids':[torch.tensor(batch_ids).to(device)],
                'primary_ids':[torch.tensor(primary_ids).to(device)]}

    
class IterEdgeChannelLoss(torch.nn.Module):
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(IterEdgeChannelLoss, self).__init__()

        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['iter_edge_model']
        else:
            self.model_config = cfg

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

    def forward(self, out, clusters, graph):
        """
        out:
            array output from the DataParallel gather function
            out['edge_index'] - n_gpus tensors of edge indexes
            out['edge_pred'] - n_gpus tensors of predicted edge weights from model forward
            out['matched'] - n_gpus arrays of group ids for each cluster
            out['counter'] - n_gpus number of iterations
        data:
            clusters: n_gpus (Nx8) Cluster tensor with row (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type)
            graph:
        """
        total_loss, total_acc = 0., 0.
        total_ari, total_ami, total_sbd, total_pur, total_eff = 0., 0., 0., 0., 0.
        total_iter = []
        ngpus = len(clusters)
        for i in range(ngpus):

            # Get the necessary data products
            clust_label = clusters[i]
            clust_ids = out['clust_ids'][i]
            group_ids = out['group_ids'][i]
            batch_ids = out['batch_ids'][i]
            primary_ids = out['primary_ids'][i]
            device = clust_ids.device
            if not len(clust_ids):
                ngpus = max(1, ngpus-1)
                continue

            # Get list of IDs of points contained in each cluster
            clusts = np.array([torch.nonzero((clust_label[:,3] == batch_ids[j]) & (clust_label[:,5] == clust_ids[j])).reshape(-1).cpu().numpy() for j in range(len(batch_ids))])

            # Append the total number of iterations
            niter = out['counter'][i]
            total_iter.append(niter)

            # Get the list of true edges (graph returned as list of [particle_id_1, particle_id_2, batch_id])
            true_edge_index = get_fragment_edges(graph[i], clust_ids, batch_ids)

            # Loop over iterations and add loss at each iter, based
            # on the graph formed at that iteration
            for j in range(niter):
                # Determine true assignments by looping over the propsed edges
                # and checking if they are in fact a true edge
                edge_index = out['edge_index'][i][j]
                #edge_assn = edge_assignment(edge_index, batch_ids, group_ids, device=device, dtype=torch.long)
                edge_assn = torch.tensor([np.any([(e == pair).all() for pair in true_edge_index]) for e in edge_index.transpose(0,1).cpu().numpy()], dtype=torch.long)
                edge_assn = edge_assn.view(-1)

                # Get edge predictions (2 channels)
                edge_pred = out['edge_pred'][i][j]

                # Increment the loss
                total_loss += self.lossfn(edge_pred, edge_assn)

            # Compute accuracy of assignment
            total_acc += secondary_matching_vox_efficiency2(
                    out['matched'][i],
                    group_ids,
                    primary_ids,
                    clusts
                )

            # Get clustering metrics
            ari, ami, sbd, pur, eff = DBSCAN_cluster_metrics2(
                out['matched'][i].cpu().numpy(),
                clusts,
                group_ids
            )
            total_ari += ari
            total_ami += ami
            total_sbd += sbd
            total_pur += pur
            total_eff += eff

        return {
            'ARI': total_ari/ngpus,
            'AMI': total_ami/ngpus,
            'SBD': total_sbd/ngpus,
            'purity': total_pur/ngpus,
            'efficiency': total_eff/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'n_iter': total_iter
        }

