# GNN that attempts to identify primaries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features
from mlreco.utils.gnn.primary import get_true_primaries
from .gnn import node_model_construct

class NodeModel(torch.nn.Module):
    """
    Driver for node prediction, assumed to be with PyTorch GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model
    
    for use in config
    model:
        modules:
            node_model:
                name: <name of node model>
                model_cfg:
                    <dictionary of arguments to pass to model>
                remove_compton: <True/False to remove compton clusters> (default True)
                compton_threshold: Minimum number of voxels
                balance_classes: <True/False for loss computation> (default False)
                loss: 'CE', 'MM' (default 'CE')
    """
    def __init__(self, cfg):
        super(NodeModel, self).__init__()

        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['node_model']
        else:
            self.model_config = cfg
        
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
            
        # Extract the model to use
        model = node_model_construct(self.model_config.get('name', 'node_econv'))
                     
        # Construct the model
        self.node_predictor = model(self.model_config.get('model_cfg', {}))

    @staticmethod
    def default_return(device):
        """
        Default forward return if the graph is empty (no node)
        """
        xg = torch.tensor([], requires_grad=True)
        x  = torch.tensor([])
        x.to(device)
        return {'node_pred':[xg], 'clust_ids':[x], 'batch_ids':[x], 'edge_index':[x]}

    def forward(self, data):
        """
        Input:
            data[0]: (Nx5) Cluster tensor with row (x, y, z, batch_id, cluster_id)
        Output:
        dictionary, with
            'node_pred': torch.tensor with node prediction weights
        """
        # Get device
        cluster_label = data[0]
        device = cluster_label.device
        
        # Find index of points that belong to the same EM clusters
        clusts = form_clusters_new(cluster_label)
        
        # If requested, remove clusters below a certain size threshold
        if self.remove_compton:
            selection = np.where(filter_compton(clusts, self.compton_thresh))[0]
            if not len(selection):
                return self.default_return(device)
            clusts = clusts[selection]

        # Get the cluster ids of each processed cluster
        clust_ids = get_cluster_label(cluster_label, clusts)

        # Get the batch ids of each cluster
        batch_ids = get_cluster_batch(cluster_label, clusts)
        
        # Form a complete graph (should add options for other structures, TODO) 
        edge_index = complete_graph(batch_ids, device=device)
        if not edge_index.shape[0]:
            return self.default_return(device)

        # Obtain vertex features
        x = cluster_vtx_features(cluster_label, clusts, device=device)

        # Obtain edge features
        e = cluster_edge_features(cluster_label, clusts, edge_index, device=device)

        # Convert the the batch IDs to a torch tensor to pass to Torch
        xbatch = torch.tensor(batch_ids).to(device)
        
        # Pass through the model, get output
        out = self.node_predictor(x, edge_index, e, xbatch)

        return {**out,
                'clust_ids':[torch.tensor(clust_ids)],
                'batch_ids':[torch.tensor(batch_ids)],
                'edge_index':[edge_index]}


class NodeChannelLoss(torch.nn.Module):
    """
    Node loss based on two channel output
    """
    def __init__(self, cfg):
        super(NodeChannelLoss, self).__init__()

        # Get the model loss parameters
        if 'modules' in cfg:
            self.model_config = cfg['modules']['node_model']
        else:
            self.model_config = cfg
        
        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')
        self.balance_classes = self.model_config.get('balance_classes', False)
        
        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Unrecognized loss: ' + self.loss)

    def forward(self, out, primary_points):
        """
        out:
            dictionary output from the DataParallel gather function
            out['node_pred'] - n_gpus tensors of predicted node weights from model forward
        data:
            primary_points - n_gpus tensor of (x, y, z, batch_id, primary_id)
        """
        total_loss, total_acc = 0., 0.
        primary_ids, primary_batch_ids = [], []
        ngpus = len(primary_points)
        for i in range(len(primary_points)):

            # Get the necessary data products
            points = primary_points[i]
            node_pred = out['node_pred'][i]
            clust_ids = out['clust_ids'][i]
            batch_ids = out['batch_ids'][i]
            device = node_pred.device
            if not len(clust_ids):
                if ngpus > 1:
                    ngpus -= 1
                continue

            # Use the primary point ids to determine the true primary clusters
            primaries = get_true_primaries(clust_ids, batch_ids, points)
            primary_ids.extend(primaries)

            # Use the primary information to determine a the node assignment
            node_assn = torch.tensor([int(i in primaries) for i in range(len(clust_ids))]).to(device)

            # Increment the loss, balance classes if requested
            if self.balance_classes and len(primaries):
                nS, nP = np.unique(node_assn, return_counts=True)[1]
                wP, wS = float(nP)/(nP+nS), float(nS)/(nP+nS)
                total_loss += wP*self.lossfn(node_pred[node_assn==1], node_assn[node_assn==1])
                total_loss += wS*self.lossfn(node_pred[node_assn==0], node_assn[node_assn==0])
            else:
                total_loss += self.lossfn(node_pred, node_assn)

            # Compute accuracy of assignment
            total_acc += float(torch.sum(torch.argmax(node_pred, dim=1) == node_assn))/len(clust_ids)

        return {
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'primary_ids': np.array(primary_ids)
        }

