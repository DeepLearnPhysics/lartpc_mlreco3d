# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries, analyze_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency3, DBSCAN_cluster_metrics
from .gnn import edge_model_construct

class EdgeModel(torch.nn.Module):
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
                compton_threshold: Minimum number of voxels
                balance_classes: <True/False for loss computation> (default False)
                loss: 'CE' or 'MM' (default 'CE')
    """
    def __init__(self, cfg):
        super(EdgeModel, self).__init__()

        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg

        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)

        # Extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))

        # Construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))
      
        # Check if primaries assignment should be thresholded
        self.pmd = self.model_config.get('primary_max_dist', None)

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
        inputs data:
            data[0]: (Nx5) Cluster tensor with row (x, y, z, batch_id, cluster_id)
            data[1]: primary data
        output:
        dictionary, with
            'edge_pred': torch.tensor with edge prediction weights
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

        # Form primary/secondary bipartite incidence graph
        # TODO Once primary IDs and cluster IDs are matched, need to change this!
        # TODO Current method does not use truth, matches points and clusters distance-wise
        # TODO for a lack of a better way (cluster ID and particle ID not matched)
        primary_ids = assign_primaries(data[1], clusts, cluster_label, max_dist=self.pmd)
        edge_index = primary_bipartite_incidence(batch_ids, primary_ids, device=device)
        if not edge_index.shape[0]:
            return self.default_return(device)
        
        # Obtain vertex features
        x = cluster_vtx_features(cluster_label, clusts, device=device)

        # Obtain edge features
        e = cluster_edge_features(cluster_label, clusts, edge_index, device=device)

        # Convert the the batch IDs to a torch tensor to pass to Torch
        xbatch = torch.tensor(batch_ids).to(device)
        
        # Pass through the model, get output
        out = self.edge_predictor(x, edge_index, e, xbatch)

        return {**out,
                'clust_ids':[torch.tensor(clust_ids)],
                'batch_ids':[torch.tensor(batch_ids)],
                'primary_ids':[torch.tensor(primary_ids)],
                'edge_index':[edge_index]}


class EdgeChannelLoss(torch.nn.Module):
    """
    Edge loss based on two channel output
    """
    def __init__(self, cfg):
        super(EdgeChannelLoss, self).__init__()

        # Ge the model loss parameters
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
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
            raise Exception('unrecognized loss: ' + self.loss)

    def forward(self, out, clusters, groups, primary):
        """
        out:
            dictionary output from the DataParallel gather function
            out['edge_pred'] - n_gpus tensors of predicted edge weights from model forward
        data:
            group_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, group_id) 
        """
        edge_ct = 0
        total_loss, total_acc, total_primary_fdr, total_primary_acc = 0., 0., 0., 0.
        ari, ami, sbd, pur, eff = 0., 0., 0., 0., 0.
        ngpus = len(clusters)
        for i in range(len(clusters)):

            # Get the necessary data products
            clust_label = clusters[i]
            group_label = groups[i]
            primary_points = primary[i]
            edge_pred = out['edge_pred'][i]
            clust_ids = out['clust_ids'][i]
            batch_ids = out['batch_ids'][i]
            primary_ids = out['primary_ids'][i]
            edge_index = out['edge_index'][i]
            device = edge_pred.device
            if not len(clust_ids):
                if ngpus > 1:
                    ngpus -= 1
                continue

            # Get list of IDs of points contained in each cluster
            clusts = np.array([np.where((clust_label[:,3] == batch_ids[j]) & (clust_label[:,4] == clust_ids[j]))[0] for j in range(len(batch_ids))])

            # Get the group ids of each processed cluster
            group_ids = get_cluster_label(group_label, clusts)

            # Get the true primary assignment and compare to effective assigment
            # TODO Vestigial feature that should go away once cluster ID and particle ID match 
            primaries_true = assign_primaries(primary_points, clusts, group_label, use_labels=True)
            primary_fdr, primary_tdr, primary_acc = analyze_primaries(primary_ids, primaries_true)
            total_primary_fdr += primary_fdr
            total_primary_acc += primary_acc

            # Use group information to determine the true edge assigment 
            edge_assn = edge_assignment(edge_index, batch_ids, group_ids, device=device, dtype=torch.long)
            edge_assn = edge_assn.view(-1)

            # Increment the loss, balance classes if requested (TODO)
            total_loss += self.lossfn(edge_pred, edge_assn)

            # Compute accuracy of assignment
            total_acc += torch.tensor(
                secondary_matching_vox_efficiency3(
                    edge_index,
                    edge_assn,
                    edge_pred,
                    primary_ids,
                    clusts,
                    len(clusts)
                )
            )

            ari0, ami0, sbd0, pur0, eff0 = DBSCAN_cluster_metrics(
                edge_index,
                edge_assn,
                edge_pred,
                primary_ids,
                clusts,
                len(clusts)
            )
            ari += ari0
            ami += ami0
            sbd += sbd0
            pur += pur0
            eff += eff0

            edge_ct += edge_index.shape[1]

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
            'edge_count': edge_ct
        }

