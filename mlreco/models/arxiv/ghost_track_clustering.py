from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
#from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.clustercnn_se import ClusterCNN, ClusteringLoss
from mlreco.models.layers.dbscan import distances
from mlreco.utils.deghosting import adapt_labels

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.message_passing.meta import MetaLayerModel as GNN
from .gnn import node_encoder_construct, edge_encoder_construct

from .cluster_cnn import spice_loss_construct
from mlreco.models.grappa import GNNLoss
from mlreco.models.gnn.losses.node_grouping import *

from mlreco.utils.gnn.evaluation import node_assignment_score, primary_assignment
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.gnn.cluster import cluster_direction, get_cluster_batch


class GhostTrackClustering(torch.nn.Module):
    """
    Run UResNet and use its encoding/decoding feature maps for PPN layers
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (float,), (3, 1)],
    # ]
    MODULES = ['spatial_embeddings', 'uresnet_lonely', 'fragment_clustering',
                'node_encoder', 'edge_encoder', 'track_edge_model', 'track_gnn', 'full_chain_loss'] + ClusterCNN.MODULES

    def __init__(self,cfg):
        super(GhostTrackClustering, self).__init__()
        self.uresnet_lonely = UResNet(cfg)
        self.spatial_embeddings = ClusterCNN(cfg)

        self.input_features = cfg['uresnet_lonely'].get('features', 1)

        # Fragment formation parameters
        self.frag_cfg     = cfg['fragment_clustering']
        self.s_thresholds = self.frag_cfg.get('s_thresholds', [0.0, 0.0, 0.0, 0.0])
        self.p_thresholds = self.frag_cfg.get('p_thresholds', [0.5, 0.5, 0.5, 0.5])
        self.cluster_all  = self.frag_cfg.get('cluster_all', True)

        # Initialize the geometric encoders
        self.node_encoder = node_encoder_construct(cfg['grappa'])
        self.edge_encoder = edge_encoder_construct(cfg['grappa'])

        # Initialize the GNN models
        self.track_gnn  = GNN(cfg['grappa']['gnn_model'])
        self.min_frag_size = cfg['grappa']['base'].get('node_min_size', -1)

    def extract_fragment(self, input, result):
        batch_labels = input[0][:,3]
        fragments = []
        frag_batch_ids = []
        semantic_labels = torch.argmax(result['segmentation'][0].detach(), dim=1).flatten()
        for batch_id in batch_labels.unique():
            for s in semantic_labels.unique():
                if s > 3: continue
                mask = torch.nonzero((batch_labels == batch_id) & (semantic_labels == s)).flatten()
                pred_labels = fit_predict(embeddings = result['embeddings'][0][mask],
                                          seediness = result['seediness'][0][mask],
                                          margins = result['margins'][0][mask],
                                          fitfunc = gaussian_kernel,
                                          s_threshold = self.s_thresholds[s],
                                          p_threshold = self.p_thresholds[s],
                                          cluster_all = self.cluster_all)
                for c in pred_labels.unique():
                    if c < 0:
                        continue
                    if torch.sum(pred_labels == c) < self.min_frag_size:
                        continue
                    fragments.append(mask[pred_labels == c])
                    frag_batch_ids.append(int(batch_id))

        fragments = np.array([f.detach().cpu().numpy() for f in fragments if len(f)])
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(cnts)].item()

        return fragments, frag_batch_ids, frag_seg

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        """
        point_cloud = input[0]
        device = input[0].device
        result1 = self.uresnet_lonely((point_cloud,))
        #print((result1['ghost'][0].argmax(dim=1) == 1).sum(), (result1['ghost'][0].argmax(dim=1) == 0).sum())
        deghost = result1['ghost'][0].argmax(dim=1) == 0
        input[0] = input[0][deghost]
        if self.input_features > 1:
            input[0] = input[0][:, :-self.input_features+1]

        #print(new_point_cloud.size())
        result2 = self.spatial_embeddings(input)
        result = {}
        result.update(result1)
        result.update(result2)

        segmentation = result['segmentation'][0].clone()
        deghost_result = {}
        deghost_result.update(result)
        deghost_result['segmentation'][0] = result['segmentation'][0][deghost]

        # Extract fragment predictions to input into the GNN
        fragments, frag_batch_ids, frag_seg = self.extract_fragment(input, deghost_result)

        # # Optionnally break the tracks using PPN points
        # if self._break_ppn:
        #

        # Initialize a complete graph for edge prediction, get track fragment and edge features
        em_mask = np.where(frag_seg == 1)[0]
        edge_index = complete_graph(frag_batch_ids[em_mask])
        x = self.node_encoder(input[0], fragments[em_mask])
        e = self.edge_encoder(input[0], fragments[em_mask], edge_index)

        # Pass shower fragment features through GNN
        index = torch.tensor(edge_index, dtype=torch.long, device=device)
        xbatch = torch.tensor(frag_batch_ids[em_mask], dtype=torch.long, device=device)
        gnn_output = self.track_gnn(x, index, e, xbatch)

        # Divide the particle GNN output out into different arrays (one per batch)
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(frag_batch_ids[em_mask], return_counts=True)[1]])
        bcids = [np.where(frag_batch_ids[em_mask] == b)[0] for b in range(len(counts))]
        beids = [np.where(frag_batch_ids[em_mask][edge_index[0]] == b)[0] for b in range(len(counts))]

        node_pred = [gnn_output['node_pred'][0][b] for b in bcids]
        edge_pred = [gnn_output['edge_pred'][0][b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        frags = [np.array([vids[c] for c in fragments[em_mask][b]]) for b in bcids]

        result.update({
            'track_fragments': [frags],
            'track_node_pred': [node_pred],
            'track_edge_pred': [edge_pred],
            'track_edge_index': [edge_index]
        })

        # Make shower group predictions based on the GNN output, use truth during training
        group_ids = []
        for b in range(len(counts)):
            if not len(frags[b]):
                group_ids.append(np.array([], dtype = np.int64))
            else:
                group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(frags[b])))

        result.update({'track_group_pred': [group_ids]})
        result['segmentation'][0] = segmentation

        return result


class GhostTrackClusteringLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(GhostTrackClusteringLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.spice_loss = ClusteringLoss(cfg)
        self.track_gnn_loss = GNNLoss(cfg, 'grappa_loss')
        self._num_classes = cfg['uresnet_lonely'].get('num_classes', 5)
        # Initialize the loss weights
        self.loss_config = cfg['full_chain_loss']
        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.track_gnn_weight = self.loss_config.get('track_gnn_weight', 0.0)

    def forward(self, out, label_seg, label_clustering):
        res_seg = self.uresnet_loss(out, label_seg)
        cluster_label = adapt_labels(out, label_seg, label_clustering)
        deghost = out['ghost'][0].argmax(dim=1) == 0
        segment_label = label_seg[0][deghost][:, -1]

        # Apply the CNN dense clustering loss to HE voxels only
        he_mask = segment_label < 4
        # sem_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,-1].view(-1,1)), dim=1)]
        #clust_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,5].view(-1,1),cluster_label[0][he_mask,4].view(-1,1)), dim=1)]
        clust_label = [cluster_label[0][he_mask].clone()]
        cnn_clust_output = {'embeddings':[out['embeddings'][0][he_mask]], 'seediness':[out['seediness'][0][he_mask]], 'margins':[out['margins'][0][he_mask]]}
        #cluster_label[0] = cluster_label[0][he_mask]
        res_cnn_clust = self.spice_loss(cnn_clust_output, clust_label)
        cnn_clust_acc, cnn_clust_loss = res_cnn_clust['accuracy'], res_cnn_clust['loss']

        # Apply the GNN particle clustering loss
        gnn_out = {
            'clusts':out['track_fragments'],
            'node_pred':out['track_node_pred'],
            'edge_pred':out['track_edge_pred'],
            'group_pred':out['track_group_pred'],
            'edge_index':out['track_edge_index'],
        }
        res_gnn_part = self.track_gnn_loss(gnn_out, cluster_label)

        # Combine the results
        accuracy = (res_seg['accuracy'] + res_cnn_clust['accuracy'] \
                    + res_gnn_part['accuracy'])/3.
        loss = self.segmentation_weight*res_seg['loss'] \
             + self.clustering_weight*res_cnn_clust['loss'] \
             + self.track_gnn_weight*res_gnn_part['loss']

        res = {}
        res.update(res_seg)
        res.update(res_cnn_clust)
        res['seg_accuracy'] = res_seg['accuracy']
        res['seg_loss'] = res_seg['loss']
        res['cnn_clust_accuracy'] = cnn_clust_acc
        res['cnn_clust_loss'] = cnn_clust_loss
        res['track_edge_loss'] = res_gnn_part['edge_loss']
        res['track_node_loss'] = res_gnn_part['node_loss']
        res['track_edge_accuracy'] = res_gnn_part['edge_accuracy']
        res['track_node_accuracy'] = res_gnn_part['node_accuracy']
        res['loss'] = loss
        res['accuracy'] = accuracy

        return res
