from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from collections import defaultdict

from mlreco.models.layers.dbscan import distances

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.message_passing.meta import MetaLayerModel as GNN
from .gnn import node_encoder_construct, edge_encoder_construct

from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.clustercnn_se import ClusterCNN, ClusteringLoss

from .cluster_cnn import spice_loss_construct
from mlreco.models.grappa import GNNLoss
from mlreco.models.gnn.losses.node_grouping import *

from mlreco.utils.gnn.evaluation import node_assignment_score, primary_assignment
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.gnn.cluster import cluster_direction, get_cluster_batch
from mlreco.utils.deghosting import adapt_labels


class GhostChain(torch.nn.Module):
    """
    Full Chain using deghosting input
    Based on mlreco.models.full_chain_4
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (float,), (3, 1)],
    # ]
    # MODULES = ['spatial_embeddings', 'uresnet_lonely'] + ClusterCNN.MODULES
    MODULES = ['full_cnn', 'network_base', 'uresnet_encoder', 'segmentation_decoder',
            'embedding_decoder', 'particle_gnn', 'interaction_gnn', 'particle_edge_model',
            'interaction_edge_model', 'full_chain_loss', 'uresnet_lonely', 'ppn', 'uresnet',
            'fragment_clustering', 'node_encoder', 'edge_encoder', 'spice_loss', 'chain']

    def __init__(self, cfg):
        super(GhostChain, self).__init__()
        self.uresnet_lonely = UResNet(cfg)
        self.ppn            = PPN(cfg)
        # Initialize the UResNet+PPN modules
        self.spatial_embeddings    = ClusterCNN(cfg)

        # Fragment formation parameters
        self.frag_cfg     = cfg['fragment_clustering']
        self.s_thresholds = self.frag_cfg.get('s_thresholds', [0.0, 0.0, 0.0, 0.0])
        self.p_thresholds = self.frag_cfg.get('p_thresholds', [0.5, 0.5, 0.5, 0.5])
        self.cluster_all  = self.frag_cfg.get('cluster_all', True)

        # Initialize the geometric encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)

        # Initialize the GNN models
        self.particle_gnn  = GNN(cfg['grappa_shower'])
        self.inter_gnn     = GNN(cfg['grappa_inter'])
        self.min_frag_size = cfg['grappa_shower']['base'].get('node_min_size', -1)
        self._use_ppn_shower = cfg['grappa_shower']['base'].get('use_ppn_shower', False)

        self.input_features = cfg['uresnet_lonely'].get('features', 1)

        # self.loss_cfg = cfg['full_chain_loss']
        # self.ppn_active = self.loss_cfg.get('ppn_weight', 0.0) > 0.
        #

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

    def full_chain(self, x):
        '''
        Forward for full reconstruction chain.

        INPUTS:
            - input (N x 5 Tensor): Input data [x, y, z, batch_id, val]

        RETURNS:
            - result (tuple of dicts): (cnn_result, gnn_result)
        '''
        input, result = x
        device = input[0].device

        # Get fragment predictions from the CNN clustering algorithm
        spatial_embeddings_output = self.spatial_embeddings([input[0][:,:5]])
        result.update(spatial_embeddings_output)

        # Extract fragment predictions to input into the GNN
        fragments, frag_batch_ids, frag_seg = self.extract_fragment(input, result)
        semantic_labels = torch.argmax(result['segmentation'][0].detach(), dim=1).flatten()

        # Initialize a complete graph for edge prediction, get shower fragment and edge features
        em_mask = np.where(frag_seg == 0)[0]
        edge_index = complete_graph(frag_batch_ids[em_mask])
        x = self.node_encoder(input[0], fragments[em_mask])
        e = self.edge_encoder(input[0], fragments[em_mask], edge_index)

        # Extract shower starts from PPN predictions (most likely prediction)
        ppn_points = result['points'][0].detach()
        if self._use_ppn_shower:
            ppn_feats = torch.empty((0,6), device=device, dtype=torch.float)
            for f in fragments[em_mask]:
                scores = torch.softmax(ppn_points[f,3:5], dim=1)
                argmax = torch.argmax(scores[:,-1])
                start  = input[0][f][argmax,:3].float()+ppn_points[f][argmax,:3]+0.5
                dir = cluster_direction(input[0][f][:,:3].float(), start, max_dist=5)
                ppn_feats = torch.cat((ppn_feats, torch.cat([start, dir]).reshape(1,-1)), dim=0)

            x = torch.cat([x, ppn_feats], dim=1)

        # Pass shower fragment features through GNN
        index = torch.tensor(edge_index, dtype=torch.long, device=device)
        xbatch = torch.tensor(frag_batch_ids[em_mask], dtype=torch.long, device=device)
        gnn_output = self.particle_gnn(x, index, e, xbatch)

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
            'fragments': [frags],
            'frag_node_pred': [node_pred],
            'frag_edge_pred': [edge_pred],
            'frag_edge_index': [edge_index]
        })

        # Make shower group predictions based on the GNN output, use truth during training
        group_ids = []
        for b in range(len(counts)):
            if not len(frags[b]):
                group_ids.append(np.array([], dtype = np.int64))
            else:
                group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(frags[b])))

        result.update({'frag_group_pred': [group_ids]})

        # Merge fragments into particle instances, retain primary fragment id of showers
        particles, part_primary_ids = [], []
        for b in range(len(counts)):
            # Append one particle per shower group
            voxel_inds = counts[:b].sum().item()+np.arange(counts[b].item())
            primary_labels = primary_assignment(node_pred[b].detach().cpu().numpy(), group_ids[b])
            for g in np.unique(group_ids[b]):
                group_mask = np.where(group_ids[b] == g)[0]
                particles.append(voxel_inds[np.concatenate(frags[b][group_mask])])
                primary_id = group_mask[primary_labels[group_mask]][0]
                part_primary_ids.append(primary_id)

            # Append non-shower fragments as is
            mask = (frag_batch_ids == b) & (frag_seg != 0)
            particles.extend(fragments[mask])
            part_primary_ids.extend(-np.ones(np.sum(mask)))

        particles = np.array(particles)
        part_batch_ids = get_cluster_batch(input[0], particles)
        part_primary_ids = np.array(part_primary_ids, dtype=np.int32)
        part_seg = np.empty(len(particles), dtype=np.int32)
        for i, p in enumerate(particles):
            vals, cnts = semantic_labels[p].unique(return_counts=True)
            assert len(vals) == 1
            part_seg[i] = vals[torch.argmax(cnts)].item()

        # Initialize a complete graph for edge prediction, get particle and edge features
        edge_index = complete_graph(part_batch_ids)
        x = self.node_encoder(input[0], particles, )
        e = self.edge_encoder(input[0], particles, edge_index)

        # Extract intersting points for particles, add semantic class, mean value and rms value
        # - For showers, take the most likely PPN voxel of the primary fragment
        # - For tracks, take the points furthest removed from each other (why not ?)
        # - For Michel and Delta, take the most likely PPN voxel
        ppn_feats = torch.empty((0,12), device=input[0].device, dtype=torch.float)
        for i, p in enumerate(particles):
            if part_seg[i] == 1:
                from mlreco.utils import local_cdist
                dist_mat = local_cdist(input[0][p,:3], input[0][p,:3])
                idx = torch.argmax(dist_mat)
                start_id, end_id = int(idx/len(p)), int(idx%len(p))
                start, end = input[0][p[start_id],:3].float(), input[0][p[end_id],:3].float()
                dir = end-start
                if dir.norm():
                    dir = dir/dir.norm()
            else:
                if part_seg[i] == 0:
                    voxel_inds = counts[:part_batch_ids[i]].sum().item()+np.arange(counts[part_batch_ids[i]].item())
                    p = voxel_inds[frags[part_batch_ids[i]][part_primary_ids[i]]]
                scores = torch.softmax(ppn_points[p,3:5], dim=1)
                argmax = torch.argmax(scores[:,-1])
                start = end = input[0][p][argmax,:3].float()+ppn_points[p][argmax,:3]+0.5
                dir = cluster_direction(input[0][p][:,:3].float(), start, max_dist=5)

            sem_type = torch.tensor([part_seg[i]], dtype=torch.float, device=device)
            values = torch.cat((input[0][p,4].mean().reshape(1), input[0][p,4].std().reshape(1))).float()
            ppn_feats = torch.cat((ppn_feats, torch.cat([values, sem_type.reshape(1), start, end, dir]).reshape(1,-1)), dim=0)

        x = torch.cat([x, ppn_feats], dim=1)

        # Pass particles through interaction clustering
        index = torch.tensor(edge_index, dtype=torch.long, device=device)
        xbatch = torch.tensor(part_batch_ids, dtype=torch.long, device=device)
        gnn_output = self.inter_gnn(x, index, e, xbatch)

        # Divide the interaction GNN output out into different arrays (one per batch)
        cids = np.concatenate([np.arange(n) for n in np.unique(part_batch_ids, return_counts=True)[1]])
        bcids = [np.where(part_batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(part_batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        edge_pred = [gnn_output['edge_pred'][0][b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        particles = [np.array([vids[c] for c in particles[b]]) for b in bcids]

        result.update({
            'particles': [particles],
            'inter_edge_pred': [edge_pred],
            'inter_edge_index': [edge_index]
        })

        return result

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        """
        # Pass the input data through UResNet+PPN (semantic segmentation + point prediction)
        result = self.uresnet_lonely([input[0][:,:4+self.input_features]])
        ppn_input = {}
        ppn_input.update(result)
        ppn_input['ppn_feature_enc'] = ppn_input['ppn_feature_enc'][0]
        ppn_input['ppn_feature_dec'] = ppn_input['ppn_feature_dec'][0]
        if 'ghost' in ppn_input:
            ppn_input['ghost'] = ppn_input['ghost'][0]
        ppn_output = self.ppn(ppn_input)
        result.update(ppn_output)

        # Update input based on deghosting results
        deghost = result['ghost'][0].argmax(dim=1) == 0
        # Also remove any extra features
        new_input = [input[0][deghost]]

        segmentation, points = result['segmentation'][0].clone(), result['points'][0].clone()

        deghost_result = {}
        deghost_result.update(result)
        deghost_result['segmentation'][0] = result['segmentation'][0][deghost]
        deghost_result['points'][0] = result['points'][0][deghost]
        # Run the rest of the full chain

        full_chain_result = self.full_chain((new_input, deghost_result))

        result.update(full_chain_result)
        result['segmentation'][0] = segmentation
        result['points'][0] = points

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
        self.uresnet_loss    = SegmentationLoss(cfg)
        self.ppn_loss        = PPNLoss(cfg)

        # Initialize loss components
        self.spatial_embeddings_loss = ClusteringLoss(cfg)
        self.particle_gnn_loss = GNNLoss(cfg, 'grappa_shower_loss')
        self.inter_gnn_loss  = GNNLoss(cfg, 'grappa_inter_loss')

        # Initialize the loss weights
        self.loss_config = cfg['full_chain_loss']
        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.ppn_weight = self.loss_config.get('ppn_weight', 0.0)
        self.particle_gnn_weight = self.loss_config.get('particle_gnn_weight', 0.0)
        self.inter_gnn_weight = self.loss_config.get('inter_gnn_weight', 0.0)

    def full_chain_loss(self, out, res_seg, res_ppn, segment_label, cluster_label):
        # Apply the CNN dense clustering loss to HE voxels only
        he_mask = segment_label < 4
        # sem_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,-1].view(-1,1)), dim=1)]
        #clust_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,5].view(-1,1),cluster_label[0][he_mask,4].view(-1,1)), dim=1)]
        clust_label = [cluster_label[0][he_mask].clone()]
        cnn_clust_output = {'embeddings':[out['embeddings'][0][he_mask]], 'seediness':[out['seediness'][0][he_mask]], 'margins':[out['margins'][0][he_mask]]}
        #cluster_label[0] = cluster_label[0][he_mask]
        res_cnn_clust = self.spatial_embeddings_loss(cnn_clust_output, clust_label)
        cnn_clust_acc, cnn_clust_loss = res_cnn_clust['accuracy'], res_cnn_clust['loss']

        # Apply the GNN particle clustering loss
        gnn_out = {
            'clusts':out['fragments'],
            'node_pred':out['frag_node_pred'],
            'edge_pred':out['frag_edge_pred'],
            'group_pred':out['frag_group_pred'],
            'edge_index':out['frag_edge_index'],
        }
        res_gnn_part = self.particle_gnn_loss(gnn_out, cluster_label)

        # Apply the GNN interaction grouping loss
        gnn_out = {
            'clusts':out['particles'],
            'edge_pred':out['inter_edge_pred'],
            'edge_index':out['inter_edge_index']
        }
        res_gnn_inter = self.inter_gnn_loss(gnn_out, cluster_label)

        # Combine the results
        accuracy = (res_seg['accuracy'] + res_ppn['ppn_acc'] + res_cnn_clust['accuracy'] \
                    + res_gnn_part['accuracy'] + res_gnn_inter['accuracy'])/5.
        loss = self.segmentation_weight*res_seg['loss'] \
             + self.ppn_weight*res_ppn['ppn_loss'] \
             + self.clustering_weight*res_cnn_clust['loss'] \
             + self.particle_gnn_weight*res_gnn_part['loss'] \
             + self.inter_gnn_weight*res_gnn_inter['loss']

        res = {}
        res.update(res_seg)
        res.update(res_ppn)
        res.update(res_cnn_clust)
        res['seg_accuracy'] = res_seg['accuracy']
        res['seg_loss'] = res_seg['loss']
        res['ppn_accuracy'] = res_ppn['ppn_acc']
        res['ppn_loss'] = res_ppn['ppn_loss']
        res['cnn_clust_accuracy'] = cnn_clust_acc
        res['cnn_clust_loss'] = cnn_clust_loss
        res['frag_edge_loss'] = res_gnn_part['edge_loss']
        res['frag_node_loss'] = res_gnn_part['node_loss']
        res['frag_edge_accuracy'] = res_gnn_part['edge_accuracy']
        res['frag_node_accuracy'] = res_gnn_part['node_accuracy']
        res['inter_edge_loss'] = res_gnn_inter['loss']
        res['inter_edge_accuracy'] = res_gnn_inter['accuracy']
        res['loss'] = loss
        res['accuracy'] = accuracy

        print('Segmentation Accuracy: {:.4f}'.format(res_seg['accuracy']))
        print('PPN Accuracy: {:.4f}'.format(res_ppn['ppn_acc']))
        print('Clustering Accuracy: {:.4f}'.format(res_cnn_clust['accuracy']))
        print('Shower fragment clustering accuracy: {:.4f}'.format(res_gnn_part['edge_accuracy']))
        print('Shower primary prediction accuracy: {:.4f}'.format(res_gnn_part['node_accuracy']))
        print('Interaction grouping accuracy: {:.4f}'.format(res_gnn_inter['accuracy']))

        return res

    def forward(self, out, seg_label, cluster_label, ppn_label):
        res_seg = self.uresnet_loss(out, seg_label)
        seg_acc, seg_loss = res_seg['accuracy'], res_seg['loss']

        # Apply the PPN loss
        res_ppn = self.ppn_loss(out, seg_label, ppn_label)

        # Adapt to ghost points
        cluster_label = adapt_labels(out, seg_label, cluster_label)

        deghost = out['ghost'][0].argmax(dim=1) == 0
        #print("cluster_label", torch.unique(cluster_label[0][:, 7]), torch.unique(cluster_label[0][:, 6]), torch.unique(cluster_label[0][:, 5]))
        result = self.full_chain_loss(out, res_seg, res_ppn, seg_label[0][deghost][:, -1], cluster_label)

        return result
