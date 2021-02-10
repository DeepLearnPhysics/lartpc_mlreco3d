from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from collections import defaultdict

from mlreco.models.layers.dbscan import distances

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.modular_meta import MetaLayerModel as GNN
from .gnn import node_encoder_construct, edge_encoder_construct

from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.clustercnn_se import ClusterCNN, ClusteringLoss

from .cluster_cnn import spice_loss_construct
from mlreco.models.cluster_full_gnn import ChainLoss as FullGNNLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss as EdgeGNNLoss
from mlreco.models.gnn.losses.grouping import *

from mlreco.utils.gnn.evaluation import node_assignment_score, primary_assignment
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.gnn.cluster import cluster_direction, get_cluster_batch
from mlreco.utils.deghosting import adapt_labels
from mlreco.models.layers.dbscan import DBSCANFragmenter, DBScanClusts2


def setup_chain_cfg(self, cfg):
    """
    Prepare both GhostChain2 and GhostChain2Loss
    Make sure config is logically sound with some basic checks
    """
    chain_cfg = cfg['chain']
    self.enable_ghost      = chain_cfg.get('enable_ghost', False)
    self.verbose           = chain_cfg.get('verbose', False)
    self.enable_uresnet    = chain_cfg.get('enable_uresnet', True)
    self.enable_ppn        = chain_cfg.get('enable_ppn', True)
    self.enable_cnn_clust  = chain_cfg.get('enable_cnn_clust', False)
    self.enable_gnn_shower = chain_cfg.get('enable_gnn_shower', False)
    self.enable_gnn_tracks = chain_cfg.get('enable_gnn_tracks', False)
    self.enable_gnn_int    = chain_cfg.get('enable_gnn_int', False)

    # whether to use CNN clustering or "dumb" DBSCAN clustering
    #self.use_dbscan_clust  = chain_cfg.get('use_dbscan_clust', False)
    # Whether to use PPN shower start information (GNN shower clustering step only)
    self.use_ppn_in_gnn    = chain_cfg.get('use_ppn_in_gnn', False)

    # Enforce basic logical order
    assert self.enable_uresnet or (self.enable_uresnet and self.enable_ppn) \
        or (self.enable_uresnet and self.enable_cnn_clust) \
        or (self.enable_uresnet and self.enable_gnn_shower) \
        or (self.enable_uresnet and self.enable_cnn_clust and self.enable_gnn_tracks) \
        or (self.enable_uresnet and self.enable_ppn and self.enable_gnn_shower and self.enable_gnn_int)
    assert (not self.use_ppn_in_gnn) or self.enable_ppn
    #assert self.use_dbscan_clust ^ self.enable_cnn_clust

    # Make sure the deghosting config is consistent
    if self.enable_ghost:
        assert cfg['uresnet_lonely']['ghost']
        if self.enable_ppn:
            assert cfg['ppn']['downsample_ghost']


class GhostChain2(torch.nn.Module):
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
            'fragment_clustering', 'node_encoder', 'edge_encoder', 'spice_loss', 'chain', 'dbscan_frag']

    def __init__(self, cfg):
        super(GhostChain2, self).__init__()

        # Configure the chain first
        setup_chain_cfg(self, cfg)

        # Initialize the UResNet+PPN modules
        if self.enable_uresnet:
            self.uresnet_lonely = UResNet(cfg)
            self.input_features = cfg['uresnet_lonely'].get('features', 1)

        if self.enable_ppn:
            self.ppn            = PPN(cfg)

        # CNN clustering
        self.min_frag_size = -1
        if self.enable_cnn_clust:
            self.spatial_embeddings    = ClusterCNN(cfg)
            # Fragment formation parameters
            self.frag_cfg     = cfg['fragment_clustering']
            self.s_thresholds = self.frag_cfg.get('s_thresholds', [0.0, 0.0, 0.0, 0.0])
            self.p_thresholds = self.frag_cfg.get('p_thresholds', [0.5, 0.5, 0.5, 0.5])
            self.cluster_all  = self.frag_cfg.get('cluster_all', True)
        elif self.enable_gnn_shower or self.enable_gnn_tracks or self.enable_gnn_int:
            # Initialize the DBSCAN fragmenter
            self.dbscan_frag = DBSCANFragmenter(cfg)
            #self.dbscan = DBScanClusts2(cfg)

        if self.enable_gnn_shower or self.enable_gnn_tracks or self.enable_gnn_int:
            # Initialize the geometric encoders
            self.node_encoder = node_encoder_construct(cfg)
            self.edge_encoder = edge_encoder_construct(cfg)

        if self.enable_gnn_shower:
            self.particle_gnn  = GNN(cfg['particle_edge_model'])
            self.min_frag_size = max(self.min_frag_size, cfg['particle_gnn'].get('node_min_size', -1))
            self.start_dir_max_dist = cfg['particle_edge_model'].get('start_dir_max_dist', 5)

        if self.enable_gnn_tracks:
            self.track_gnn  = GNN(cfg['track_edge_model'])
            self.min_frag_size = max(self.min_frag_size, cfg['track_gnn'].get('node_min_size', -1))

        if self.enable_gnn_int:
            # Initialize the GNN models
            self.inter_gnn     = GNN(cfg['interaction_edge_model'])


    def extract_fragment(self, input, result):
        """
        Extracting clustering predictions from CNN clustering output
        """
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


    def select_particle_in_group(self, result, counts, b, particles, part_primary_ids, node_pred, group_pred, fragments):
        """
        Merge fragments into particle instances, retain primary fragment id of each group
        """
        voxel_inds = counts[:b].sum().item()+np.arange(counts[b].item())
        primary_labels = primary_assignment(result[node_pred][0][b].detach().cpu().numpy(), result[group_pred][0][b])
        for g in np.unique(result[group_pred][0][b]):
            group_mask = np.where(result[group_pred][0][b] == g)[0]
            particles.append(voxel_inds[np.concatenate(result[fragments][0][b][group_mask])])
            primary_id = group_mask[primary_labels[group_mask]][0]
            part_primary_ids.append(primary_id)

    def run_gnn(self, gnn_model, input, result, frag_batch_ids, fragments, edge_index, x, e, labels):
        device = input[0].device

        # Pass fragment features through GNN
        index = torch.tensor(edge_index, dtype=torch.long, device=device)
        xbatch = torch.tensor(frag_batch_ids, dtype=torch.long, device=device)
        gnn_output = gnn_model(x, index, e, xbatch)

        # Divide the particle GNN output out into different arrays (one per batch)
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(frag_batch_ids, return_counts=True)[1]])
        bcids = [np.where(frag_batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(frag_batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        node_pred = [gnn_output['node_pred'][0][b] for b in bcids]
        edge_pred = [gnn_output['edge_pred'][0][b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        frags = [np.array([vids[c] for c in fragments[b]]) for b in bcids]

        result.update({
            labels['frags']: [frags],
            labels['node_pred']: [node_pred],
            labels['edge_pred']: [edge_pred],
            labels['edge_index']: [edge_index]
        })

        # Make shower group predictions based on the GNN output, use truth during training
        group_ids = []
        for b in range(len(counts)):
            if not len(frags[b]):
                group_ids.append(np.array([], dtype = np.int64))
            else:
                group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(frags[b])))

        result.update({labels['group_pred']: [group_ids]})

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

        if self.enable_cnn_clust:
            # Get fragment predictions from the CNN clustering algorithm
            spatial_embeddings_output = self.spatial_embeddings([input[0][:,:5]])
            result.update(spatial_embeddings_output)

            # Extract fragment predictions to input into the GNN
            fragments, frag_batch_ids, frag_seg = self.extract_fragment(input, result)
            semantic_labels = torch.argmax(result['segmentation'][0].detach(), dim=1).flatten()
        elif self.enable_gnn_shower or self.enable_gnn_tracks or self.enable_gnn_int:
            # Get the fragment predictions from the DBSCAN fragmenter
            semantic_labels = torch.argmax(result['segmentation'][0], dim=1).flatten().double()
            semantic_data = torch.cat((input[0][:,:4], semantic_labels.reshape(-1,1)), dim=1)
            fragments = self.dbscan_frag(semantic_data, result)
            frag_batch_ids = get_cluster_batch(input[0], fragments)
            frag_seg = np.empty(len(fragments), dtype=np.int32)
            for i, f in enumerate(fragments):
                vals, cnts = semantic_labels[f].unique(return_counts=True)
                assert len(vals) == 1
                frag_seg[i] = vals[torch.argmax(cnts)].item()

        if self.enable_gnn_shower:
            # Initialize a complete graph for edge prediction, get shower fragment and edge features
            em_mask = np.where(frag_seg == 0)[0]
            edge_index = complete_graph(frag_batch_ids[em_mask])
            x = self.node_encoder(input[0], fragments[em_mask])
            e = self.edge_encoder(input[0], fragments[em_mask], edge_index)

            if self.use_ppn_in_gnn:
                # Extract shower starts from PPN predictions (most likely prediction)
                ppn_points = result['points'][0].detach()
                ppn_feats = torch.empty((0,8), device=device, dtype=torch.float)
                for f in fragments[em_mask]:
                    scores = torch.softmax(ppn_points[f,3:5], dim=1)
                    argmax = torch.argmax(scores[:,-1])
                    start  = input[0][f][argmax,:3].float()+ppn_points[f][argmax,:3]+0.5
                    dir = cluster_direction(input[0][f][:,:3].float(), start, max_dist=self.start_dir_max_dist)
                    ppn_feats = torch.cat((ppn_feats, torch.cat([start, dir, scores[argmax]]).reshape(1,-1)), dim=0)

                x = torch.cat([x, ppn_feats], dim=1)

            self.run_gnn(self.particle_gnn, input, result, frag_batch_ids[em_mask], fragments[em_mask], edge_index, x, e,
                        {'frags': 'fragments', 'node_pred': 'frag_node_pred', 'edge_pred': 'frag_edge_pred', 'edge_index': 'frag_edge_index', 'group_pred': 'frag_group_pred'})


        if self.enable_gnn_tracks:
            # Initialize a complete graph for edge prediction, get track fragment and edge features
            em_mask = np.where(frag_seg == 1)[0]
            edge_index = complete_graph(frag_batch_ids[em_mask])
            x = self.node_encoder(input[0], fragments[em_mask])
            e = self.edge_encoder(input[0], fragments[em_mask], edge_index)

            self.run_gnn(self.track_gnn, input, result, frag_batch_ids[em_mask], fragments[em_mask], edge_index, x, e,
                        {'frags': 'track_fragments', 'node_pred': 'track_node_pred', 'edge_pred': 'track_edge_pred', 'edge_index': 'track_edge_index', 'group_pred': 'track_group_pred'})

        if self.enable_gnn_int:
            _, counts = torch.unique(input[0][:,3], return_counts=True)
            # Merge fragments into particle instances, retain primary fragment id of showers
            particles, part_primary_ids = [], []
            for b in range(len(counts)):
                # Append one particle per shower group
                self.select_particle_in_group(result, counts, b, particles, part_primary_ids, 'frag_node_pred', 'frag_group_pred', 'fragments')
                # Append one particle per track group
                if self.enable_gnn_tracks:
                    self.select_particle_in_group(result, counts, b, particles, part_primary_ids, 'track_node_pred', 'track_group_pred', 'track_fragments')

                # Append non-shower fragments as is
                mask = (frag_batch_ids == b) & (frag_seg != 0)
                if self.enable_gnn_tracks:
                    # Ignore tracks fragments as well
                    mask = mask & (frag_seg != 1)
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
            ppn_points = result['points'][0].detach()
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
                        p = voxel_inds[result['fragments'][0][part_batch_ids[i]][part_primary_ids[i]]]
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
            vids = np.concatenate([np.arange(n.item()) for n in counts])
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

            _, counts = torch.unique(input[0][:,3], return_counts=True)
            # Make interaction group predictions based on the GNN output, use truth during training
            group_ids = []
            for b in range(len(counts)):
                if not len(result['particles'][0][b]):
                    group_ids.append(np.array([], dtype = np.int64))
                else:
                    group_ids.append(node_assignment_score(result['inter_edge_index'][0][b], result['inter_edge_pred'][0][b].detach().cpu().numpy(), len(result['particles'][0][b])))

            result.update({'inter_group_pred': [group_ids]})

        return result

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        """
        # Pass the input data through UResNet+PPN (semantic segmentation + point prediction)
        result = {}
        if self.enable_uresnet:
            result = self.uresnet_lonely([input[0][:,:4+self.input_features]])
        if self.enable_ppn:
            ppn_input = {}
            ppn_input.update(result)
            ppn_input['ppn_feature_enc'] = ppn_input['ppn_feature_enc'][0]
            ppn_input['ppn_feature_dec'] = ppn_input['ppn_feature_dec'][0]
            if 'ghost' in ppn_input:
                ppn_input['ghost'] = ppn_input['ghost'][0]
            ppn_output = self.ppn(ppn_input)
            result.update(ppn_output)

        # The rest of the chain only needs 1 input feature
        if self.input_features > 1:
            input[0] = input[0][:, :-self.input_features+1]

        if self.enable_ghost:
            # Update input based on deghosting results
            deghost = result['ghost'][0].argmax(dim=1) == 0
            new_input = [input[0][deghost]]

            segmentation, points = result['segmentation'][0].clone(), result['points'][0].clone()

            deghost_result = {}
            deghost_result.update(result)
            deghost_result.pop('ghost')
            deghost_result['segmentation'][0] = result['segmentation'][0][deghost]
            deghost_result['points'][0] = result['points'][0][deghost]
            # Run the rest of the full chain
            full_chain_result = self.full_chain((new_input, deghost_result))
            full_chain_result['ghost'] = result['ghost']
        else:
            full_chain_result = self.full_chain((input, result))

        result.update(full_chain_result)

        if self.enable_ghost:
            result['segmentation'][0] = segmentation
            result['points'][0] = points

        return result


class GhostChain2Loss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(GhostChain2Loss, self).__init__()
        setup_chain_cfg(self, cfg)

        # Initialize loss components
        if self.enable_uresnet:
            self.uresnet_loss    = SegmentationLoss(cfg)
        if self.enable_ppn:
            self.ppn_loss        = PPNLoss(cfg)
        if self.enable_cnn_clust:
            self.spatial_embeddings_loss = ClusteringLoss(cfg)
        if self.enable_gnn_shower:
            self.particle_gnn_loss = FullGNNLoss(cfg, 'particle_gnn')
        if self.enable_gnn_tracks:
            self.track_gnn_loss = FullGNNLoss(cfg, 'track_gnn')
        if self.enable_gnn_int:
            self.inter_gnn_loss  = EdgeGNNLoss(cfg, 'interaction_gnn')

        # Initialize the loss weights
        self.loss_config = cfg['full_chain_loss']
        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.ppn_weight = self.loss_config.get('ppn_weight', 0.0)
        self.particle_gnn_weight = self.loss_config.get('particle_gnn_weight', 0.0)
        self.track_gnn_weight = self.loss_config.get('track_gnn_weight', 0.0)
        self.inter_gnn_weight = self.loss_config.get('inter_gnn_weight', 0.0)

    def forward(self, out, seg_label, ppn_label=None, cluster_label=None):
        res = {}
        accuracy, loss = 0., 0.

        if self.enable_uresnet:
            res_seg = self.uresnet_loss(out, seg_label)
            res.update(res_seg)
            res['seg_accuracy'] = res_seg['accuracy']
            res['seg_loss'] = res_seg['loss']
            accuracy += res_seg['accuracy']
            loss += self.segmentation_weight*res_seg['loss']

        if self.enable_ppn:
            # Apply the PPN loss
            res_ppn = self.ppn_loss(out, seg_label, ppn_label)
            res.update(res_ppn)
            res['ppn_accuracy'] = res_ppn['ppn_acc']
            res['ppn_loss'] = res_ppn['ppn_loss']

            accuracy += res_ppn['ppn_acc']
            loss += self.ppn_weight*res_ppn['ppn_loss']

        if self.enable_ghost:
            # Adapt to ghost points
            cluster_label = adapt_labels(out, seg_label, cluster_label)

            deghost = out['ghost'][0].argmax(dim=1) == 0
            #print("cluster_label", torch.unique(cluster_label[0][:, 7]), torch.unique(cluster_label[0][:, 6]), torch.unique(cluster_label[0][:, 5]))
            #result = self.full_chain_loss(out, res_seg, res_ppn, seg_label[0][deghost][:, -1], cluster_label)
            segment_label = seg_label[0][deghost][:, -1]
        else:
            #result = self.full_chain_loss(out, res_seg, res_ppn, seg_label[0][:, -1], cluster_label)
            segment_label = seg_label[0][:, -1]

        if self.enable_cnn_clust:
            # Apply the CNN dense clustering loss to HE voxels only
            he_mask = segment_label < 4
            # sem_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,-1].view(-1,1)), dim=1)]
            #clust_label = [torch.cat((cluster_label[0][he_mask,:4],cluster_label[0][he_mask,5].view(-1,1),cluster_label[0][he_mask,4].view(-1,1)), dim=1)]
            clust_label = [cluster_label[0][he_mask].clone()]
            cnn_clust_output = {'embeddings':[out['embeddings'][0][he_mask]], 'seediness':[out['seediness'][0][he_mask]], 'margins':[out['margins'][0][he_mask]]}
            #cluster_label[0] = cluster_label[0][he_mask]
            res_cnn_clust = self.spatial_embeddings_loss(cnn_clust_output, clust_label)
            res.update(res_cnn_clust)
            res['cnn_clust_accuracy'] = res_cnn_clust['accuracy']
            res['cnn_clust_loss'] = res_cnn_clust['loss']

            accuracy += res_cnn_clust['accuracy']
            loss += self.clustering_weight*res_cnn_clust['loss']

        if self.enable_gnn_shower:
            # Apply the GNN particle clustering loss
            gnn_out = {
                'clusts':out['fragments'],
                'node_pred':out['frag_node_pred'],
                'edge_pred':out['frag_edge_pred'],
                'group_pred':out['frag_group_pred'],
                'edge_index':out['frag_edge_index'],
            }
            res_gnn_part = self.particle_gnn_loss(gnn_out, cluster_label)
            res['frag_edge_loss'] = res_gnn_part['edge_loss']
            res['frag_node_loss'] = res_gnn_part['node_loss']
            res['frag_edge_accuracy'] = res_gnn_part['edge_accuracy']
            res['frag_node_accuracy'] = res_gnn_part['node_accuracy']

            accuracy += res_gnn_part['accuracy']
            loss += self.particle_gnn_weight*res_gnn_part['loss']

        if self.enable_gnn_tracks:
            # Apply the GNN particle clustering loss
            gnn_out = {
                'clusts':out['track_fragments'],
                'node_pred':out['track_node_pred'],
                'edge_pred':out['track_edge_pred'],
                'group_pred':out['track_group_pred'],
                'edge_index':out['track_edge_index'],
            }
            res_gnn_track = self.track_gnn_loss(gnn_out, cluster_label)
            res['track_edge_loss'] = res_gnn_track['edge_loss']
            res['track_node_loss'] = res_gnn_track['node_loss']
            res['track_edge_accuracy'] = res_gnn_track['edge_accuracy']
            res['track_node_accuracy'] = res_gnn_track['node_accuracy']

            accuracy += res_gnn_track['accuracy']
            loss += self.track_gnn_weight*res_gnn_track['loss']

        if self.enable_gnn_int:
            # Apply the GNN interaction grouping loss
            gnn_out = {
                'clusts':out['particles'],
                'edge_pred':out['inter_edge_pred'],
                'edge_index':out['inter_edge_index']
            }
            res_gnn_inter = self.inter_gnn_loss(gnn_out, cluster_label, None)
            res['inter_edge_loss'] = res_gnn_inter['loss']
            res['inter_edge_accuracy'] = res_gnn_inter['accuracy']

            accuracy += res_gnn_inter['accuracy']
            loss += self.inter_gnn_weight*res_gnn_inter['loss']

        # Combine the results
        accuracy /= int(self.enable_uresnet) + int(self.enable_ppn) + int(self.enable_gnn_shower) \
                    + int(self.enable_gnn_int) + int(self.enable_gnn_tracks) + int(self.enable_cnn_clust)

        res['loss'] = loss
        res['accuracy'] = accuracy

        if self.verbose:
            if self.enable_uresnet:
                print('Segmentation Accuracy: {:.4f}'.format(res_seg['accuracy']))
            if self.enable_ppn:
                print('PPN Accuracy: {:.4f}'.format(res_ppn['ppn_acc']))
            if self.enable_cnn_clust:
                print('Clustering Accuracy: {:.4f}'.format(res_cnn_clust['accuracy']))
            if self.enable_gnn_shower:
                print('Shower fragment clustering accuracy: {:.4f}'.format(res_gnn_part['edge_accuracy']))
                print('Shower primary prediction accuracy: {:.4f}'.format(res_gnn_part['node_accuracy']))
            if self.enable_gnn_tracks:
                print('Track fragment clustering accuracy: {:.4f}'.format(res_gnn_track['edge_accuracy']))
            if self.enable_gnn_int:
                print('Interaction grouping accuracy: {:.4f}'.format(res_gnn_inter['accuracy']))

        return res
