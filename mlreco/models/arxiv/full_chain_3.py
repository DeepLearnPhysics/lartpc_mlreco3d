import torch
import numpy as np
from collections import defaultdict

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.modular_nnconv import NNConvModel as GNN
from .gnn import node_encoder_construct, edge_encoder_construct

from mlreco.models.uresnet_lonely import SegmentationLoss
from mlreco.models.ppn import PPNLoss
from .cluster_cnn import spice_loss_construct
from mlreco.models.cluster_full_gnn import ChainLoss as FullGNNLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss as EdgeGNNLoss
from mlreco.models.gnn.losses.grouping import *

from mlreco.utils.gnn.evaluation import node_assignment_score, primary_assignment
from mlreco.utils.gnn.network import complete_graph

class FullChain(torch.nn.Module):
    """
    Driver class for the end-to-end reconstruction chain
    1) UResNet
        1) Semantic - for point classification
        2) PPN - for particle point locations
        3) Fragment - to form particle fragments
    2) GNN
        1) Particle - to group showers and identify their primaries
        2) Interaction - to group particles

    For use in config:
    model:
      name: full_chain
      modules:
        full_cnn:
          <full CNN global parameters, see mlreco/models/chain/full_cnn.py>
        network_base:
          <uresnet archicteture core parameters, see mlreco/models/layers/base.py>
        uresnet_encoder:
          <uresnet encoder parameters, see mlreco/models/layers/uresnet.py>
        segmentation_decoder:
          <uresnet segmention decoder paraters, see mlreco/models/chain/full_cnn.py>
        seediness_decoder:
          <uresnet seediness decoder paraters, see mlreco/models/chain/full_cnn.py>
        embedding_decoder:
          <uresnet embedding decoder paraters, see mlreco/models/chain/full_cnn.py>
        particle_gnn:
          node_type    : <fragment semantic class to include in the particle grouping task>
          node_min_size: <fragment size to include in the particle grouping task>
        interaction_gnn:
          node_type    : <particle semantic class to include in the interaction grouping task>
          node_min_size: <particle size to include in the interaction grouping task>
        particle_edge_model:
          <GNN parameters for particle clustering, see mlreco/models/gnn/modular_nnconv.py>
        interaction_edge_model:
          <GNN parameters for interaction clustering, see mlreco/models/gnn/modular_nnconv.py>
        full_chain_loss:
          name: <name of the loss function for the CNN fragment clustering model>
          spatial_size: <spatial size of input images>
          segmentation_weight: <relative weight of the segmentation loss>
          clustering_weight: <relative weight of the clustering loss>
          seediness_weight: <relative weight of the seediness loss>
          embedding_weight: <relative weight of the embedding loss>
          smoothing_weight: <relative weight of the smoothing loss>
          ppn_weight: <relative weight of the ppn loss>
          particle_gnn_weight: <relative weight of the particle gnn loss>
          interaction_gnn_weight: <relative weight of the interaction gnn loss>
    """

    MODULES = ['full_cnn', 'network_base', 'uresnet_encoder', 'segmentation_decoder', 'seediness_decoder',
            'embedding_decoder', 'particle_gnn', 'interaction_gnn', 'particle_edge_model',
            'interaction_edge_model', 'full_chain_loss', 'ppn', 'node_encoder', 'edge_encoder', 'full_chain']

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()

        # Initialize the full CNN model (includes UResNet+PPN+Fragmentation)
        self.full_cnn = FullCNN(cfg)

        # Initialize the geometric encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)

        # Initialize the GNN models
        self.particle_gnn = GNN(cfg['particle_edge_model'])
        self.inter_gnn    = GNN(cfg['interaction_edge_model'])
        self.min_frag_size = cfg['particle_gnn'].get('node_min_size', -1)

        self.train_stage = cfg['full_chain'].get('train', True)

    def forward(self, input):
        '''
        Forward for full reconstruction chain.

        INPUTS:
            - input (N x 5 Tensor): Input data [x, y, z, batch_id, val]

        RETURNS:
            - result (tuple of dicts): (cnn_result, gnn_result)
        '''
        # Run all CNN modules
        device = input[0].device
        result = self.full_cnn(input)

        # Extract fragment predictions to input into the GNN
        batch_labels = input[0][:,3]
        fragments = []
        frag_batch_ids = []
        if self.train_stage:
            fragment_labels = input[0][:,5]
            semantic_labels = input[0][:,-1]
            for batch_id in batch_labels.unique():
                batch_mask = torch.nonzero(batch_labels == batch_id).flatten()
                fragment_batch = fragment_labels[batch_mask]
                for fid in fragment_batch.unique():
                    if torch.sum(fragment_batch == fid) < self.min_frag_size:
                        continue
                    fragments.append(batch_mask[fragment_batch == fid])
                    frag_batch_ids.append(int(batch_id))
        else:
            semantic_labels = torch.argmax(result['segmentation'][0], dim=1).flatten()
            for batch_id in batch_labels.unique():
                for s in semantic_labels.unique():
                    if s > 3: continue
                    mask = torch.nonzero((batch_labels == batch_id) & (semantic_labels == s)).flatten()
                    pred_labels = fit_predict(result['embeddings'][0][mask],
                        result['seediness'][0][mask], result['margins'][0][mask], gaussian_kernel)
                    for c in pred_labels.unique():
                        if torch.sum(pred_labels == c) < self.min_frag_size:
                            continue
                        fragments.append(mask[pred_labels == c])
                        frag_batch_ids.append(int(batch_id))

        fragments = np.array([f.detach().cpu().numpy() for f in fragments if len(f)])
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = semantic_labels[f].unique(return_counts=True)
            #assert len(vals) == 1 # PROBLEM HERE, ?????
            frag_seg[i] = vals[torch.argmax(cnts)].item()

        # Initialize a complete graph for edge prediction, get shower fragment and edge features
        em_mask = np.where(frag_seg == 0)[0]
        edge_index = complete_graph(frag_batch_ids[em_mask])
        x = self.node_encoder(input[0], fragments[em_mask])
        e = self.edge_encoder(input[0], fragments[em_mask], edge_index)

        # Extract shower starts from PPN predictions (most likely prediction)
        ppn_feats = torch.empty((0,6), device=device, dtype=torch.float)
        for f in fragments[em_mask]:
            scores = torch.softmax(result['points'][0][f][:,3:5], dim=1)
            argmax = torch.argmax(scores[:,-1])
            start  = input[0][f][argmax,:3].float()+result['points'][0][f][argmax,:3]+0.5
            dir = (input[0][f][:,:3].float()-start).mean(dim=0)
            if dir.norm():
                dir = dir/dir.norm()
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
                if self.train_stage:
                    batch_group_ids = []
                    for i, f in enumerate(frags[b]):
                        vals, cnts = input[0][f,6].unique(return_counts=True)
                        #assert len(vals) == 1 # PROBLEM HERE
                        batch_group_ids.append(vals[torch.argmax(cnts)].item())
                    group_ids.append(np.array(batch_group_ids, dtype=np.int32))
                else:
                    group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(frags[b])))

        result.update({'frag_group_pred': [group_ids]})

        # Merge fragments into particle instances, retain primary id of showers
        non_em_mask = np.where((frag_seg != 0) & (frag_seg != 4))[0]
        particles = fragments[non_em_mask].tolist()
        part_batch_ids = frag_batch_ids[non_em_mask]
        part_primary_ids = -1*np.ones(len(non_em_mask), dtype=np.int32)
        part_seg = frag_seg[non_em_mask]
        for b in range(len(counts)):
            batch_mask = np.where(frag_batch_ids[em_mask] == b)[0]
            primary_labels = primary_assignment(node_pred[b].detach().cpu().numpy(), group_ids[b])
            for g in np.unique(group_ids[b]):
                group_mask = np.where(group_ids[b] == g)[0]
                particles.append(np.concatenate(frags[b][group_mask]))
                group_batch = np.unique(frag_batch_ids[em_mask][batch_mask][group_mask])
                primary_id = group_mask[primary_labels[group_mask]]
                assert len(group_batch) == 1
                part_batch_ids = np.concatenate((part_batch_ids, group_batch[0].reshape(1)))
                part_seg = np.concatenate((part_seg, [0]))
                part_primary_ids = np.concatenate((part_primary_ids, primary_id))

        part_order = np.argsort(part_batch_ids)
        particles = np.array(particles)[part_order]
        part_batch_ids = part_batch_ids[part_order]
        part_seg = part_seg[part_order]
        part_primary_ids = part_primary_ids[part_order]

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
            else:
                if part_seg[i] == 0:
                    p = frags[part_batch_ids[i]][part_primary_ids[i]]
                scores = torch.softmax(result['points'][0][p][:,3:5], dim=1)
                argmax = torch.argmax(scores[:,-1])
                start = end = input[0][p][argmax,:3].float()+result['points'][0][p][argmax,:3]+0.5
                dir = (input[0][p][:,:3].float()-start).mean(dim=0)

            if dir.norm():
                dir = dir/dir.norm()
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


class FullChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(FullChainLoss, self).__init__()

        # Initialize loss components
        self.loss_config = cfg['full_chain_loss']
        self.segmentation_loss = SegmentationLoss({'uresnet_lonely':cfg['full_cnn']})
        self.ppn_loss = PPNLoss(cfg)
        self.spice_loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.spice_loss = spice_loss_construct(self.spice_loss_name)
        self.spice_loss = self.spice_loss(cfg, name='full_chain_loss')
        self.particle_gnn_loss = FullGNNLoss(cfg, 'particle_gnn')
        self.inter_gnn_loss  = EdgeGNNLoss(cfg, 'interaction_gnn')

        # Initialize the loss weights
        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        self.ppn_weight = self.loss_config.get('ppn_weight', 0.0)
        self.particle_gnn_weight = self.loss_config.get('particle_gnn_weight', 0.0)
        self.inter_gnn_weight = self.loss_config.get('particle_gnn_weight', 0.0)

    def forward(self, out, cluster_label, ppn_label):
        '''
        Forward propagation for FullChain

        INPUTS:
            - out (dict): result from forwarding three-tailed UResNet, with
            1) segmenation decoder 2) clustering decoder 3) seediness decoder,
            and PPN attachment to the segmentation branch.

            - cluster_label (list of Tensors): input data tensor of shape N x 10
              In row-index order:
              1. x coordinates
              2. y coordinates
              3. z coordinates
              4. batch indices
              5. energy depositions
              6. fragment labels
              7. group labels
              8. interaction labels
              9. neutrino labels
              10. segmentation labels (0-5, includes ghosts)

            - ppn_label (list of Tensors): particle labels for ppn ground truth
        '''

        # Apply the segmenation loss
        coords = cluster_label[0][:, :4]
        segment_label = cluster_label[0][:, -1]
        segment_label_tensor = torch.cat((coords, segment_label.reshape(-1,1)), dim=1)
        res_seg = self.segmentation_loss(out, [segment_label_tensor])
        seg_acc, seg_loss = res_seg['accuracy'], res_seg['loss']

        # Apply the PPN loss
        res_ppn = self.ppn_loss(out, [segment_label_tensor], ppn_label)

        # Apply the CNN dense clustering loss
        fragment_label = cluster_label[0][:, 5]
        batch_idx = coords[:, -1].unique()
        res_cnn_clust = defaultdict(int)
        for bidx in batch_idx:
            # Get the loss input for this batch
            batch_mask = coords[:, -1] == bidx
            highE_mask = segment_label[batch_mask] != 4
            embedding_batch_highE = out['embeddings'][0][batch_mask][highE_mask]
            margins_batch_highE = out['margins'][0][batch_mask][highE_mask]
            seed_batch_highE = out['seediness'][0][batch_mask][highE_mask]
            slabels_highE = segment_label[batch_mask][highE_mask]
            clabels_batch_highE = fragment_label[batch_mask][highE_mask]

            # Get the clustering loss, append results
            loss_class, acc_class = self.spice_loss.combine_multiclass(
                embedding_batch_highE, margins_batch_highE,
                seed_batch_highE, slabels_highE, clabels_batch_highE)

            loss, accuracy = 0, 0
            for key, val in loss_class.items():
                res_cnn_clust[key+'_loss'] += (sum(val) / len(val))
                loss += (sum(val) / len(val))
            for key, val in acc_class.items():
                res_cnn_clust[key+'_accuracy'] += val
                accuracy += val

            res_cnn_clust['loss'] += loss/len(loss_class.values())/len(batch_idx)
            res_cnn_clust['accuracy'] += accuracy/len(acc_class.values())/len(batch_idx)

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
        res_gnn_inter = self.inter_gnn_loss(gnn_out, cluster_label, None)

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
        res['seg_accuracy'] = seg_acc
        res['seg_loss'] = seg_loss
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
