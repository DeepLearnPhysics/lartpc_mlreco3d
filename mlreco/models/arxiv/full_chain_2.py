import torch
import numpy as np
from collections import defaultdict

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.modular_nnconv import NNConvModel as GNN
from mlreco.utils.gnn.evaluation import node_assignment_score

from mlreco.models.uresnet_lonely import SegmentationLoss
from mlreco.models.ppn import PPNLoss
from .cluster_cnn import spice_loss_construct
from mlreco.models.cluster_full_gnn import ChainLoss as FullGNNLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss as EdgeGNNLoss
from mlreco.models.gnn.losses.grouping import *

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
            'interaction_edge_model', 'full_chain_loss', 'ppn']

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()

        # Initialize the full CNN model (includes UResNet+PPN+Fragmentation)
        self.full_cnn = FullCNN(cfg)

        # Initialize GNN feature extractor (from fragment voxel embeddings to global features)
        num_edge_feats = cfg['full_cnn']['num_gnn_features']
        self.particle_edge_net = EdgeFeatureNet(num_edge_feats, num_edge_feats)
        self.inter_edge_net = EdgeFeatureNet(num_edge_feats, num_edge_feats)

        # Initialize the GNN models
        self.particle_gnn = GNN(cfg['particle_edge_model'])
        self.inter_gnn    = GNN(cfg['interaction_edge_model'])

        self.train_stage = cfg['full_chain']['train']

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

        # Extract fragment prediction and features to input into the GNN
        frag_args = {}
        frag_args['fit_predict']  = fit_predict
        frag_args['kernel_func']  = gaussian_kernel
        frag_args['coords']       = input[0][:,:4]
        for key in ['features_gnn', 'embeddings', 'seediness', 'margins']:
            frag_args[key] = result[key][0]
        if self.train_stage:
            frag_args['fragment_labels'] = input[0][:,5]
            frag_args['semantic_labels'] = input[0][:,-1]
        else:
            frag_args['semantic_labels'] = torch.argmax(result['segmentation'][0], dim=1).flatten()

        x, xbatch, fragments = get_gnn_input(train=self.train_stage, **frag_args)
        fragment_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = frag_args['semantic_labels'][f].unique(return_counts=True)
            #assert len(vals) == 1 # PROBLEM HERE, ?????
            fragment_seg[i] = vals[torch.argmax(cnts)].item()

        # Only pass shower fragments through particle clustering GNN
        em_mask = np.where(fragment_seg == 0)[0]
        edge_index, e = get_edge_features(x[em_mask], xbatch[em_mask], self.particle_edge_net)
        index = torch.tensor(edge_index.T, dtype=torch.long, device=device)
        gnn_output = self.particle_gnn(x[em_mask], index, e, xbatch[em_mask])

        # Divide the particle GNN output out into different arrays (one per batch)
        batch_ids = xbatch[em_mask].detach().cpu().numpy()
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[:,0]] == b)[0] for b in range(len(counts))]

        node_pred = [gnn_output['node_pred'][0][b] for b in bcids]
        edge_pred = [gnn_output['edge_pred'][0][b] for b in beids]
        edge_index = [cids[edge_index[b]] for b in beids]
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
                        #assert len(vals) == 1 # PROBEM HERE
                        batch_group_ids.append(vals[torch.argmax(cnts)].item())
                    group_ids.append(np.array(batch_group_ids, dtype=np.int32))
                else:
                    group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(frags[b])))

        result.update({'frag_group_pred': [group_ids]})

        # Merge fragments, concatenate features
        non_em_mask = np.where((fragment_seg != 0) & (fragment_seg != 4))[0]
        particles = fragments[non_em_mask].tolist()
        xp, xpbatch = x[non_em_mask], xbatch[non_em_mask]
        for b in range(len(counts)):
            batch_mask = np.where(batch_ids == b)[0]
            for g in np.unique(group_ids[b]):
                group_mask = np.where(group_ids[b] == g)[0]
                particles.append(np.concatenate(frags[b][group_mask]))
                group_feats = torch.mean(x[em_mask][batch_mask][group_mask], dim=0).reshape(1,-1)
                xp = torch.cat((xp, group_feats), dim=0)
                group_batch = xbatch[em_mask][batch_mask][group_mask].unique()
                assert len(group_batch) == 1
                xpbatch = torch.cat((xpbatch, group_batch[0].reshape(1)))
        order = torch.argsort(xpbatch)
        particles = np.array(particles)[order.detach().cpu().numpy()]
        xp = xp[order]
        xpbatch = xpbatch[order]

        # Pass particles through interaction clustering
        edge_index, e = get_edge_features(xp, xpbatch, self.inter_edge_net)
        index = torch.tensor(edge_index.T, dtype=torch.long, device=device)
        gnn_output = self.particle_gnn(xp, index, e, xpbatch)

        # Divide the interaction GNN output out into different arrays (one per batch)
        batch_ids = xpbatch.detach().cpu().numpy()
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[:,0]] == b)[0] for b in range(len(counts))]

        edge_pred = [gnn_output['edge_pred'][0][b] for b in beids]
        edge_index = [cids[edge_index[b]] for b in beids]
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
