from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.ghost_chain import GhostChain, GhostChainLoss
from mlreco.models.layers.dbscan import distances
from mlreco.utils.deghosting import adapt_labels
from mlreco.utils.gnn.evaluation import node_assignment_score, primary_assignment
import numpy as np
from .gnn import node_encoder_construct
from mlreco.utils.gnn.cluster import get_cluster_label, get_cluster_batch
from mlreco.models.layers.extract_feature_map import Multiply


class GhostNuClassification(GhostChain):
    """
    Run UResNet and use its encoding/decoding feature maps for PPN layers
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (float,), (3, 1)],
    # ]
    MODULES = GhostChain.MODULES + ['spatial_embeddings', 'ghost_nu']

    def __init__(self, model_config):
        import sparseconvnet as scn

        super(GhostNuClassification, self).__init__(model_config)
        self._input_features = model_config['ghost_nu'].get('input_features', 1)
        m = model_config['ghost_nu'].get('features', 64)
        self._dimension = model_config['ghost_nu'].get('data_dim', 3)

        #self.classifier = torch.nn.Linear(self._input_features, 2)
        self.classifier = torch.nn.Linear(m, 2)

        #self.interaction_encoder = node_encoder_construct(model_config)
        filters = model_config['uresnet_lonely'].get('filters', 64)
        self.interaction_cnn = scn.Sequential()\
            .add(scn.BatchNormLeakyReLU(filters, leakiness=0.1))\
            .add(scn.SubmanifoldConvolution(self._dimension, filters, m, 3, False))\
            .add(scn.BatchNormLeakyReLU(m, leakiness=0.1))\
            .add(scn.SubmanifoldConvolution(self._dimension, m, 2*m, 3, False))\
            .add(scn.BatchNormLeakyReLU(2*m, leakiness=0.1))\
            .add(scn.SubmanifoldConvolution(self._dimension, 2*m, m, 3, False))
        self.multiply = Multiply()

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        """
        result = super(GhostNuClassification, self).forward(input)
        # Update input based on deghosting results
        deghost = result['ghost'][0].argmax(dim=1) == 0
        input = [input[0][deghost]]

        _, counts = torch.unique(input[0][:,3], return_counts=True)
        # Make interaction group predictions based on the GNN output, use truth during training
        group_ids = []
        for b in range(len(counts)):
            if not len(result['particles'][0][b]):
                group_ids.append(np.array([], dtype = np.int64))
            else:
                group_ids.append(node_assignment_score(result['inter_edge_index'][0][b], result['inter_edge_pred'][0][b].detach().cpu().numpy(), len(result['particles'][0][b])))

        result.update({'inter_group_pred': [group_ids]})
        #print(group_ids)
        #print(type(deghost))
        feature_map = self.multiply(result['ppn_feature_dec'][0][-1], deghost.float().view((-1, 1)))
        feature_map = self.interaction_cnn(feature_map).features[deghost]
        #print("feature_map", feature_map.size(), input[0].size())
        interactions = []
        x = []
        #new_batch_id = torch.zeros((input[0].size(0)))
        #interactions_batch_ids2 = []
        _, counts = torch.unique(input[0][:,3], return_counts=True)
        for b in range(len(counts)):
            voxel_inds = counts[:b].sum().item()+np.arange(counts[b].item())
            #print(b, len(np.unique(group_ids[b])))
            for g in np.unique(group_ids[b]):
                interaction_idx = np.concatenate(result['particles'][0][b][group_ids[b]==g])
                # inter = feature_map[group_ids == g]
                interactions.append(voxel_inds[interaction_idx])
                #new_batch_id[voxel_inds[interaction_idx]] = interaction_count
                #interactions_batch_ids2.append(b)
        #x = self.interaction_encoder(feature_map.features, interactions, )
        # #x = []
        for c in interactions:
        #     out_c = self.interaction_cnn(feature_map.features[c])
        #     print(out_c.features.size())
            #print(feature_map[c].mean(dim=0).size())
            x.append(feature_map[c].mean(dim=0))
        x = torch.stack(x, dim=0)
        #print(x.size())
        x = self.classifier(x)

        interactions = np.array(interactions)
        interactions_batch_ids = get_cluster_batch(input[0], interactions)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        bcids = [np.where(interactions_batch_ids == b)[0] for b in range(len(counts))]
        interactions = [np.array([vids[c] for c in interactions[b]]) for b in bcids]
        nu_pred = [x[b] for b in bcids]
        #nu_pred = [x[interaction] for interaction in interactions]

        result['nu_pred'] = [nu_pred]
        result['interactions'] = [interactions]
        return result


class GhostNuClassificationLoss(GhostChainLoss):
    """
    Loss for UResNet + PPN chain
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn", (int,), (3, 1)],
    #     ["parse_particle_points", (int,), (3, 1)]
    # ]

    def __init__(self, cfg):
        super(GhostNuClassificationLoss, self).__init__(cfg)
        #self.ghost_chain_loss = GhostChainLoss(cfg)
        #self.classification_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self._nu_weight = cfg['full_chain_loss'].get('nu_classification_weight', 1.)

    def forward(self, result, label_seg, label_clustering, label_ppn):
        #res = self.ghost_chain_loss(result, label_seg, label_clustering, label_ppn)
        res = super(GhostNuClassificationLoss, self).forward(result, label_seg, label_clustering, label_ppn)
        # Adapt to ghost points
        label_clustering = adapt_labels(result, label_seg, label_clustering)
        # Make nu labels for each interaction
        nu_loss, total_acc = 0., 0.
        nu_acc, cosmic_acc = 0., 0.
        device = result['nu_pred'][0][0].device
        n_interactions, n_nu, n_cosmic = 0, 0, 0
        for i in range(len(label_seg)):
            #print("i = ", i, len(result['nu_pred'][i]), len(result['inter_group_pred'][i]))
            #for b in range(len(label_seg[i][:, 3].unique())):
            #    print(b, len(result['nu_pred'][i][b]), len(result['inter_group_pred'][i][b]), len(result['particles'][i][b]))
            for b in range(len(label_seg[i][:, 3].unique())):
                batch_mask = label_clustering[i][:, 3] == b
                #nu_label = get_cluster_label(label_clustering[i][batch_mask], result['inter_group_pred'][i][b], column=8)
                nu_label = get_cluster_label(label_clustering[i][batch_mask], result['interactions'][i][b], column=8)
                #nu_batch = get_cluster_batch(label_clustering[i][batch_mask], result['inter_group_pred'][i][int(b.item())])
                nu_label = torch.tensor(nu_label, requires_grad=False, device=device).view(-1)
                nu_label = (nu_label > -1).long()
                #nu_loss += self.classification_loss(result['nu_pred'][i][b], nu_label)
                nu_count = (nu_label == 1).sum().float()
                #cosmic_count = len(nu_label) == nu_count
                w = torch.tensor([nu_count / float(len(nu_label)) + 0.01, 1 - nu_count / float(len(nu_label)) - 0.01], device=device)
                print("Weight: ", w)
                nu_loss += torch.nn.functional.cross_entropy(result['nu_pred'][i][b], nu_label, weight=w.float())

                total_acc += (result['nu_pred'][i][b].argmax(dim=1) == nu_label).sum().float()
                nu_acc += (result['nu_pred'][i][b].argmax(dim=1) == nu_label)[nu_label == 1].sum().float()
                cosmic_acc += (result['nu_pred'][i][b].argmax(dim=1) == nu_label)[nu_label == 0].sum().float()
                #print(b, (result['nu_pred'][i][b].argmax(dim=1) == nu_label).sum().float()/len(nu_label))
                #print(result['nu_pred'][i][b])
                print(result['nu_pred'][i][b].argmax(dim=1), nu_label)
                n_interactions += len(nu_label)
                n_nu += (nu_label == 1).sum().float()
                n_cosmic += (nu_label == 0).sum().float()
        # Don't forget to sum all losses
        res['nu_total_loss'] = nu_loss/n_interactions
        res['nu_total_acc'] = total_acc/n_interactions
        res['nu_acc'] = nu_acc/n_nu
        res['cosmic_acc'] = cosmic_acc/n_cosmic
        res['loss'] += self._nu_weight * res['nu_total_loss']
        res['accuracy'] = (5 * res['accuracy'] + res['nu_total_acc'])/6.
        print("Nu Classification Accuracy: ", res['nu_total_acc'].item(), "(", res['nu_acc'].item(), "for nu class and ", res['cosmic_acc'].item(), " for cosmic class)")
        return res
