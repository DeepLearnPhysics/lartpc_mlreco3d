import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.utils import local_cdist
from mlreco.models.layers.common.blocks import ResNetBlock
from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration

from mlreco.models.layers.cluster_cnn.losses.misc import BinaryCELogDiceLoss
from mlreco.models.layers.common.ppnplus import *
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label

class VertexPPN(nn.Module):

    def __init__(self, cfg, name='vertex_ppn'):
        super(VertexPPN, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_cfg = cfg.get(name, {})
        # UResNet Configurations
        self.reps = self.model_cfg.get('reps', 2)
        self.depth = self.model_cfg.get('depth', 5)
        self.num_filters = self.model_cfg.get('filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.vertex_score_threshold = self.model_cfg.get('score_threshold', 0.5)
        self.input_kernel = self.model_cfg.get('input_kernel', 3)

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        self.vertex_pred = nn.ModuleList()
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                     self.nPlanes[i],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            self.vertex_pred.append(ME.MinkowskiLinear(self.nPlanes[i], 1))
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)

        self.sigmoid = ME.MinkowskiSigmoid()
        self.expand_as = ExpandAs()

        self.final_block = ResNetBlock(self.nPlanes[0],
                                       self.nPlanes[0],
                                       dimension=self.D,
                                       activation=self.activation_name,
                                       activation_args=self.activation_args)

        self.vertex_regression = ME.MinkowskiConvolution(self.nPlanes[0],
                                                      self.D,
                                                      kernel_size=3,
                                                      stride=1,
                                                      dimension=self.D)
        self.vertexness_score = ME.MinkowskiConvolution(self.nPlanes[0],
                                                       2,
                                                       kernel_size=3,
                                                       stride=1,
                                                       dimension=self.D)


    def forward(self, final, decoderTensors, 
                input_sparse_tensor=None, 
                primary_labels=None,
                segment_labels=None):
        vertex_layers, vertex_coords = [], []
        tmp = []
        segment_label_scales = []
        primary_label_scales = []
        device = final.device

        # We need to make labels on-the-fly to include true points in the
        # propagated masks during training

        decoder_feature_maps = decoderTensors

        x = final

        for i, layer in enumerate(self.decoding_conv):

            decTensor = decoder_feature_maps[i]
            x = layer(x)
            x = ME.cat(decTensor, x)
            x = self.decoding_block[i](x)
            scores = self.vertex_pred[i](x)
            tmp.append(scores.F)

            vertex_coords.append(scores.C)
            scores = self.sigmoid(scores)

            s_expanded = self.expand_as(scores, x.F.shape)
            x = x * s_expanded.detach()

        device = x.F.device
        vertex_output_coordinates = x.C
        for p in tmp:
            a = p.to(dtype=torch.float32, device=device)
            vertex_layers.append(a)

        x = self.final_block(x)
        pixel_pred = self.vertex_regression(x)
        vertex_final_score = self.vertexness_score(x)

        # X, Y, Z, logits
        points = torch.cat([pixel_pred.F, vertex_final_score.F], dim=1)

        if primary_labels is not None:

            primary_labels_curr = primary_labels
            segment_labels_curr = segment_labels

            for i, x in enumerate(reversed(decoderTensors[:3])):
                print(i, x.tensor_stride, decoderTensors[::-1][i].tensor_stride)

                primary_label_layer = torch.zeros(x.C.shape[0], dtype=torch.bool, device=x.F.device)
                segment_label_layer = torch.zeros(x.C.shape[0], dtype=torch.long, device=x.F.device)
                
                kernel_map = input_sparse_tensor.coordinate_manager.kernel_map(decoderTensors[::-1][i].tensor_stride, x.tensor_stride)

                for k, curr in kernel_map.items():
                    primary_label_layer[curr[1].long()] = primary_labels_curr[curr[0].long()].bool()
                    segment_label_layer[curr[1].long()] = segment_labels_curr[curr[0].long()].long()

                primary_label_scales.append(primary_label_layer)
                segment_label_scales.append(segment_label_layer)

                primary_labels_curr = primary_label_layer
                segment_labels_curr = segment_label_layer

        res = {
            'vertex_points': [points],
            'vertex_layers': [vertex_layers],
            'vertex_coords': [vertex_coords],
            'vertex_output_coordinates': [vertex_output_coordinates],
        }

        if primary_labels is not None:
            res['primary_label_scales'] = [primary_label_scales[::-1]]
            res['segment_label_scales'] = [segment_label_scales[::-1]]

        return res


class VertexPPNLoss(torch.nn.modules.loss._Loss):
    """
    Loss function for PPN.

    Output
    ------
    reg_loss: float
        Distance loss
    mask_loss: float
        Binary voxel-wise prediction (is there an object of interest or not)
    type_loss: float
        Semantic prediction loss.
    classify_endpoints_loss: float
    classify_endpoints_acc: float

    See Also
    --------
    PPN, mlreco.models.uresnet_ppn_chain
    """

    def __init__(self, cfg, name='ppn'):
        super(VertexPPNLoss, self).__init__()
        self.loss_config = cfg.get(name, {})
        # pprint(self.loss_config)
        self.mask_loss_name = self.loss_config.get('mask_loss_name', 'BCE')
        if self.mask_loss_name == "BCE":
            self.lossfn = torch.nn.functional.binary_cross_entropy_with_logits
        elif self.mask_loss_name == "LogDice":
            self.lossfn = BinaryCELogDiceLoss()
        else:
            raise NotImplementedError
        self.resolution = self.loss_config.get('ppn_resolution', 1.0)
        self.regloss = torch.nn.MSELoss()
        self.use_numpy = self.loss_config.get('use_numpy', False)


    def get_vertices(self, kinematics_label):
        
        data =  kinematics_label[0].cpu().numpy()
        batch_ids = data[:, 0]
        clusts = form_clusters(data, column=7)
        inter_label = get_cluster_label(data, clusts, column=7)
        batch_index = get_cluster_batch(data, clusts)
        primary_labels = get_cluster_label(data, clusts, column=8)
        vtx_x = get_cluster_label(data, clusts, column=10)
        vtx_y = get_cluster_label(data, clusts, column=11)
        vtx_z = get_cluster_label(data, clusts, column=12)

        vertices_label = np.hstack([batch_index.reshape(-1, 1), 
                                    vtx_x.reshape(-1, 1), 
                                    vtx_y.reshape(-1, 1), 
                                    vtx_z.reshape(-1, 1), 
                                    inter_label.reshape(-1, 1),
                                    primary_labels.reshape(-1, 1)])
    
        return [vertices_label]

    
    def compute_heatmap(self, coords_batch, vertices_batch, primaries_batch):

        heatmap = torch.zeros(coords_batch.shape[0], dtype=torch.float32, 
                                                     device=coords_batch.device)

        primary_mask = primaries_batch == 1
        heatmap[primary_mask] = 1.0

        valid_coords = coords_batch[primary_mask]

        print(heatmap, valid_coords.shape, vertices_batch.shape)
        print(vertices_batch)
        assert False
        return 0


    def forward(self, result, kinematics_label):

        batch_ids = [result['vertex_coords'][0][-1][:, 0]]
        num_batches = len(batch_ids[0].unique())
        total_loss = 0
        total_acc = 0
        device = kinematics_label[0].device

        res = {
            'vertex_reg_loss': 0.,
            'vertex_mask_loss': 0.
        }

        particles_label = self.get_vertices(kinematics_label)

        print("Particles Label = ", particles_label[0], type(particles_label))

        # Semantic Segmentation Loss
        for igpu in range(len(kinematics_label)):
            particles = particles_label[igpu]
            ppn_layers = result['vertex_layers'][igpu]
            ppn_coords = result['vertex_coords'][igpu]
            points = result['vertex_points'][igpu]
            primary_label_scales = result['primary_label_scales'][igpu]
            loss_gpu, acc_gpu = 0.0, 0.0
            num_layers = len(ppn_layers)
            for layer in range(len(ppn_layers)):
                # print("Layer = ", layer)
                ppn_score_layer = ppn_layers[layer]
                coords_layer = ppn_coords[layer]
                primaries_layer = primary_label_scales[layer]
                loss_layer = 0.0
                for b in batch_ids[igpu].int().unique():

                    batch_index_layer = coords_layer[:, 0].int() == b
                    batch_particle_index = batch_ids[igpu].int() == b
                    print(particles, particles[:, 0], b)
                    print("particles[:, 0] == b = ", particles[:, 0] == b.item())
                    points_label_batch = particles[particles[:, 0] == b]
                    vertices_batch = points_label_batch[:, 1:4]
                    print("points_label_batch = ", points_label_batch)
                    scores_event = ppn_score_layer[batch_index_layer].squeeze()
                    coords_batch = coords_layer[batch_index_layer]
                    primaries_batch = primaries_layer[batch_index_layer]

                    heatmap = self.compute_heatmap(coords_batch, 
                                                   vertices_batch, 
                                                   primaries_batch)

            #         d_true = pairwise_distances(
            #             points_label,
            #             points_event[:, 1:4].float().to(device))

            #         d_positives = (d_true < self.resolution * \
            #                        2**(len(ppn_layers) - layer)).any(dim=0)

            #         num_positives = d_positives.sum()
            #         num_negatives = d_positives.nelement() - num_positives

            #         w = num_positives.float() / \
            #             (num_positives + num_negatives).float()

            #         weight_ppn = torch.zeros(d_positives.shape[0]).to(device)
            #         weight_ppn[d_positives] = 1 - w
            #         weight_ppn[~d_positives] = w

            #         loss_batch = self.lossfn(scores_event,
            #                                  d_positives.float(),
            #                                  weight=weight_ppn,
            #                                  reduction='mean')

            #         loss_layer += loss_batch
            #         if layer == len(ppn_layers)-1:

            #             # Get Final Layers
            #             anchors = coords_layer[batch_particle_index][:, 1:4].float().to(device) + 0.5
            #             pixel_score = points[batch_particle_index][:, -1]
            #             pixel_logits = points[batch_particle_index][:, 3:8]
            #             pixel_pred = points[batch_particle_index][:, :3] + anchors

            #             d = pairwise_distances(points_label, pixel_pred)
            #             positives = (d < self.resolution).any(dim=0)
            #             if (torch.sum(positives) < 1):
            #                 continue
            #             acc = (positives == (pixel_score > 0)).sum().float() / float(pixel_score.shape[0])
            #             total_acc += acc

            #             # Mask Loss
            #             mask_loss_final = self.lossfn(pixel_score,
            #                                           positives.float(),
            #                                           weight=weight_ppn,
            #                                           reduction='mean')

            #             distance_positives = d[:, positives]

            #             # Distance Loss
            #             d2, _ = torch.min(distance_positives, dim=0)
            #             reg_loss = d2.mean()
            #             res['vertex_reg_loss'] += float(reg_loss) / num_batches
            #             res['vertex_mask_loss'] += float(mask_loss_final) / num_batches
            #             total_loss += (reg_loss + mask_loss_final) / num_batches

            #     loss_layer /= num_batches
            #     loss_gpu += loss_layer
            # loss_gpu /= len(ppn_layers)
            # total_loss += loss_gpu

        total_acc /= num_batches
        res['vertex_loss'] = total_loss
        res['vertex_acc'] = float(total_acc)
        return res
