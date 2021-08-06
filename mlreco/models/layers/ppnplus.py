import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.layers.blocks import ResNetBlock, SPP, ASPP
from mlreco.models.layers.factories import activations_construct
from mlreco.models.layers.network_base import MENetworkBase
from mlreco.models.layers.extract_feature_map import MinkGhostMask

from collections import Counter

from mlreco.models.cluster_cnn.losses.misc import BinaryCELogDiceLoss

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x, scores):
        features = x.F
        features = features * scores
        coords = x.C
        output = ME.SparseTensor(
            coordinates=coords, features=features)
        return output


class AttentionMask(torch.nn.Module):
    '''
    Returns a masked tensor of x according to mask, where the number of
    coordinates between x and mask differ
    '''
    def __init__(self, score_threshold=0.5):
        super(AttentionMask, self).__init__()
        self.prune = ME.MinkowskiPruning()
        self.score_threshold=score_threshold

    def forward(self, x, mask):

        assert x.tensor_stride == mask.tensor_stride

        device = x.F.device
        # Create a mask sparse tensor in x-coordinates
        x0 = ME.SparseTensor(
            coordinates=x.C,
            features=torch.zeros(x.F.shape[0], mask.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride)

        mask_in_xcoords = x0 + mask

        x_expanded = ME.SparseTensor(
            coordinates=mask_in_xcoords.C,
            features=torch.zeros(mask_in_xcoords.F.shape[0],
                                 x.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride)

        x_expanded = x_expanded + x

        target = mask_in_xcoords.F.int().bool().squeeze()
        x_pruned = self.prune(x_expanded, target)
        return x_pruned


class MergeConcat(torch.nn.Module):

    def __init__(self):
        super(MergeConcat, self).__init__()

    def forward(self, input, other):

        assert input.tensor_stride == other.tensor_stride
        device = input.F.device

        # Create a placeholder tensor with input.C coordinates
        x0 = ME.SparseTensor(
            coordinates=input.C,
            features=torch.zeros(input.F.shape[0], other.F.shape[1]).to(device),
            coordinate_manager=input.coordinate_manager,
            tensor_stride=input.tensor_stride)

        # Set placeholder values with other.F features by performing
        # sparse tensor addition.
        x1 = x0 + other

        # Same procedure, but with other
        x_expanded = ME.SparseTensor(
            coordinates=x1.C,
            features=torch.zeros(x1.F.shape[0],
                                 input.F.shape[1]).to(device),
            coordinate_manager=input.coordinate_manager,
            tensor_stride=input.tensor_stride)

        x2 = x_expanded + input

        # Now input and other share the same coordinates and shape
        concated = ME.cat(x1, x2)
        return concated


class ExpandAs(nn.Module):
    def __init__(self):
        super(ExpandAs, self).__init__()

    def forward(self, x, shape, labels=None):
        '''
            x: feature tensor of input sparse tensor (N x F)
            labels: N x 0 tensor of labels

        '''
        device = x.F.device
        features = x.F
        features[labels] = 1.0
        features = features.expand(*shape)
        # if labels is not None:
        #     features_expand = features.expand(*shape).clone()
        #     features_expand[labels] = 1.0
        # else:
        #     features_expand = features.expand(*shape)
        output = ME.SparseTensor(
            features=features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        return output


class PPN(MENetworkBase):
    '''
    MinkowskiEngine PPN

    Configurations
    --------------
    depth : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    num_filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    input_kernel : int, optional
        Receptive field size for very first convolution after input layer.
    '''
    def __init__(self, cfg, name='ppn'):
        super(PPN, self).__init__(cfg)
        self.model_cfg = cfg[name]
        # UResNet Configurations
        self.reps = self.model_cfg.get('reps', 2)
        self.depth = self.model_cfg.get('num_strides', 5)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.num_filters = self.model_cfg.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.ppn_score_threshold = self.model_cfg.get('score_threshold', 0.5)
        self.input_kernel = self.model_cfg.get('input_kernel', 3)

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        self.ppn_pred = nn.ModuleList()
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
            self.ppn_pred.append(ME.MinkowskiLinear(self.nPlanes[i], 1))
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)

        self.sigmoid = ME.MinkowskiSigmoid()
        self.expand_as = ExpandAs()

        self.final_block = ResNetBlock(self.nPlanes[0],
                                       self.nPlanes[0],
                                       dimension=self.D,
                                       activation=self.activation_name,
                                       activation_args=self.activation_args)

        self.ppn_pixel_pred = ME.MinkowskiConvolution(self.nPlanes[0],
                                                      self.D,
                                                      kernel_size=3,
                                                      stride=1,
                                                      dimension=self.D)
        self.ppn_type = ME.MinkowskiConvolution(self.nPlanes[0],
                                                self.num_classes,
                                                kernel_size=3,
                                                stride=1,
                                                dimension=self.D)
        self.ppn_final_score = ME.MinkowskiConvolution(self.nPlanes[0],
                                                       2,
                                                       kernel_size=3,
                                                       stride=1,
                                                       dimension=self.D)

        self.resolution = self.model_cfg.get('ppn_resolution', 1.0)

        # Ghost point removal options
        self.ghost = self.model_cfg.get('ghost', False)

        self.masker = AttentionMask()
        self.merge_concat = MergeConcat()

        if self.ghost:
            print("Ghost Masking is enabled for MinkPPN.")
            self.ghost_mask = MinkGhostMask(self.D)
            self.use_true_ghost_mask = self.model_cfg.get(
                'use_true_ghost_mask', False)
            self.downsample_ghost = self.model_cfg.get('downsample_ghost', True)

        print('Total Number of Trainable Parameters (mink_ppnplus)= {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, final, decoderTensors, ghost=None, ghost_labels=None):
        '''
        Vanilla UResNet Decoder
        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''

        ppn_layers, ppn_coords = [], []
        tmp = []
        mask_ppn = []
        device = final.device

        # We need to make labels on-the-fly to include true points in the
        # propagated masks during training

        decoder_feature_maps = []

        if self.ghost:
            # Downsample stride 1 ghost mask to all intermediate decoder layers
            with torch.no_grad():
                if self.use_true_ghost_mask:
                    assert ghost_labels is not None
                    # TODO: Not sure what's going on here
                    ghost_mask_tensor = ghost_labels[:, -1] < self.num_classes
                    ghost_coords = ghost_labels[:, :4]
                else:
                    ghost_mask_tensor = 1.0 - torch.argmax(ghost.F,
                                                           dim=1,
                                                           keepdim=True)
                    ghost_coords = ghost.C
                    ghost_coords_man = final.coordinate_manager
                    ghost_tensor_stride = ghost.tensor_stride
                ghost_mask = ME.SparseTensor(
                    features=ghost_mask_tensor,
                    coordinates=ghost_coords,
                    coordinate_manager=ghost_coords_man,
                    tensor_stride=ghost_tensor_stride)

            for t in decoderTensors[::-1]:
                scaled_ghost_mask = self.ghost_mask(ghost_mask, t)
                nonghost_tensor = self.masker(t, scaled_ghost_mask)
                decoder_feature_maps.append(nonghost_tensor)

            decoder_feature_maps = decoder_feature_maps[::-1]

        else:
            decoder_feature_maps = decoderTensors

        x = final

        for i, layer in enumerate(self.decoding_conv):

            decTensor = decoder_feature_maps[i]
            x = layer(x)
            if self.ghost:
                x = self.merge_concat(decTensor, x)
            else:
                x = ME.cat(decTensor, x)
            x = self.decoding_block[i](x)
            scores = self.ppn_pred[i](x)
            tmp.append(scores.F)
            ppn_coords.append(scores.C)
            scores = self.sigmoid(scores)

            s_expanded = self.expand_as(scores, x.F.shape)

            mask_ppn.append((scores.F > self.ppn_score_threshold))
            x = x * s_expanded.detach()

        # Note that we skipped ghost masking for the final sparse tensor,
        # namely the tensor with the same resolution as the input to uresnet.
        # This is done at the full chain cnn stage, for consistency with SCN

        device = x.F.device
        ppn_output_coordinates = x.C
        for p in tmp:
            a = p.to(dtype=torch.float32, device=device)
            ppn_layers.append(a)

        x = self.final_block(x)
        pixel_pred = self.ppn_pixel_pred(x)
        ppn_type = self.ppn_type(x)
        ppn_final_score = self.ppn_final_score(x)

        # X, Y, Z, logits, and prob score
        points = torch.cat([pixel_pred.F, ppn_type.F, ppn_final_score.F], dim=1)

        res = {
            'points': [points],
            'mask_ppn': [mask_ppn],
            'ppn_layers': [ppn_layers],
            'ppn_coords': [ppn_coords],
            'ppn_output_coordinates': [ppn_output_coordinates],
        }

        return res


class PPNLonelyLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='ppn'):
        super(PPNLonelyLoss, self).__init__()
        self.loss_config = cfg[name]
        # pprint(self.loss_config)
        self.mask_loss_name = self.loss_config.get('mask_loss_name', 'BCE')
        if self.mask_loss_name == "BCE":
            self.lossfn = torch.nn.functional.binary_cross_entropy_with_logits
        elif self.mask_loss_name == "LogDice":
            self.lossfn = BinaryCELogDiceLoss()
        else:
            NotImplementedError
        self.resolution = self.loss_config.get('ppn_resolution', 5.0)
        self.regloss = torch.nn.MSELoss()
        self.segloss = torch.nn.functional.cross_entropy
        self.particles_label_seg_col = self.loss_config.get(
            'particles_label_seg_col', -2)

    @staticmethod
    def pairwise_distances(v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


    def forward(self, result, segment_label, particles_label):
        '''
        '''
        # TODO Add weighting
        assert len(particles_label) == len(segment_label)

        ppn_output_coordinates = result['ppn_output_coordinates']
        # print("PPN Output Coordinates = ", ppn_output_coordinates[0].shape)
        # assert False
        # print(result['ppn_coords'][0][-1])
        batch_ids = [result['ppn_coords'][0][-1][:, 0]]
        num_batches = len(batch_ids[0].unique())
        total_loss = 0
        total_acc = 0
        device = segment_label[0].device

        res = {}
        # Semantic Segmentation Loss
        for igpu in range(len(segment_label)):
            particles = particles_label[igpu]
            ppn_layers = result['ppn_layers'][igpu]
            ppn_coords = result['ppn_coords'][igpu]
            points = result['points'][igpu]
            loss_gpu, acc_gpu = 0.0, 0.0
            for layer in range(len(ppn_layers)):
                # print("Layer = ", layer)
                ppn_score_layer = ppn_layers[layer]
                coords_layer = ppn_coords[layer]
                loss_layer = 0.0
                for b in batch_ids[igpu].int().unique():

                    batch_index_layer = coords_layer[:, 0].int() == b
                    batch_particle_index = batch_ids[igpu].int() == b
                    points_label = particles[particles[:, 0].int() == b][:, 1:4]
                    scores_event = ppn_score_layer[batch_index_layer].squeeze()
                    points_event = coords_layer[batch_index_layer]

                    d = self.pairwise_distances(
                        points_label,
                        points_event[:, 1:4].float().cuda())

                    d_positives = (d < self.resolution * \
                                   2**(len(ppn_layers) - layer)).any(dim=0)

                    num_positives = d_positives.sum()
                    num_negatives = d_positives.nelement() - num_positives

                    w = num_positives.float() / \
                        (num_positives + num_negatives).float()

                    weight_ppn = torch.zeros(d_positives.shape[0]).to(device)
                    weight_ppn[d_positives] = 1 - w
                    weight_ppn[~d_positives] = w

                    loss_batch = self.lossfn(scores_event,
                                             d_positives.float(),
                                             weight=weight_ppn,
                                             reduction='mean')

                    loss_layer += loss_batch
                    if layer == len(ppn_layers)-1:

                        # Get Final Layers
                        anchors = coords_layer[batch_particle_index][:, 1:4].float().cuda() + 0.5
                        pixel_score = points[batch_particle_index][:, -1]
                        pixel_logits = points[batch_particle_index][:, 3:8]
                        pixel_pred = points[batch_particle_index][:, :3] + anchors

                        d = self.pairwise_distances(points_label, pixel_pred)
                        positives = (d < self.resolution).any(dim=0)
                        if (torch.sum(positives) < 1):
                            continue
                        acc = (positives == (pixel_score > 0)).sum().float() / float(pixel_score.shape[0])
                        total_acc += acc

                        # Mask Loss
                        mask_loss_final = self.lossfn(pixel_score,
                                                      positives.float(),
                                                      weight=weight_ppn,
                                                      reduction='mean')

                        # Type Segmentation Loss
                        # d = self.pairwise_distances(points_label, pixel_pred)
                        # positives = (d < self.resolution).any(dim=0)
                        distance_positives = d[:, positives]
                        event_types_label = particles[particles[:, 0] == b]\
                                                     [:, self.particles_label_seg_col]
                        counter = Counter({0:0, 1:0, 2:0, 3:0})
                        counter.update(list(event_types_label.int().cpu().numpy()))

                        w = torch.Tensor([counter[0],
                                          counter[1],
                                          counter[2],
                                          counter[3], 0]).float()

                        w = float(sum(counter.values())) / (w + 1.0)
                        positive_labels = event_types_label[torch.argmin(distance_positives, dim=0)]
                        type_loss = self.segloss(pixel_logits[positives],
                                                 positive_labels.long(),
                                                 weight=w.to(device))

                        # Distance Loss
                        d2, _ = torch.min(distance_positives, dim=0)
                        reg_loss = d2.mean()
                        res['reg_loss'] = float(reg_loss)
                        res['type_loss'] = float(type_loss)
                        res['mask_loss'] = float(mask_loss_final)
                        total_loss += (reg_loss + type_loss + mask_loss_final) / num_batches
                loss_layer /= num_batches
                loss_gpu += loss_layer
            loss_gpu /= len(ppn_layers)
            total_loss += loss_gpu

        total_acc /= num_batches
        res['ppn_loss'] = total_loss
        res['ppn_acc'] = float(total_acc)
        return res
