import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.blocks import ResNetBlock, CascadeDilationBlock, SPP, ASPP
from mlreco.mink.layers.factories import activations_dict, activations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.models.ppn import define_ppn12

from collections import Counter

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


class ExpandAs(nn.Module):
    def __init__(self):
        super(ExpandAs, self).__init__()

    def forward(self, x, shape):
        device = x.F.device
        features = x.F.expand(*shape)
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
        self.depth = self.model_cfg.get('depth', 5)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.num_filters = self.model_cfg.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.ppn_score_threshold = self.model_cfg.get('ppn_score_threshold', 0.5)
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

        self.ppn_pixel_pred = ME.MinkowskiLinear(self.nPlanes[0], self.D)
        self.ppn_type = ME.MinkowskiLinear(self.nPlanes[0], self.num_classes)
        self.ppn_final_score = ME.MinkowskiLinear(self.nPlanes[0], 1)


    def forward(self, final, encoderTensors):
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
        x = final
        for i, layer in enumerate(self.decoding_conv):
            eTensor = encoderTensors[-i-2]
            x = layer(x)
            x = ME.cat(eTensor, x)
            x = self.decoding_block[i](x)
            scores = self.ppn_pred[i](x)
            tmp.append(scores.F)
            ppn_coords.append(scores.C)
            scores = self.sigmoid(scores)
            mask_ppn.append((scores.F > self.ppn_score_threshold))
            s_expanded = self.expand_as(scores, x.F.shape)
            x = x * s_expanded
        device = x.F.device
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
            'points': points,
            'mask_ppn': mask_ppn,
            'ppn_layers': ppn_layers,
            'ppn_coords': ppn_coords
        }

        return res


class PPNLonelyLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='ppn_loss'):
        super(PPNLonelyLoss, self).__init__()
        self.loss_config = cfg[name]
        self.lossfn = torch.nn.functional.binary_cross_entropy_with_logits
        self.resolution = self.loss_config.get('ppn_resolution', 5.0)
        self.regloss = torch.nn.MSELoss()
        self.segloss = torch.nn.functional.cross_entropy

    def pairwise_distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


    def forward(self, outputs, segment_label, particles_label):
        '''
        '''
        # TODO Add weighting
        assert len(particles_label) == len(segment_label)
        batch_ids = [d[:, 0] for d in segment_label]
        num_batches = len(batch_ids[0].unique())
        highE = [t[t[:, -1].long() != 4] for t in segment_label]
        total_loss = 0
        total_acc = 0
        count = 0
        device = segment_label[0].device

        loss, accuracy = 0, []
        res = {}
        # Semantic Segmentation Loss
        for igpu in range(len(segment_label)):
            particles = particles_label[igpu]
            ppn_layers = outputs['ppn_layers'][igpu]
            ppn_coords = outputs['ppn_coords'][igpu]
            points = outputs['points'][igpu]
            loss_gpu, acc_gpu = 0.0, 0.0
            for layer in range(len(ppn_layers)):
                ppn_score_layer = ppn_layers[layer]
                coords_layer = ppn_coords[layer]
                loss_layer = 0.0
                for b in batch_ids[igpu].int().unique():
                    batch_index_layer = coords_layer[:, 0].int() == b
                    batch_particle_index = batch_ids[igpu].int() == b
                    points_label = particles[particles[:, 0] == b][:, 1:4]
                    scores_event = ppn_score_layer[batch_index_layer].squeeze()
                    points_event = coords_layer[batch_index_layer]
                    d = self.pairwise_distances(points_label, points_event[:, 1:4].float().cuda())
                    d_positives = (d < self.resolution * 2**(len(ppn_layers) - layer)).any(dim=0)
                    num_positives = d_positives.sum()
                    num_negatives = d_positives.nelement() - num_positives
                    w = num_positives.float() / (num_positives + num_negatives).float()
                    weight_ppn = torch.zeros(d_positives.shape[0]).to(device)
                    weight_ppn[d_positives] = 1 - w
                    weight_ppn[~d_positives] = w
                    loss_batch = self.lossfn(scores_event, d_positives.float(), weight=weight_ppn, reduction='mean')
                    loss_layer += loss_batch
                    if layer == len(ppn_layers)-1:

                        # Get Final Layers
                        pixel_pred = coords_layer[batch_particle_index][:, 1:4].float().cuda()
                        pixel_score = points[batch_particle_index][:, -1]
                        pixel_logits = points[batch_particle_index][:, 3:8]

                        d = self.pairwise_distances(points_label, pixel_pred)
                        positives = (d < self.resolution).any(dim=0)
                        if (torch.sum(positives) < 1):
                            continue
                        acc = (positives == (pixel_score > 0)).sum().float() / float(pixel_score.shape[0])
                        total_acc += acc

                        # Mask Loss
                        mask_loss_final = self.lossfn(pixel_score, positives.float(), weight=weight_ppn, reduction='mean')

                        # Type Segmentation Loss
                        pixel_pred += points[batch_particle_index][:, :3]
                        d = self.pairwise_distances(points_label, pixel_pred + 0.5)
                        positives = (d < self.resolution).any(dim=0)
                        distance_positives = d[:, positives]
                        event_types_label = particles[particles[:, 0] == b][:, -2]
                        counter = Counter({0:0, 1:0, 2:0, 3:0})
                        counter.update(list(event_types_label.int().cpu().numpy()))
                        w = torch.Tensor([counter[0], counter[1], counter[2], counter[3], 0]).float()
                        w = float(sum(counter.values())) / (w + 1.0)
                        positive_labels = event_types_label[torch.argmin(distance_positives, dim=0)]
                        type_loss = self.segloss(pixel_logits[positives], positive_labels.long(), weight=w.to(device))

                        # Distance Loss
                        d2, _ = torch.min(distance_positives, dim=0)
                        reg_loss = d2.mean()
                        total_loss += (reg_loss + type_loss + mask_loss_final) / num_batches
                loss_layer /= num_batches
                loss_gpu += loss_layer
            loss_gpu /= len(ppn_layers)
            total_loss += loss_gpu

        total_acc /= num_batches
        res['loss'] = total_loss
        res['accuracy'] = float(total_acc)
        return res
