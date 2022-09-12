import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.layers.common.blocks import ResNetBlock, SPP, ASPP
from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.layers.common.extract_feature_map import MinkGhostMask

from collections import Counter

from mlreco.models.layers.cluster_cnn.losses.misc import BinaryCELogDiceLoss


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


class PPN(torch.nn.Module):
    '''
    Point Proposal Network (PPN) implementation using MinkowskiEngine

    It requires a UResNet network as a backbone.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth: int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters: int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps: int, default 2
        Convolution block repetition factor
    input_kernel: int, default 3
        Receptive field size for very first convolution after input layer.
    num_classes: int, default 5
    score_threshold: float, default 0.5
    classify_endpoints: bool, default False
        Enable classification of points into start vs end points.
    ppn_resolution: float, default 1.0
    ghost: bool, default False
    downsample_ghost: bool, default True
    use_true_ghost_mask: bool, default False
    mask_loss_name: str, default 'BCE'
        Can be 'BCE' or 'LogDice'
    particles_label_seg_col: int, default -2
        Which column corresponds to particles' semantic label
    track_label: int, default 1

    Output
    ------
    points: torch.Tensor
        Contains  X, Y, Z predictions, semantic class prediction logits, and prob score
    mask_ppn: list of torch.Tensor
        Binary mask at various spatial scales of PPN predictions (voxel-wise score > some threshold)
    ppn_coords: list of torch.Tensor
        List of XYZ coordinates at various spatial scales.
    ppn_layers: list of torch.Tensor
        List of score features at various spatial scales.
    ppn_output_coordinates: torch.Tensor
        XYZ coordinates tensor at the very last layer of PPN (initial spatial scale)
    classify_endpoints: torch.Tensor
        Logits for end/start point classification.

    See Also
    --------
    PPNLonelyLoss, mlreco.models.uresnet_ppn_chain
    '''
    def __init__(self, cfg, name='ppn'):
        super(PPN, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_cfg = cfg.get(name, {})
        # UResNet Configurations
        self.reps = self.model_cfg.get('reps', 2)
        self.depth = self.model_cfg.get('depth', 5)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.num_filters = self.model_cfg.get('filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.ppn_score_threshold = self.model_cfg.get('score_threshold', 0.5)
        self.input_kernel = self.model_cfg.get('input_kernel', 3)
        self._classify_endpoints = self.model_cfg.get('classify_endpoints', False)

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

        if self._classify_endpoints:
            self.ppn_endpoint = ME.MinkowskiConvolution(self.nPlanes[0],
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
            #print("Ghost Masking is enabled for MinkPPN.")
            self.ghost_mask = MinkGhostMask(self.D)
            self.use_true_ghost_mask = self.model_cfg.get(
                'use_true_ghost_mask', False)
            self.downsample_ghost = self.model_cfg.get('downsample_ghost', True)

        # print('Total Number of Trainable Parameters (mink_ppnplus)= {}'.format(
        #             sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, final, decoderTensors, ghost=None, ghost_labels=None):
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
        if self._classify_endpoints:
            ppn_endpoint = self.ppn_endpoint(x)

        # X, Y, Z, logits, and prob score
        points = torch.cat([pixel_pred.F, ppn_type.F, ppn_final_score.F], dim=1)

        res = {
            'points': [points],
            'mask_ppn': [mask_ppn],
            'ppn_layers': [ppn_layers],
            'ppn_coords': [ppn_coords],
            'ppn_output_coordinates': [ppn_output_coordinates],
        }
        if self._classify_endpoints:
            res['classify_endpoints'] = [ppn_endpoint.F]

        return res


class PPNLonelyLoss(torch.nn.modules.loss._Loss):
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
        super(PPNLonelyLoss, self).__init__()
        self.loss_config = cfg.get(name, {})
        # pprint(self.loss_config)
        self.mask_loss_name = self.loss_config.get('mask_loss_name', 'BCE')
        if self.mask_loss_name == "BCE":
            self.lossfn = torch.nn.functional.binary_cross_entropy_with_logits
        elif self.mask_loss_name == "LogDice":
            self.lossfn = BinaryCELogDiceLoss()
        else:
            NotImplementedError
        self.resolution = self.loss_config.get('ppn_resolution', 1.0)
        self.regloss = torch.nn.MSELoss()
        self.segloss = torch.nn.functional.cross_entropy
        self.particles_label_seg_col = self.loss_config.get(
            'particles_label_seg_col', -2)

        # Endpoint classification (optional)
        self._classify_endpoints = self.loss_config.get('classify_endpoints', False)
        self._track_label = self.loss_config.get('track_label', 1)

        # Restrict the label points to specific classes (pass a list if needed)
        self._point_classes = self.loss_config.get('point_classes', [])

    @staticmethod
    def pairwise_distances(v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


    def forward(self, result, segment_label, particles_label):
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

        res = {
            'reg_loss': 0.,
            'mask_loss': 0.,
            'type_loss': 0.,
            'classify_endpoints_loss': 0.,
            'classify_endpoints_accuracy': 0.
        }
        # Semantic Segmentation Loss
        for igpu in range(len(segment_label)):
            particles = particles_label[igpu]
            if len(self._point_classes) > 0:
                classes    = particles[:, self.particles_label_seg_col]
                class_mask = torch.zeros(len(particles), dtype=torch.bool, device=particles.device)
                for c in self._point_classes:
                    class_mask |= classes == c
                particles = particles[class_mask]
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
                    if len(scores_event.shape) == 0:
                        continue

                    d_true = self.pairwise_distances(
                        points_label,
                        points_event[:, 1:4].float().to(device))

                    d_positives = (d_true < self.resolution * \
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
                        anchors = coords_layer[batch_particle_index][:, 1:4].float().to(device) + 0.5
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

                        # --- Endpoint classification loss
                        if self._classify_endpoints:
                            tracks = positive_labels == self._track_label
                            loss_point_class, acc_point_class, point_class_count = 0., 0., 0.
                            loss_classify_endpoints, acc_classify_endpoints = 0., 1.
                            if tracks.sum().item() > 0:
                                # Start and end points separately in case of overlap
                                for point_class in range(2):
                                    point_class_mask = particles[particles[:, 0].int() == b][:, -1] == point_class
                                    #true = event_particles[event_particles[:, -4] == b][torch.argmin(distances_positives, dim=0), -1]
                                    point_class_positives = (d_true[point_class_mask, :] < self.resolution).any(dim=0)
                                    point_class_index = d[point_class_mask, :][:, point_class_positives]

                                    if point_class_index.nelement():
                                        point_class_index = torch.argmin(point_class_index, dim=0)

                                        true = particles[particles[:, 0].int() == b][point_class_mask][point_class_index, -1]
                                        #pred = result['classify_endpoints'][i][batch_index][event_mask][positives]
                                        pred = result['classify_endpoints'][igpu][batch_index_layer][point_class_positives]
                                        tracks = event_types_label[point_class_index] == self._track_label
                                        if tracks.sum().item():
                                            loss_point_class += torch.mean(self.segloss(pred[tracks].double(), true[tracks].long()))
                                            acc_point_class += (torch.argmax(pred[tracks], dim=-1) == true[tracks]).sum().item() / float(true[tracks].nelement())
                                            point_class_count += 1

                                if point_class_count:
                                    loss_classify_endpoints = loss_point_class / point_class_count
                                    acc_classify_endpoints = acc_point_class / point_class_count
                                    #total_loss += loss_classify_endpoints.float()
                            res['classify_endpoints_loss'] += float(loss_classify_endpoints) / num_batches
                            res['classify_endpoints_accuracy'] += float(acc_classify_endpoints) / num_batches
                        # --- end of Endpoint classification

                        # Distance Loss
                        d2, _ = torch.min(distance_positives, dim=0)
                        reg_loss = d2.mean()
                        res['reg_loss'] += float(reg_loss) / num_batches
                        res['type_loss'] += float(type_loss) / num_batches
                        res['mask_loss'] += float(mask_loss_final) / num_batches
                        total_loss += (reg_loss + type_loss + mask_loss_final) / num_batches
                        if self._classify_endpoints:
                            total_loss += loss_classify_endpoints / num_batches

                loss_layer /= num_batches
                loss_gpu += loss_layer
            loss_gpu /= len(ppn_layers)
            total_loss += loss_gpu

        total_acc /= num_batches
        res['loss'] = total_loss
        res['accuracy'] = float(total_acc)
        return res
