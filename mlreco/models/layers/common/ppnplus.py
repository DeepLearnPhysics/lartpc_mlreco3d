import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.utils import local_cdist
from mlreco.utils.globals import *
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

    def forward(self, x, shape, labels=None, 
                propagate_all=False,
                use_binary_mask=False):
        '''
            x: feature tensor of input sparse tensor (N x F)
            labels: N x 0 tensor of labels
            propagate_all: If True, PPN will not perform masking at each layer.

        '''
        device = x.F.device
        features = x.F
        if labels is not None:
            assert labels.shape[0] == x.F.shape[0]
            features[labels] = 1.0
        if propagate_all:
            features[None] = 1.0
        if use_binary_mask:
            features = (features > 0.5).float().expand(*shape)
        else:
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


def get_ppn_weights(d_positives, mode='const', eps=1e-3):

    device = d_positives.device

    num_positives = d_positives.sum()
    num_negatives = d_positives.nelement() - num_positives

    w = num_positives.float() / \
        (num_positives + num_negatives).float()

    weight_ppn = torch.ones(d_positives.shape[0]).to(device)

    if mode == 'const':
        weight_ppn[d_positives] = 1-w
        weight_ppn[~d_positives] = w
    elif mode == 'log':
        weight_ppn[d_positives] = -torch.log(w + eps)
        weight_ppn[~d_positives] = -torch.log(1-w + eps)
    elif mode == 'sqrt':
        weight_ppn[d_positives] = torch.sqrt(w + eps)
        weight_ppn[~d_positives] = torch.sqrt(1-w + eps)
    else:
        raise ValueError("Weight mode {} not supported!".format(mode))

    return weight_ppn


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
    ppn_points: torch.Tensor
        Contains  X, Y, Z predictions, semantic class prediction logits, and prob score
    ppn_masks: list of torch.Tensor
        Binary masks at various spatial scales of PPN predictions (voxel-wise score > some threshold)
    ppn_coords: list of torch.Tensor
        List of XYZ coordinates at various spatial scales.
    ppn_layers: list of torch.Tensor
        List of score features at various spatial scales.
    ppn_output_coords: torch.Tensor
        XYZ coordinates tensor at the very last layer of PPN (initial spatial scale)
    ppn_classify_endpoints: torch.Tensor
        Logits for end/start point classification.

    See Also
    --------
    PPNLonelyLoss, mlreco.models.uresnet_ppn_chain
    '''

    RETURNS = {
        'ppn_points': ['tensor', 'ppn_output_coords'],
        'ppn_masks': ['tensor_list', 'ppn_coords'],
        'ppn_layers': ['tensor_list', 'ppn_coords'],
        'ppn_coords': ['tensor_list', 'ppn_coords', False, True],
        'ppn_output_coords': ['tensor', 'ppn_output_coords', False, True],
        'ppn_classify_endpoints': ['tensor', 'ppn_output_coords']
    }

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
        self.propagate_all = self.model_cfg.get('propagate_all', False)
        self.use_binary_mask_ppn = self.model_cfg.get('use_binary_mask_ppn', False)

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
        ppn_masks = []
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
            mask = self.sigmoid(scores)
            s_expanded = self.expand_as(mask, x.F.shape, 
                                        propagate_all=self.propagate_all,
                                        use_binary_mask=self.use_binary_mask_ppn)
            ppn_masks.append((mask.F > self.ppn_score_threshold))
            x = x * s_expanded.detach()

        # Note that we skipped ghost masking for the final sparse tensor,
        # namely the tensor with the same resolution as the input to uresnet.
        # This is done at the full chain cnn stage, for consistency with SCN

        device = x.F.device
        ppn_output_coords = x.C
        # print(x.tensor_stride, x.shape, "ppn_score_threshold = ", self.ppn_score_threshold)
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
        ppn_points = torch.cat([pixel_pred.F, ppn_type.F, ppn_final_score.F], dim=1)

        res = {
            'ppn_points': [ppn_points],
            'ppn_masks':  [ppn_masks],
            'ppn_layers': [ppn_layers],
            'ppn_coords': [ppn_coords],
            'ppn_output_coords': [ppn_output_coords],
        }
        if self._classify_endpoints:
            res['ppn_classify_endpoints'] = [ppn_endpoint.F]

        return res


class PPNLonelyLoss(torch.nn.modules.loss._Loss):
    """
    Loss function for PPN.

    Output
    ------
    reg_loss : float
        Distance loss
    mask_loss : float
        Binary voxel-wise prediction loss (is there an object of interest or not)
    classify_endpoints_loss : float
        Endpoint classification loss
    type_loss : float
        Semantic prediction loss
    output_mask_accuracy: float
        Binary voxel-wise prediction accuracy in the last layer
    type_accuracy : float
        Semantic prediction accuracy
    classify_endpoints_accuracy : float
        Endpoint classification accuracy

    See Also
    --------
    PPN, mlreco.models.uresnet_ppn_chain
    """

    RETURNS = {
        'reg_loss': ['scalar'],
        'mask_loss': ['scalar'],
        'type_loss': ['scalar'],
        'classify_endpoints_loss': ['scalar'],
        'output_mask_accuracy': ['scalar'],
        'type_accuracy': ['scalar'],
        'classify_endpoints_accuracy': ['scalar'],
        'num_positives': ['scalar'],
        'num_voxels': ['scalar']
    }

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

        self.ppn_weighing_mode = self.loss_config.get('ppn_weighing_mode', 'const')

        # Endpoint classification (optional)
        self._classify_endpoints = self.loss_config.get('classify_endpoints', False)
        self._track_label = self.loss_config.get('track_label', 1)

        # Restrict the label points to specific classes (pass a list if needed)
        self._point_classes = self.loss_config.get('point_classes', [])

        self.mask_loss_weight = self.loss_config.get('mask_loss_weight', 1.0)
        self.reg_loss_weight = self.loss_config.get('reg_loss_weight', 1.0)
        self.point_type_loss_weight = self.loss_config.get('point_type_loss_weight', 1.0)
        self.classify_endpoints_loss_weight = self.loss_config.get('classify_endpoints_loss_weight', 1.0)

        #print("Mask Loss Weight = ", self.mask_loss_weight)


    def forward(self, result, segment_label, particles_label):
        # TODO Add weighting
        assert len(particles_label) == len(segment_label)

        ppn_output_coords = result['ppn_output_coords']
        batch_ids = [result['ppn_coords'][0][-1][:, 0]]
        num_batches = len(batch_ids[0].unique())
        num_layers = len(result['ppn_layers'][0])
        total_loss = 0
        total_acc = 0
        device = segment_label[0].device

        res = {
            'reg_loss': 0.,
            'mask_loss': 0.,
            'type_loss': 0.,
            'classify_endpoints_loss': 0.,
            'output_mask_accuracy': 0.,
            'type_accuracy': 0.,
            'classify_endpoints_accuracy': 0.,
            'num_positives': 0.,
            'num_voxels': 0.
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
            ppn_points = result['ppn_points'][igpu]
            loss_gpu, acc_gpu = 0.0, 0.0
            for layer in range(len(ppn_layers)):
                # print("Layer = ", layer)
                ppn_score_layer = ppn_layers[layer]
                coords_layer = ppn_coords[layer]
                loss_layer = 0.0

                acc_layer = 0.0

                for b in batch_ids[igpu].int().unique():

                    batch_index_layer = coords_layer[:, 0].int() == b
                    batch_particle_index = batch_ids[igpu].int() == b
                    points_label = particles[particles[:, 0].int() == b][:, 1:4]
                    scores_event = ppn_score_layer[batch_index_layer].squeeze()
                    points_event = coords_layer[batch_index_layer]
                    if len(scores_event.shape) == 0:
                        continue

                    d_true = local_cdist(
                        points_label,
                        points_event[:, 1:4].float().to(device))

                    d_positives = (d_true < self.resolution * \
                                   2**(len(ppn_layers) - layer)).any(dim=0)

                    weight_ppn = get_ppn_weights(d_positives, self.ppn_weighing_mode)

                    loss_batch = self.lossfn(scores_event,
                                             d_positives.float(),
                                             weight=weight_ppn,
                                             reduction='mean')

                    with torch.no_grad():
                        acc = ((scores_event > 0).long() == d_positives.long()).sum() \
                                                          / float(scores_event.shape[0])
                        acc_layer += float(acc) / num_batches

                    loss_layer += loss_batch
                    if layer == len(ppn_layers)-1:

                        # Get Final Layers
                        anchors = coords_layer[batch_particle_index][:, 1:4].float().to(device) + 0.5
                        pixel_score = ppn_points[batch_particle_index][:, -1]
                        pixel_logits = ppn_points[batch_particle_index][:, 3:8]
                        pixel_pred = ppn_points[batch_particle_index][:, :3] + anchors

                        d = local_cdist(points_label, pixel_pred)
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

                        with torch.no_grad():
                            mask_final_acc = ((pixel_score > 0).long() == positives.long()).sum()\
                                        / float(pixel_score.shape[0])
                            res['output_mask_accuracy'] += float(mask_final_acc) / float(num_batches)
                            res['num_positives'] += float(torch.sum(positives)) / float(num_batches)
                            res['num_voxels'] += float(pixel_pred.shape[0]) / float(num_batches)

                        # Type Segmentation Loss
                        # d = local_cdist(points_label, pixel_pred)
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
                        with torch.no_grad():
                            pred_type_labels = torch.argmax(pixel_logits[positives], dim=1)
                            type_acc = float(torch.sum(pred_type_labels.long() == positive_labels.long()) / float(pred_type_labels.shape[0]))
                            res['type_accuracy'] += type_acc / float(num_batches)

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
                                        pred = result['ppn_classify_endpoints'][igpu][batch_index_layer][point_class_positives]
                                        tracks = event_types_label[point_class_index] == self._track_label
                                        if tracks.sum().item():
                                            loss_point_class += torch.mean(self.segloss(pred[tracks], true[tracks].long()))
                                            acc_point_class += (torch.argmax(pred[tracks], dim=-1) == true[tracks]).sum().item() / float(true[tracks].nelement())
                                            point_class_count += 1

                                if point_class_count:
                                    loss_classify_endpoints = loss_point_class / point_class_count
                                    acc_classify_endpoints = acc_point_class / point_class_count
                                    #total_loss += loss_classify_endpoints.float()
                            res['classify_endpoints_loss'] += self.classify_endpoints_loss_weight * float(loss_classify_endpoints) / num_batches
                            res['classify_endpoints_accuracy'] += float(acc_classify_endpoints) / num_batches
                        # --- end of Endpoint classification

                        # Distance Loss
                        d2, _ = torch.min(distance_positives, dim=0)
                        reg_loss = d2.mean()
                        res['reg_loss'] += float(self.reg_loss_weight * reg_loss) / num_batches if num_batches else float(self.reg_loss_weight * reg_loss)
                        res['type_loss'] += float(self.point_type_loss_weight * type_loss) / num_batches if num_batches else float(self.point_type_loss_weight * type_loss)
                        res['mask_loss'] += float(self.mask_loss_weight * mask_loss_final) / num_batches if num_batches else float(self.mask_loss_weight * mask_loss_final)
                        if num_batches:
                            total_loss += (self.reg_loss_weight * reg_loss \
                                        + self.point_type_loss_weight * type_loss \
                                        + self.mask_loss_weight * mask_loss_final) / num_batches 
                        else:
                            total_loss += (self.reg_loss_weight * reg_loss \
                                        + self.point_type_loss_weight * type_loss \
                                        + self.mask_loss_weight * mask_loss_final)
                        if self._classify_endpoints:
                            if num_batches:
                                total_loss += self.classify_endpoints_loss_weight * loss_classify_endpoints / num_batches
                            else:
                                total_loss += self.classify_endpoints_loss_weight * loss_classify_endpoints

                loss_layer /= max(1, num_batches)
                loss_gpu += self.mask_loss_weight * loss_layer

            loss_gpu /= len(ppn_layers)
            total_loss += loss_gpu

        # We have to add the mask loss from each layers.
        res['mask_loss'] += float(loss_gpu)

        total_acc = total_acc / num_batches if num_batches else 1.
        res['loss'] = total_loss
        res['accuracy'] = float(total_acc)
        return res
