import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.ppn import define_ppn12
from pprint import pprint
from collections import defaultdict


class AddLabels(nn.Module):
    def __init__(self):
        super(AddLabels, self).__init__()

    def forward(self, attention, label):
        features = attention.F.new().resize_(1).expand_as(attention.F).fill_(1.0)
        features = features * attention.F
        coords = attention.C
        output = ME.SparseTensor(
            coordinates=coords, features=features,
            coords_key=attention.coords_key,
            coords_manager=attention.coords_man)
        for l in label:
            index = (coords == l).all(dim=1)
            output.F[index] = 1.0
        return output


# class Multiply(nn.Module):
#     def __init__(self):
#         super(Multiply, self).__init__()
#
#     def forward(self, x, y):

# class Selection(nn.Module):
#     '''
#     Inputs:
#         - logits (ME.SparseTensor): SparseTensor with N x 2 PPN
#             score feature map.

#     Returns:
#         - out (ME.SparseTensor): SparseTensor where coordinates with score
#         less than the threshold is pruned.
#     '''
#     def __init__(self, threshold=0.8):
#         super(Selection, self).__init__()
#         self.threshold = threshold
#         self.softmax = nn.Softmax(dim=1)
#         self.prune = ME.MinkowskiPruning()

#     def forward(self, logits):
#         with torch.no_grad():
#             mask = (self.softmax(logits.F)[:, 1] > self.threshold).cpu()
#             out = self.prune(logits, mask)
#             return out


# def get_target(out, target_key, kernel_size=1):
#     with torch.no_grad():
#         target = torch.zeros(len(out), dtype=torch.bool)
#         cm = out.coords_man
#         strided_target_key = cm.stride(
#             target_key, out.tensor_stride[0], force_creation=True)
#         ins, outs = cm.get_kernel_map(
#             out.coords_key,
#             strided_target_key,
#             kernel_size=kernel_size,
#             region_type=1)
#         for curr_in in ins:
#             target[curr_in] = 1
#     return target


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
        # print(x.F.shape)
        device = x.F.device
        # Create a mask sparse tensor in x-coordinates
        x0 = ME.SparseTensor(
            coordinates=x.C,
            features=torch.zeros(x.F.shape[0], mask.F.shape[1]).to(device),
            coords_manager=x.coords_man,
            force_creation=True,
            tensor_stride=x.tensor_stride
        )
        mask_in_xcoords = x0 + mask
        # print(mask_in_xcoords.F.shape)
        x_expanded = ME.SparseTensor(
            coordinates=mask_in_xcoords.C,
            features=torch.zeros(mask_in_xcoords.F.shape[0], x.F.shape[1]).to(device),
            coords_manager=x.coords_man,
            force_creation=True,
            tensor_stride=x.tensor_stride
        )
        x_expanded = x_expanded + x
        # print(x_expanded.F.shape)
        target = (mask_in_xcoords.F > self.score_threshold).squeeze() & (x_expanded.F > 0).any(dim=1)
        # print((mask_in_xcoords.F > self.score_threshold).squeeze())
        # print((x_expanded.F > 0).any(dim=1))
        # print(target)
        x_pruned = self.prune(x_expanded, target.squeeze().cpu())
        return x_pruned



class PPN(nn.Module):
    '''
    MinkowskiEngine implemnetation of PPN.

    There are some major differences in architecture design, due to
    limitations of MinkowskiEngine (it does not support generating new coords
    with unpooling). 
    '''
    def __init__(self, cfg, name='ppn'):
        super(PPN, self).__init__()
        self.model_config = cfg[name]

        print("PPN Config = ", self.model_config)

        self._dimension = self.model_config.get('data_dim', 3)
        self._num_strides = self.model_config.get('num_strides', 5)
        m = self.model_config.get('filters', 16)  # Unet number of features
        self._num_classes = self.model_config.get('num_classes', 5)
        self._downsample_ghost = self.model_config.get('downsample_ghost', False)
        self._use_encoding = self.model_config.get('use_encoding', False)
        self._use_true_ghost_mask = self.model_config.get('use_true_ghost_mask', False)
        self._ppn_num_conv = self.model_config.get('ppn_num_conv', 1)
        self._ppn1_size = self.model_config.get('ppn1_size', -1)
        self._ppn2_size = self.model_config.get('ppn2_size', -1)
        self._spatial_size = self.model_config.get('spatial_size', 512)
        self._ppn_threshold = self.model_config.get('score_threshold', 0.8)
        self.ppn1_stride, self.ppn2_stride = define_ppn12(
            self._ppn1_size, self._ppn2_size,
            self._spatial_size, self._num_strides)

        self.deepest = self.model_config.get('deepest_layer_feature_size', -1)

        kernel_size = 2  # Use input_spatial_size method for other values?
        # UNet number of features per level
        nPlanes = [i*m for i in range(1, self._num_strides+1)]
        if self.deepest > 0:
            nPlanes[-1] = self.deepest
        # UNet number of features per level (multiplicative)
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]
        # downsample = [filter size, filter stride]
        downsample = [kernel_size, 2]

        self.ppn1_conv = []
        for i in range(self._ppn_num_conv):
            self.ppn1_conv.append(
                ME.MinkowskiConvolution(
                    nPlanes[self.ppn1_stride-self._num_strides],
                    nPlanes[self.ppn1_stride-self._num_strides],
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=self._dimension))
        self.ppn1_conv = nn.Sequential(*self.ppn1_conv)
        self.ppn1_scores = ME.MinkowskiConvolution(
            nPlanes[self.ppn1_stride-self._num_strides], 1,
            kernel_size=3,
            bias=False,
            dimension=self._dimension)

        self.unpool1 = []
        self.unpool_norm1 = nn.ModuleList()
        for i in range(self.ppn1_stride-self.ppn2_stride):
            self.unpool_norm1.append(
                nn.Sequential(
                    ME.MinkowskiBatchNorm(
                        nPlanes[self.ppn1_stride-self._num_strides+i]),
                    ME.MinkowskiELU(),
                ))
            self.unpool1.append(
                ME.MinkowskiConvolutionTranspose(
                    nPlanes[self.ppn1_stride-self._num_strides+i],
                    (nPlanes[self.ppn1_stride-self._num_strides+i+1] \
                        if i != self.ppn1_stride-self.ppn2_stride-1 else 1),
                    kernel_size=downsample[0], stride=downsample[1],
                    dimension=self._dimension, generate_new_coords=True))
        self.unpool1 = nn.Sequential(*self.unpool1)

        self.unpool2 = []
        self.unpool_norm2 = nn.ModuleList()
        for i in range(self.ppn2_stride):
            self.unpool2.append(
                ME.MinkowskiConvolutionTranspose(
                    nPlanes[self.ppn2_stride-self._num_strides+i],
                    (nPlanes[self.ppn2_stride-self._num_strides+i+1] \
                        if i != self.ppn2_stride-1 else 1),
                    kernel_size=downsample[0], stride=downsample[1],
                    dimension=self._dimension, generate_new_coords=True))
            self.unpool_norm2.append(
                nn.Sequential(
                    ME.MinkowskiBatchNorm(
                        nPlanes[self.ppn2_stride-self._num_strides+i]),
                    ME.MinkowskiELU()))
            # self.unpool2.append(ME.MinkowskiPoolingTranspose(
            #     downsample[0], downsample[1], dimension=self._dimension))
        self.unpool2 = nn.Sequential(*self.unpool2)

        self.ppn3_conv = []
        for i in range(self._ppn_num_conv):
            self.ppn3_conv.append(
                ME.MinkowskiConvolution(nPlanes[0], nPlanes[0],
                    kernel_size=3, bias=False, dimension=self._dimension))
        self.ppn3_conv = nn.Sequential(*self.ppn3_conv)
        self.ppn3_pixel_pred = ME.MinkowskiConvolution(nPlanes[0], self._dimension,
            kernel_size=3, bias=False, dimension=self._dimension)
        self.ppn3_scores = ME.MinkowskiConvolution(nPlanes[0], 1,
            kernel_size=3, bias=False, dimension=self._dimension)
        self.ppn3_type = ME.MinkowskiConvolution(nPlanes[0], self._num_classes,
            kernel_size=3, bias=False, dimension=self._dimension)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.me_sigmoid = ME.MinkowskiSigmoid()

        self.prune = ME.MinkowskiPruning()
        self.attn = AttentionMask(score_threshold=self._ppn_threshold)


    def forward(self, input):
        """
        spatial size of feature_map1 (PPN1) = spatial_size / 2**self.ppn1_stride
        spatial size of feature_map2 (PPN2) = spatial_size / 2**self.ppn2_stride
        spatial size of feature_map3 = spatial_size (original)
        """
        label = None if not 'label' in input else input['label'][:, :-1]
        ppn1_feature_enc = input['ppn_feature_enc']
        ppn1_feature_dec = input['ppn_feature_dec']
        assert len(ppn1_feature_enc) == self._num_strides+1
        assert len(ppn1_feature_dec) == self._num_strides
        #print("PPN1/2 stride = ", self.ppn1_stride, self.ppn2_stride)
        if self._use_encoding:
            feature_map1 = ppn1_feature_enc[self.ppn1_stride]
            feature_map2 = ppn1_feature_enc[self.ppn2_stride]
            feature_map3 = ppn1_feature_enc[0]
        else:
            feature_map1 = ppn1_feature_dec[self._num_strides-1-self.ppn1_stride]
            feature_map2 = ppn1_feature_dec[self._num_strides-1-self.ppn2_stride]
            feature_map3 = ppn1_feature_dec[self._num_strides-1]

        # If ghost mask is present, downsample it and use it before conv
        if self._downsample_ghost:
            with torch.no_grad():
                if self._use_true_ghost_mask:
                    ghost_mask = input['segment_label'] < self._num_classes
                else:
                    ghost_mask = 1.0 - torch.argmax(input['ghost'], dim=1)
                coords = ppn1_feature_enc[0].C
                # TODO:
                feature_map1, ghost_mask1 = self.ghost_mask(
                    ghost_mask, coords, feature_map1, factor=self.ppn1_stride)
                feature_map2, ghost_mask2 = self.ghost_mask(
                    ghost_mask, coords, feature_map2, factor=self.ppn2_stride)
                feature_map3, _ = self.ghost_mask(
                    ghost_mask, coords, feature_map3, factor=0.0)

        # Feature map 1 = deepest
        device = feature_map2.F.device
        x = self.ppn1_conv(feature_map1)
        ppn1_scores = self.ppn1_scores(x)
        mask = (self.sigmoid(ppn1_scores.F) > self._ppn_threshold).cpu()
        x = self.prune(x, mask)
        for i, layer in enumerate(self.unpool1):
            x = self.unpool_norm1[i](x)
            x = layer(x)

        # TODO: Ground Truth Label Training
        if self.training and label is not None:
            with torch.no_grad():
                pass
                # attention = self.add_labels1(attention, \
                # torch.cat([label[:, :-2]/2**self.ppn2_stride, label[:, -2][:, None]], dim=1).long())

        # Feature map 2 = intermediate
        ppn2_scores = x
        attention1 = self.me_sigmoid(x)
        y = self.attn(feature_map2, attention1)
        for i, layer in enumerate(self.unpool2):
            y = self.unpool_norm2[i](y)
            y = layer(y)
        attention2 = self.me_sigmoid(y)
        # print(attention2)

        # TODO: Ground Truth Label Training
        if self.training and label is not None:
            with torch.no_grad():
                attention2 = self.add_labels2(attention2, label[:,:-1].long())

        z = self.attn(feature_map3, attention2)
        # print(z)
        # print(sum(z.F > 0))
        z = self.ppn3_conv(z)
        # print(z, sum(z.F > 0))
        ppn3_pixel_pred = self.ppn3_pixel_pred(z)
        # print("ppn3_pixel_pred = ", ppn3_pixel_pred)
        ppn3_scores = self.ppn3_scores(z)
        # print("ppn3_scores = ", ppn3_scores)
        ppn3_type = self.ppn3_type(z)
        # print("ppn3_type = ", ppn3_type)
        # assert False

        anchors = ME.SparseTensor(
            coordinates=ppn3_pixel_pred.C,
            features=ppn3_pixel_pred.C[:, 1:].float().to(device) + 0.5,
            coords_manager=ppn3_pixel_pred.coords_man,
            force_creation=True
        )

        ppn3_pixel_pred = anchors + ppn3_pixel_pred
        pixel_pred = ppn3_pixel_pred.F
        # print(ppn3_scores)
        scores = ppn3_scores.F
        # print(ppn3_type)
        point_type = ppn3_type.F

        # print(ppn3_pixel_pred.shape)
        # print(pixel_pred.shape)
        # print(scores.shape)
        # print(point_type.shape)

        if torch.cuda.is_available():
            result = {'points' : [torch.cat([ppn3_pixel_pred.C.float().to(device), 
                            pixel_pred, scores, point_type], dim=1)],
                      'ppn1'  : [torch.cat([ppn1_scores.C.cuda().float(), ppn1_scores.F], dim=1)],
                      'ppn2'  : [torch.cat([ppn2_scores.C.cuda().float(), ppn2_scores.F], dim=1)],
                      'mask_ppn1'  : [attention1.F > self._ppn_threshold],
                      'mask_ppn2' : [attention2.F > self._ppn_threshold]}
        else:
            result = {'points' : [torch.cat([ppn3_pixel_pred.C.float().to(device), 
                            pixel_pred, scores, point_type], dim=1)],
                      'ppn1'  : [torch.cat([ppn1_scores.C.float(), ppn1_scores.F], dim=1)],
                      'ppn2'  : [torch.cat([ppn2_scores.C.float(), ppn2_scores.F], dim=1)],
                      'mask_ppn1'  : [attention1.F > self._ppn_threshold],
                      'mask_ppn2' : [attention2.F > self._ppn_threshold]}
        if self._downsample_ghost:
            result['ghost_mask1'] = [ghost_mask1]
            result['ghost_mask2'] = [ghost_mask2]
        return result


class PPNLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, cfg, reduction='sum'):
        super(PPNLoss, self).__init__(reduction=reduction)
        pprint(cfg)
        self.loss_config = cfg['ppn']
        self._dimension = self.loss_config.get('data_dim', 3)
        self._num_strides = self.loss_config.get('num_strides', 5)
        self._num_classes = self.loss_config.get('num_classes', 5)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        self.half_stride = int((self._num_strides-1)/2.0)
        self.half_stride2 = int(self._num_strides/2.0)
        self._downsample_ghost = self.loss_config.get('downsample_ghost', False)
        self._weight_ppn1 = self.loss_config.get('weight_ppn1', 1.0)
        self._weight_ppn2 = self.loss_config.get('weight_ppn2', 1.0)
        self._true_distance_ppn1 = self.loss_config.get('true_distance_ppn1', 1.0)
        self._true_distance_ppn2 = self.loss_config.get('true_distance_ppn2', 1.0)
        self._true_distance_ppn3 = self.loss_config.get('true_distance_ppn3', 5.0)
        self._score_threshold = self.loss_config.get('score_threshold', 0.5)
        self._random_sample_negatives = self.loss_config.get(
            'random_sample_negatives', False)
        self._near_sampling = self.loss_config.get('near_sampling', False)
        self._sampling_factor = self.loss_config.get('sampling_factor', 20)

        self._ppn1_size = self.loss_config.get('ppn1_size', -1)
        self._ppn2_size = self.loss_config.get('ppn2_size', -1)
        self._spatial_size = self.loss_config.get('spatial_size', 512)
        self.ppn1_stride, self.ppn2_stride = define_ppn12(
            self._ppn1_size, self._ppn2_size, 
            self._spatial_size, self._num_strides)

        self.bceloss = nn.functional.binary_cross_entropy_with_logits
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, result, segment_label, particle_label):
        '''
        Forward function for PPNLoss

        INPUTS:
            - result: dict output of PPN
            - segment_label: list of semantic label tensors
            - particle_label: list of PPN point label tensors.
        '''
        assert len(result['points']) == len(particle_label)
        assert len(result['points']) == len(segment_label)

        device = segment_label[0].device
        # print(particle_label)
        batch_ids = [d[:, 0] for d in segment_label]
        total_loss = 0.
        total_acc = 0.
        ppn_count = 0.
        total_distance, total_class = 0., 0.
        total_loss_ppn1, total_loss_ppn2 = 0., 0.
        total_acc_ppn1, total_acc_ppn2 = 0., 0.
        total_fraction_positives_ppn1, total_fraction_negatives_ppn1 = 0., 0.
        total_fraction_positives_ppn2, total_fraction_negatives_ppn2 = 0., 0.
        total_acc_type, total_loss_type = 0., 0.
        num_labels = 0.
        num_discarded_labels_ppn1, num_discarded_labels_ppn2 = 0., 0.
        total_num_positives_ppn1, total_num_positives_ppn2 = 0., 0.
        data_dim = self._dimension

        output_dict = defaultdict(list)

        for i in range(len(segment_label)):
            particles_event = particle_label[i]
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_data = segment_label[i][batch_index][:, 1:data_dim+1]
                ppn1_batch_index = result['ppn1'][i][:, 0] == b.float()
                ppn2_batch_index = result['ppn2'][i][:, 0] == b.float()
                ppn1_batch = result['ppn1'][i][ppn1_batch_index][:, :4]
                ppn2_batch = result['ppn2'][i][ppn2_batch_index][:, :4]

                points_batch_index = result['points'][i][:, 0] == b
                event_pixel_pred = result['points'][i][points_batch_index][:, 4:4+data_dim]
                event_scores = result['points'][i][points_batch_index][:, 4+data_dim:(4+data_dim+1)]
                event_types = result['points'][i][points_batch_index][:, (4+data_dim+1):]
                ppn1_scores = result['ppn1'][i][ppn1_batch_index][:, 4:]
                ppn2_scores = result['ppn2'][i][ppn2_batch_index][:, 4:]

                event_label = particles_event[particles_event[:, 0] == b][:, 1:4]
                if event_label.shape[0] < 1:
                    continue
                output_dict['num_labels'].append(event_label.shape[0])
                event_types_label = particles_event[particles_event[:, 0] == b][:, -2]
                ppn_loss = 0

                d = self.distances(event_label, event_pixel_pred)
                d_true = self.distances(event_label, event_data)
                positives = (d < self._true_distance_ppn3).any(dim=0) 
                num_positives = positives.long().sum()
                num_negatives = positives.nelement() - num_positives
                w = num_positives.float() / (num_positives + num_negatives).float()
                weight_ppn3 = torch.zeros(positives.shape[0]).to(device)
                weight_ppn3[positives] = 1 - w
                weight_ppn3[~positives] = w

                # PPN1 Segmentation 
                distances_ppn1 = self.distances(event_label, ppn1_batch[:, 1:])
                positives_ppn1 = (distances_ppn1 < self._true_distance_ppn3 \
                    * 2**self.ppn1_stride).any(dim=0)

                num_positives_ppn1 = positives_ppn1.long().sum()
                num_negatives_ppn1 = positives_ppn1.nelement() - num_positives_ppn1
                w1 = num_positives_ppn1.float() / \
                    (num_positives_ppn1 + num_negatives_ppn1).float()
                weight_ppn1 = torch.zeros(positives_ppn1.shape[0]).to(device)
                weight_ppn1[~positives_ppn1] = w1
                weight_ppn1[positives_ppn1] = 1 - w1

                pred_ppn1 = ppn1_scores.squeeze() > 0
                if pred_ppn1.nelement() > 0:
                    acc_ppn1 = (pred_ppn1 == positives_ppn1).sum().item() / \
                        float(pred_ppn1.nelement())
                    ppn1_loss = self.bceloss(ppn1_scores.squeeze(1), 
                        positives_ppn1.float(), 
                        weight=weight_ppn1, reduction='mean')
                    ppn_loss += self._weight_ppn1 * ppn1_loss
                    output_dict['loss_ppn1'].append(float(ppn1_loss))
                    output_dict['acc_ppn1'].append(float(acc_ppn1))
                    output_dict['num_positives_ppn1'].append(int(num_positives_ppn1))

                # PPN2 Segmentation 
                distances_ppn2 = self.distances(event_label, ppn2_batch[:, 1:])
                positives_ppn2 = (distances_ppn2 < self._true_distance_ppn3 \
                    * 2**self.ppn2_stride).any(dim=0)

                num_positives_ppn2 = positives_ppn2.long().sum()
                num_negatives_ppn2 = positives_ppn2.nelement() - num_positives_ppn2
                w2 = num_positives_ppn2.float() / (num_positives_ppn2 + num_negatives_ppn2).float()
                weight_ppn2 = torch.zeros(positives_ppn2.shape[0]).to(device)
                weight_ppn2[~positives_ppn2] = w2
                weight_ppn2[positives_ppn2] = 1 - w2
                
                pred_ppn2 = ppn2_scores.squeeze() > 0

                if pred_ppn2.nelement() > 0:
                    acc_ppn2 = (pred_ppn2 == positives_ppn2).sum().item() / \
                        float(pred_ppn2.nelement())
                    ppn2_loss = self.bceloss(ppn2_scores.squeeze(1), 
                        positives_ppn2.float(), 
                        weight=weight_ppn2, reduction='mean')
                    ppn_loss += self._weight_ppn2 * ppn2_loss
                    output_dict['loss_ppn2'].append(float(ppn2_loss))
                    output_dict['acc_ppn2'].append(float(acc_ppn2))
                    output_dict['num_positives_ppn2'].append(int(num_positives_ppn2))

                # PPN3 (Final) Segmentation
                pred_ppn3 = event_scores.squeeze() > 0

                if pred_ppn3.nelement() > 0 and positives.shape[0] > 0:
                    acc_ppn3 = (pred_ppn3 == positives).sum().item() / \
                        float(pred_ppn3.nelement())
                    ppn3_loss = self.bceloss(event_scores.squeeze(1), 
                        positives.float(), weight=weight_ppn3)
                    ppn_loss += ppn3_loss
                    output_dict['loss_class'].append(float(ppn3_loss))
                    output_dict['ppn_acc'].append(float(acc_ppn3))

                    types_indices = torch.argmin(d, dim=0)
                    event_semantic_label = event_types_label[types_indices]
                    loss_seg = self.cross_entropy(event_types, event_semantic_label.long())
                    pred_seg = torch.argmax(event_types, dim=1)
                    acc_seg = float((pred_seg.long() == event_semantic_label.long()).sum()) \
                        / float(pred_seg.shape[0])
                    output_dict['loss_type'].append(loss_seg)
                    output_dict['acc_ppn_type'].append(float(acc_seg))

                ppn_count += 1
                output_dict['ppn_loss'].append(ppn_loss)
        
        for key, val in output_dict.items():
            if len(val) > 0:
                output_dict[key] = sum(val) / len(val)
            else:
                output_dict[key] = 0
        pprint(output_dict)

        return output_dict