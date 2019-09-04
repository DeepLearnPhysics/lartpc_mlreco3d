from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.layers.extract_feature_map import Selection, Multiply, AddLabels, GhostMask
import numpy as np

def define_ppn12(ppn1_size, ppn2_size, spatial_size, num_strides):
    if ppn1_size == -1:
        ppn1_size = spatial_size/2**4
    if ppn2_size == -1:
        ppn2_size = spatial_size/2**2
    # Make sure spatial sizes match
    if ppn1_size < 0 or ppn1_size > spatial_size:
        raise Exception("PPN1 size must be within [0, %d], got %d" % (spatial_size, ppn1_size))
    if ppn2_size < 0 or ppn2_size > spatial_size:
        raise Exception("PPN2 size must be within [0, %d], got %d" % (spatial_size, ppn2_size))
    if spatial_size % ppn1_size != 0:
        raise Exception("PPN1 size must divide original spatial size (got %d and %d)" % (spatial_size, ppn1_size))
    if spatial_size % ppn2_size != 0:
        raise Exception("PPN2 size must divide the original spatial size (got %d and %d)" % (spatial_size, ppn2_size))
    if ppn1_size >= ppn2_size:
        raise Exception("PPN1 size must be greater than PPN2 size (got %d and %d respectively)" % (ppn1_size, ppn2_size))
    ppn2_stride = int(np.log2(spatial_size / ppn2_size))
    ppn1_stride = int(np.log2(spatial_size / ppn1_size))
    if ppn1_stride > num_strides:
        raise Exception("Depth (number of strides) must be great enough for PPN1 size %d" % ppn1_size)
    return ppn1_stride, ppn2_stride


class PPN(torch.nn.Module):
    """
    Point Prediction Network (PPN)

    PPN is *not* a standalone network. It is an additional layer to go on top of
    a UResNet-style network that provides feature maps on top of which PPN will
    run some convolutions and predict point positions.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    """
    def __init__(self, cfg):
        super(PPN, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg['modules']['ppn']

        self._dimension = self._model_config.get('data_dim', 3)
        self._num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        num_classes = self._model_config.get('num_classes', 5)
        self._downsample_ghost = self._model_config.get('downsample_ghost', False)
        self._use_encoding = self._model_config.get('use_encoding', False)
        self._ppn_num_conv = self._model_config.get('ppn_num_conv', 1)
        self._ppn1_size = self._model_config.get('ppn1_size', -1)
        self._ppn2_size = self._model_config.get('ppn2_size', -1)
        self._spatial_size = self._model_config.get('spatial_size', 512)
        self.ppn1_stride, self.ppn2_stride = define_ppn12(self._ppn1_size, self._ppn2_size, self._spatial_size, self._num_strides)

        kernel_size = 2  # Use input_spatial_size method for other values?
        nPlanes = [i*m for i in range(1, self._num_strides+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]# downsample = [filter size, filter stride]

        # PPN stuff
        #self.half_stride = int((self._num_strides-1)/2.0)
        #self.half_stride2 = int(self._num_strides/2.0)
        self.ppn1_conv = scn.Sequential()
        for i in range(self._ppn_num_conv):
            self.ppn1_conv.add(scn.SubmanifoldConvolution(self._dimension, nPlanes[self.ppn1_stride-self._num_strides], nPlanes[self.ppn1_stride-self._num_strides], 3, False))
        self.ppn1_scores = scn.SubmanifoldConvolution(self._dimension, nPlanes[self.ppn1_stride-self._num_strides], 2, 3, False)

        self.selection1 = Selection()
        self.selection2 = Selection()
        self.unpool1 = scn.Sequential()
        for i in range(self.ppn1_stride-self.ppn2_stride):
            self.unpool1.add(scn.UnPooling(self._dimension, downsample[0], downsample[1]))

        self.unpool2 = scn.Sequential()
        for i in range(self.ppn2_stride):
            self.unpool2.add(scn.UnPooling(self._dimension, downsample[0], downsample[1]))

        #middle_filters = int(m * self.half_stride * (self.half_stride + 1) / 2.0)
        #print(middle_filters, (self.ppn2_stride+1)*m)
        middle_filters = (self.ppn2_stride+1)*m
        self.ppn2_conv = scn.Sequential()
        for i in range(self._ppn_num_conv):
            self.ppn2_conv.add(scn.SubmanifoldConvolution(self._dimension, middle_filters, middle_filters, 3, False))
        self.ppn2_scores = scn.SubmanifoldConvolution(self._dimension, middle_filters, 2, 3, False)
        self.multiply1 = Multiply()
        self.multiply2 = Multiply()

        self.ppn3_conv = scn.Sequential()
        for i in range(self._ppn_num_conv):
            self.ppn3_conv.add(scn.SubmanifoldConvolution(self._dimension, nPlanes[0], nPlanes[0], 3, False))
        self.ppn3_pixel_pred = scn.SubmanifoldConvolution(self._dimension, nPlanes[0], self._dimension, 3, False)
        self.ppn3_scores = scn.SubmanifoldConvolution(self._dimension, nPlanes[0], 2, 3, False)
        self.ppn3_type = scn.SubmanifoldConvolution(self._dimension, nPlanes[0], num_classes, 3, False)


        self.add_labels1 = AddLabels()
        self.add_labels2 = AddLabels()

        self.ghost_mask = GhostMask()

    def forward(self, input):
        """
        spatial size of feature_map1 (PPN1) = spatial_size / 2**self.ppn1_stride
        spatial size of feature_map2 (PPN2) = spatial_size / 2**self.ppn2_stride
        spatial size of feature_map3 = spatial_size (original)
        """
        label = input['label'][:, :-1]
        ppn1_feature_enc = input['ppn_feature_enc']
        ppn1_feature_dec = input['ppn_feature_dec']
        assert len(ppn1_feature_enc) == self._num_strides+1
        assert len(ppn1_feature_dec) == self._num_strides
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
                ghost_mask = 1.0 - torch.argmax(input['ghost'], dim=1)
                coords = ppn1_feature_enc[0].get_spatial_locations()
                feature_map1, ghost_mask1 = self.ghost_mask(ghost_mask, coords, feature_map1, factor=self.ppn1_stride)
                feature_map2, ghost_mask2 = self.ghost_mask(ghost_mask, coords, feature_map2, factor=self.ppn2_stride)
                feature_map3, _ = self.ghost_mask(ghost_mask, coords, feature_map3, factor=0.0)

        # Feature map 1 = deepest
        x = self.ppn1_conv(feature_map1)
        ppn1_scores = self.ppn1_scores(x)
        mask = self.selection1(ppn1_scores)
        attention = self.unpool1(mask)
        if self.training:
            with torch.no_grad():
                attention = self.add_labels1(attention, torch.cat([label[:, :-1]/2**self.ppn2_stride, label[:, -1][:, None]], dim=1).long())

        # Feature map 2 = intermediate
        y = self.multiply1(feature_map2, attention)
        y = self.ppn2_conv(y)
        ppn2_scores = self.ppn2_scores(y)
        mask2 = self.selection2(ppn2_scores)
        attention2 = self.unpool2(mask2)
        if self.training:
            with torch.no_grad():
                attention2 = self.add_labels2(attention2, label.long())

        # Feature map 3 = original spatial size
        z = self.multiply2(feature_map3, attention2)
        z = self.ppn3_conv(z)
        ppn3_pixel_pred = self.ppn3_pixel_pred(z)
        ppn3_scores = self.ppn3_scores(z)
        ppn3_type = self.ppn3_type(z)
        pixel_pred = ppn3_pixel_pred.features
        scores = ppn3_scores.features
        point_type = ppn3_type.features

        if torch.cuda.is_available():
            result = {'points' : [torch.cat([pixel_pred, scores, point_type], dim=1)],
                      'ppn1'  : [torch.cat([ppn1_scores.get_spatial_locations().cuda().float(), ppn1_scores.features], dim=1)],
                      'ppn2'  : [torch.cat([ppn2_scores.get_spatial_locations().cuda().float(), ppn2_scores.features], dim=1)],
                      'mask_ppn1'  : [attention.features],
                      'mask_ppn2' : [attention2.features]}
        else:
            result = {'points' : [torch.cat([pixel_pred, scores, point_type], dim=1)],
                      'ppn1'  : [torch.cat([ppn1_scores.get_spatial_locations().float(), ppn1_scores.features], dim=1)],
                      'ppn2'  : [torch.cat([ppn2_scores.get_spatial_locations().float(), ppn2_scores.features], dim=1)],
                      'mask_ppn1'  : [attention.features],
                      'mask_ppn2' : [attention2.features]}
        if self._downsample_ghost:
            result['ghost_mask1'] = [ghost_mask1]
            result['ghost_mask2'] = [ghost_mask2]
        return result


class PPNLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg, reduction='sum'):
        super(PPNLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['modules']['ppn']
        self._dimension = self._cfg.get('data_dim', 3)
        self._num_strides = self._cfg.get('num_strides', 5)
        self._num_classes = self._cfg.get('num_classes', 5)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        self.half_stride = int((self._num_strides-1)/2.0)
        self.half_stride2 = int(self._num_strides/2.0)
        self._downsample_ghost = self._cfg.get('downsample_ghost', False)
        self._weight_ppn1 = self._cfg.get('weight_ppn1', 1.0)
        self._true_distance_ppn1 = self._cfg.get('true_distance_ppn1', 1.0)
        self._true_distance_ppn2 = self._cfg.get('true_distance_ppn2', 1.0)
        self._true_distance_ppn3 = self._cfg.get('true_distance_ppn3', 5.0)
        self._score_threshold = self._cfg.get('score_threshold', 0.5)

        self._ppn1_size = self._cfg.get('ppn1_size', -1)
        self._ppn2_size = self._cfg.get('ppn2_size', -1)
        self._spatial_size = self._cfg.get('spatial_size', 512)
        self.ppn1_stride, self.ppn2_stride = define_ppn12(self._ppn1_size, self._ppn2_size, self._spatial_size, self._num_strides)

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, result, label, particles):
        """
        result[0], label and weight are lists of size #gpus = batch_size.
        result has only 1 element because UResNet returns only 1 element.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        weight can be None.
        """
        assert len(result['points']) == len(particles)
        assert len(result['points']) == len(label)
        batch_ids = [d[:, -2] for d in label]
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
        data_dim = self._dimension
        for i in range(len(label)):
            event_particles = particles[i]
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_data = label[i][batch_index][:, :data_dim]  # (N, 3)
                ppn1_batch_index = result['ppn1'][i][:, -3] == b.float()
                ppn2_batch_index = result['ppn2'][i][:, -3] == b.float()
                event_ppn1_data = result['ppn1'][i][ppn1_batch_index][:, :-3]  # (N1, 3)
                event_ppn2_data = result['ppn2'][i][ppn2_batch_index][:, :-3]  # (N2, 3)
                anchors = (event_data + 0.5).float()

                event_pixel_pred = result['points'][i][batch_index][:, :data_dim] + anchors # (N, 3)
                event_scores = result['points'][i][batch_index][:, data_dim:(data_dim+2)]  # (N, 2)
                event_types = result['points'][i][batch_index][:, (data_dim+2):]  # (N, num_classes)
                event_ppn1_scores = result['ppn1'][i][ppn1_batch_index][:, -2:]  # (N1, 2)
                event_ppn2_scores = result['ppn2'][i][ppn2_batch_index][:, -2:]  # (N2, 2)

                # PPN stuff
                event_label = event_particles[event_particles[:, -2] == b][:, :-2]  # (N_gt, 3)
                event_types_label = event_particles[event_particles[:, -2] == b][:, -1]
                # print(b, event_label.size())
                if event_label.size(0) > 0:
                    # Mask: only consider pixels that were selected
                    event_mask = result['mask_ppn2'][i][batch_index]
                    event_mask = (~(event_mask == 0)).any(dim=1)  # (N,)

                    if self._downsample_ghost:
                        event_ghost = 1.0-torch.argmax(result['ghost'][i][batch_index], dim=1)
                        # event_ghost = label[i][batch_index][:, -1] < self._num_classes
                        event_mask = event_mask & event_ghost.byte()

                    if event_mask.int().sum() == 0:
                        continue

                    # event_label = event_label[event_mask]
                    # event_segmentation = event_segmentation[event_mask]
                    event_pixel_pred = event_pixel_pred[event_mask]
                    event_scores = event_scores[event_mask]
                    event_types = event_types[event_mask]
                    event_data = event_data[event_mask]

                    # Mask for PPN2 (intermediate)
                    # We mask by ghost predictions + ppn1 predictions
                    event_ppn2_mask = (~(result['mask_ppn1'][i][ppn2_batch_index] == 0)).any(dim=1)
                    # event_ppn2_mask = torch.nn.functional.softmax(result['ppn2'][i][ppn2_batch_index][:, 4:], dim=1)[:, 1] > self._score_threshold
                    if self._downsample_ghost:
                        event_ppn2_mask = event_ppn2_mask & result['ghost_mask2'][i][ppn2_batch_index].byte().reshape((-1,))
                        # l = label[i][batch_index]
                        # coords = torch.floor(l[:, :3] / float(2**self.half_stride) ).cpu().detach().numpy()
                        # _, indices = np.unique(coords, axis=0, return_index=True)
                        # l = torch.cat([torch.floor(l[indices][:, :3]/float(2**self.half_stride)), l[indices][:, 3:]], dim=1)

                        # coords = event_ppn2_data.cpu().detach().numpy()
                        # perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
                        # inv_perm = np.argsort(perm)
                        # event_ppn2_mask = event_ppn2_mask & (l[inv_perm][:, -1] < self._num_classes)
                    if event_ppn2_mask.int().sum() == 0:
                        continue
                    event_ppn2_data = event_ppn2_data[event_ppn2_mask]
                    event_ppn2_scores = event_ppn2_scores[event_ppn2_mask]

                    # Mask for PPN1 (coarsest)
                    # predicted ghost mask applied at this spatial size
                    # event_mask_ppn1 = torch.nn.functional.softmax(result['ppn1'][i][ppn1_batch_index][:, 4:], dim=1)[:, 1] > self._score_threshold
                    if self._downsample_ghost:
                        event_mask_ppn1 = result['ghost_mask1'][i][ppn1_batch_index].byte().reshape((-1,))
                        event_ppn1_data = event_ppn1_data[event_mask_ppn1]
                        event_ppn1_scores = event_ppn1_scores[event_mask_ppn1]
                        if event_mask_ppn1.int().sum() == 0:
                            continue

                    # Segmentation loss (predict positives)
                    d = self.distances(event_label, event_pixel_pred)
                    d_true = self.distances(event_label, event_data)
                    positives = (d_true < self._true_distance_ppn3).any(dim=0)  # FIXME can be empty
                    if positives.shape[0] == 0:
                        continue
                    loss_seg = torch.mean(self.cross_entropy(event_scores.double(), positives.long()))
                    total_class += loss_seg

                    # Accuracy for scores
                    predicted_labels = torch.argmax(event_scores, dim=-1)
                    acc = (predicted_labels == positives.long()).sum().item() / float(predicted_labels.nelement())

                    # Loss ppn1 & ppn2 (predict positives)
                    # print(event_label.size())
                    event_label_ppn1 = torch.floor(event_label/float(2**self.ppn1_stride))
                    event_label_ppn2 = torch.floor(event_label/float(2**self.ppn2_stride))
                    d_true_ppn1 = self.distances(event_label_ppn1, event_ppn1_data)
                    d_true_ppn2 = self.distances(event_label_ppn2, event_ppn2_data)
                    positives_ppn1 = (d_true_ppn1 < self._true_distance_ppn1).any(dim=0).long()
                    positives_ppn2 = (d_true_ppn2 < self._true_distance_ppn2).any(dim=0).long()

                    num_positives_ppn1 = positives_ppn1.sum()
                    num_negatives_ppn1 = positives_ppn1.nelement() - num_positives_ppn1
                    w = num_positives_ppn1.float() / (num_positives_ppn1 + num_negatives_ppn1).float()
                    weight_ppn1 = torch.stack([w, 1-w]).double()

                    num_positives_ppn2 = positives_ppn2.sum()
                    num_negatives_ppn2 = positives_ppn2.nelement() - num_positives_ppn2
                    w2 = num_positives_ppn2.float() / (num_positives_ppn2 + num_negatives_ppn2).float()
                    weight_ppn2 = torch.stack([w2, 1-w2]).double()
                    # print('num positives ppn1', num_positives_ppn1)
                    # print((~(d_true_ppn1 < self._true_distance).any(dim=1)).sum())
                    # print((~(d_true_ppn2 < self._true_distance).any(dim=1)).sum())
                    # print(event_label_ppn1[~(d_true_ppn1 < self._true_distance).any(dim=1)]*16.0)
                    # print(d_true_ppn1[~(d_true_ppn1 < self._true_distance).any(dim=1)].min(dim=1))
                    # print(event_ppn1_data[positives_ppn1.byte()]*16.0)
                    num_labels += event_label.size(0)
                    num_discarded_labels_ppn1 += (~(d_true_ppn1 < self._true_distance_ppn1).any(dim=1)).sum().item()
                    num_discarded_labels_ppn2 += (~(d_true_ppn2 < self._true_distance_ppn2).any(dim=1)).sum().item()

                    # loss_seg_ppn1 = torch.mean(self.cross_entropy(event_ppn1_scores.double(), positives_ppn1))
                    loss_seg_ppn1 = torch.nn.functional.cross_entropy(event_ppn1_scores.double(), positives_ppn1, weight=weight_ppn1)
                    # loss_seg_ppn2 = torch.mean(self.cross_entropy(event_ppn2_scores.double(), positives_ppn2))
                    loss_seg_ppn2 = torch.nn.functional.cross_entropy(event_ppn2_scores.double(), positives_ppn2, weight=weight_ppn2)
                    predicted_labels_ppn1 = torch.argmax(event_ppn1_scores, dim=-1)
                    predicted_labels_ppn2 = torch.argmax(event_ppn2_scores, dim=-1)
                    acc_ppn1 = (predicted_labels_ppn1 == positives_ppn1.long()).sum().item() / float(predicted_labels_ppn1.nelement())
                    acc_ppn2 = (predicted_labels_ppn2 == positives_ppn2.long()).sum().item() / float(predicted_labels_ppn2.nelement())
                    if predicted_labels_ppn1[positives_ppn1 > 0].nelement() > 0:
                        fraction_positives_ppn1 = (predicted_labels_ppn1[positives_ppn1 > 0] == positives_ppn1[positives_ppn1 > 0].long()).sum().item() / float(predicted_labels_ppn1[positives_ppn1 > 0].nelement())
                        total_fraction_positives_ppn1 += fraction_positives_ppn1
                    if predicted_labels_ppn1[positives_ppn1 == 0].nelement() > 0:
                        fraction_negatives_ppn1 = (predicted_labels_ppn1[positives_ppn1 == 0] == positives_ppn1[positives_ppn1 == 0].long()).sum().item() / float(predicted_labels_ppn1[positives_ppn1 == 0].nelement())
                        total_fraction_negatives_ppn1 += fraction_negatives_ppn1
                    if predicted_labels_ppn2[positives_ppn2 > 0].nelement() > 0:
                        fraction_positives_ppn2 = (predicted_labels_ppn2[positives_ppn2 > 0] == positives_ppn2[positives_ppn2 > 0].long()).sum().item() / float(predicted_labels_ppn2[positives_ppn2 > 0].nelement())
                        total_fraction_positives_ppn2 += fraction_positives_ppn2
                    if predicted_labels_ppn2[positives_ppn2 == 0].nelement() > 0:
                        fraction_negatives_ppn2 = (predicted_labels_ppn2[positives_ppn2 == 0] == positives_ppn2[positives_ppn2 == 0].long()).sum().item() / float(predicted_labels_ppn2[positives_ppn2 == 0].nelement())
                        total_fraction_negatives_ppn2 += fraction_negatives_ppn2
                    #print(num_positives_ppn1, num_negatives_ppn1, w, acc_ppn1)
                    #total_num_positives_ppn1 += num_positives_ppn1
                    #total_num_negatives_ppn1 += num_negatives_ppn1
                    # Distance loss
                    # positives = (d_true[:, event_mask] < 5).any(dim=0)
                    # distances_positives = d[:, event_mask][:, positives]
                    distances_positives = d[:, positives]
                    if distances_positives.shape[1] > 0:
                        d2, _ = torch.min(distances_positives, dim=0)
                        loss_seg += d2.mean()
                        total_distance += d2.mean()

                        # Loss for point type
                        labels = event_types_label[torch.argmin(distances_positives, dim=0)]
                        loss_type = torch.mean(self.cross_entropy(event_types[positives].double(), labels.long()))

                        # Accuracy for point type
                        predicted_types = torch.argmax(event_types[positives], dim=-1)
                        acc_type = (predicted_types == labels.long()).sum().item() / float(predicted_types.nelement())

                        total_acc_type += acc_type
                        total_loss_type += loss_type
                        total_loss += loss_type.float()

                    total_loss_ppn1 += loss_seg_ppn1
                    total_loss_ppn2 += loss_seg_ppn2
                    total_acc_ppn1 += acc_ppn1
                    total_acc_ppn2 += acc_ppn2
                    total_loss += (loss_seg + self._weight_ppn1*loss_seg_ppn1 + loss_seg_ppn2).float()
                    total_acc += acc
                    ppn_count += 1
                else:
                    print("No particles !")

        ppn_results = {
            'ppn_acc': total_acc,
            'ppn_loss': total_loss,
            'loss_class': total_class,
            'loss_distance': total_distance,
            'loss_ppn1': total_loss_ppn1,
            'loss_ppn2': total_loss_ppn2,
            'acc_ppn1': total_acc_ppn1,
            'acc_ppn2': total_acc_ppn2,
            'fraction_positives_ppn1': total_fraction_positives_ppn1,
            'fraction_positives_ppn2': total_fraction_positives_ppn2,
            'fraction_negatives_ppn1': total_fraction_negatives_ppn1,
            'fraction_negatives_ppn2': total_fraction_negatives_ppn2,
            'acc_ppn_type': total_acc_type,
            'loss_type': total_loss_type,
            'num_labels': num_labels,
            'num_discarded_labels_ppn1': num_discarded_labels_ppn1,
            'num_discarded_labels_ppn2': num_discarded_labels_ppn2
        }
        for key in ppn_results:
            if not isinstance(ppn_results[key], torch.Tensor):
                ppn_results[key] = torch.tensor(ppn_results[key])
        if ppn_count > 0:
            for key in ppn_results:
                ppn_results[key] = ppn_results[key] / float(ppn_count)
        return ppn_results
