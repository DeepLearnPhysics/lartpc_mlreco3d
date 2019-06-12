from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.layers.extract_feature_map import Selection, Multiply, AddLabels


class PPN(torch.nn.Module):
    def __init__(self, cfg):
        super(PPN, self).__init__()
        import sparseconvnet as scn
        model_config = cfg['modules']['ppn']
        self._model_config = model_config
        dimension = model_config['data_dim']
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = model_config['filters']  # Unet number of features
        nPlanes = [i*m for i in range(1, model_config['num_strides']+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]# downsample = [filter size, filter stride]

        # PPN stuff
        self.half_stride = int(model_config['num_strides']/2.0)
        self.ppn1_conv = scn.SubmanifoldConvolution(dimension, nPlanes[-1], nPlanes[-1], 3, False)
        self.ppn1_scores = scn.SubmanifoldConvolution(dimension, nPlanes[-1], 2, 3, False)

        self.selection1 = Selection()
        self.selection2 = Selection()
        self.unpool1 = scn.Sequential()
        for i in range(model_config['num_strides']-self.half_stride-1):
            self.unpool1.add(scn.UnPooling(dimension, downsample[0], downsample[1]))

        self.unpool2 = scn.Sequential()
        for i in range(self.half_stride):
            self.unpool2.add(scn.UnPooling(dimension, downsample[0], downsample[1]))

        middle_filters = int(m * self.half_stride * (self.half_stride + 1) / 2.0)
        self.ppn2_conv = scn.SubmanifoldConvolution(dimension, middle_filters, middle_filters, 3, False)
        self.ppn2_scores = scn.SubmanifoldConvolution(dimension, middle_filters, 2, 3, False)
        self.multiply1 = Multiply()
        self.multiply2 = Multiply()

        self.ppn3_conv = scn.SubmanifoldConvolution(dimension, nPlanes[0], nPlanes[0], 3, False)
        self.ppn3_pixel_pred = scn.SubmanifoldConvolution(dimension, nPlanes[0], dimension, 3, False)
        self.ppn3_scores = scn.SubmanifoldConvolution(dimension, nPlanes[0], 2, 3, False)
        self.ppn3_type = scn.SubmanifoldConvolution(dimension, nPlanes[0], model_config['num_classes'], 3, False)

        self.add_labels1 = AddLabels()
        self.add_labels2 = AddLabels()

    def forward(self, input):
        use_encoding = False
        label, x, feature_ppn, feature_ppn2 = input
        label = label[:, :-1]

        if use_encoding:
            y = self.ppn1_conv(feature_ppn[-1])
        else:
            y = self.ppn1_conv(feature_ppn2[0])

        ppn1_scores = self.ppn1_scores(y)
        mask = self.selection1(ppn1_scores)
        attention = self.unpool1(mask)
        if self.training:
            with torch.no_grad():
                attention = self.add_labels1(attention, torch.cat([label[:, :-1]/2**self.half_stride, label[:, -1][:, None]], dim=1).long())
                # for b in range(self._flags.BATCH_SIZE):
                #     batch_index = attention.get_spatial_locations()[:, -1] == b
                #     print(attention.features.shape, batch_index.shape)
                #     if attention.features[batch_index].sum() == 0:
                #         print(label[label[:, -1] == b])
                #         print((label/2**self.half_stride).long()[label[:, -1] == b])
                #         print(attention.features[batch_index])
                #         print(attention.get_spatial_locations()[batch_index])
                # print(attention.features[attention.get_spatial_locations()[:, -1] == 11].size())
                # print(attention.features[attention.get_spatial_locations()[:, -1] == 11][attention.features[attention.get_spatial_locations()[:, -1] == 11]>0])
        if use_encoding:
            y = feature_ppn[self.half_stride]
        else:
            y = feature_ppn2[self.half_stride]
        y = self.multiply1(y, attention)
        y = self.ppn2_conv(y)
        ppn2_scores = self.ppn2_scores(y)
        mask2 = self.selection2(ppn2_scores)
        attention2 = self.unpool2(mask2)
        if self.training:
            with torch.no_grad():
                attention2 = self.add_labels2(attention2, label.long())
        if use_encoding:
            z = feature_ppn[0]
        else:
            z = feature_ppn2[-1]

        z = self.multiply2(z, attention2)
        z = self.ppn3_conv(z)
        ppn3_pixel_pred = self.ppn3_pixel_pred(z)
        ppn3_scores = self.ppn3_scores(z)
        # FIXME wrt batch index
        pixel_pred = ppn3_pixel_pred.features
        scores = ppn3_scores.features
        return [[torch.cat([pixel_pred, scores], dim=1)],
                [torch.cat([ppn1_scores.get_spatial_locations().cuda().float(), ppn1_scores.features], dim=1)],
                [torch.cat([ppn2_scores.get_spatial_locations().cuda().float(), ppn2_scores.features], dim=1)],
                [attention.features],
                [attention2.features]]


class PPNLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg, reduction='sum'):
        super(PPNLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['modules']['ppn']
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, segmentation, label, particles):
        """
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has only 1 element because UResNet returns only 1 element.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        weight can be None.
        """
        assert len(segmentation[0]) == len(particles)
        assert len(segmentation[0]) == len(label)
        batch_ids = [d[:, -2] for d in label]
        total_loss = 0.
        total_acc = 0.
        ppn_count = 0.
        total_distance, total_class = 0., 0.
        total_loss_ppn1, total_loss_ppn2 = 0., 0.
        total_acc_ppn1, total_acc_ppn2 = 0., 0.
        data_dim = self._cfg['data_dim']
        for i in range(len(label)):
            event_particles = particles[i]
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_data = label[i][batch_index][:, :-2]  # (N, 3)
                ppn1_batch_index = segmentation[1][i][:, -3] == b.float()
                ppn2_batch_index = segmentation[2][i][:, -3] == b.float()
                event_ppn1_data = segmentation[1][i][ppn1_batch_index][:, :-3]  # (N1, 3)
                event_ppn2_data = segmentation[2][i][ppn2_batch_index][:, :-3]  # (N2, 3)
                anchors = (event_data + 0.5).float()

                event_pixel_pred = segmentation[0][i][batch_index][:, :data_dim] + anchors # (N, 3)
                event_scores = segmentation[0][i][batch_index][:, data_dim:(data_dim+2)]  # (N, 2)
                # event_types = segmentation[0][i][batch_index][:, (data_dim+2):]  # (N, num_classes)
                event_ppn1_scores = segmentation[1][i][ppn1_batch_index][:, -2:]  # (N1, 2)
                event_ppn2_scores = segmentation[2][i][ppn2_batch_index][:, -2:]  # (N2, 2)

                # PPN stuff
                event_label = event_particles[event_particles[:, -1] == b][:, :-2]  # (N_gt, 3)
                # event_types_label = event_particles[event_particles[:, -1] == b][:, data_dim+1]
                # print(b, event_label.size())
                if event_label.size(0) > 0:
                    ppn_count += 1
                    # Segmentation loss (predict positives)
                    d = self.distances(event_label, event_pixel_pred)
                    d_true = self.distances(event_label, event_data)
                    positives = (d_true < 5).any(dim=0)  # FIXME can be empty
                    if positives.shape[0] == 0:
                        continue
                    loss_seg = torch.mean(self.cross_entropy(event_scores.double(), positives.long()))
                    total_class += loss_seg

                    # Accuracy for scores
                    predicted_labels = torch.argmax(event_scores, dim=-1)
                    acc = (predicted_labels == positives.long()).sum().item() / float(predicted_labels.nelement())

                    # Loss ppn1 & ppn2 (predict positives)
                    d_true_ppn1 = self.distances(event_label/(2**(self._cfg['num_strides']-1)), event_ppn1_data)
                    d_true_ppn2 = self.distances(event_label/(2**(int(self._cfg['num_strides']/2))), event_ppn2_data)
                    positives_ppn1 = (d_true_ppn1 < 1).any(dim=0)
                    positives_ppn2 = (d_true_ppn2 < 1).any(dim=0)
                    loss_seg_ppn1 = torch.mean(self.cross_entropy(event_ppn1_scores.double(), positives_ppn1.long()))
                    loss_seg_ppn2 = torch.mean(self.cross_entropy(event_ppn2_scores.double(), positives_ppn2.long()))
                    predicted_labels_ppn1 = torch.argmax(event_ppn1_scores, dim=-1)
                    predicted_labels_ppn2 = torch.argmax(event_ppn2_scores, dim=-1)
                    acc_ppn1 = (predicted_labels_ppn1 == positives_ppn1.long()).sum().item() / float(predicted_labels_ppn1.nelement())
                    acc_ppn2 = (predicted_labels_ppn2 == positives_ppn2.long()).sum().item() / float(predicted_labels_ppn2.nelement())

                    # Mask: only consider pixels that were selected
                    event_mask = segmentation[4][i][batch_index]
                    event_mask = (~(event_mask == 0)).any(dim=1)  # (N,)
                    # event_label = event_label[event_mask]
                    # event_segmentation = event_segmentation[event_mask]
                    event_pixel_pred = event_pixel_pred[event_mask]
                    event_scores = event_scores[event_mask]
                    # event_types = event_types[event_mask]
                    event_data = event_data[event_mask]
                    # Mask for PPN2
                    # event_ppn2_mask = (~(segmentation[4][i][ppn2_batch_index] == 0)).any(dim=1)
                    # event_ppn2_data = event_ppn2_data[event_ppn2_mask]
                    # event_ppn2_scores = event_ppn2_scores[event_ppn2_mask]

                    # Distance loss
                    positives = (d_true[:, event_mask] < 5).any(dim=0)
                    distances_positives = d[:, event_mask][:, positives]
                    if distances_positives.shape[1] > 0:
                        d2, _ = torch.min(distances_positives, dim=0)
                        loss_seg += d2.mean()
                        total_distance += d2.mean()

                    total_loss_ppn1 += loss_seg_ppn1
                    total_loss_ppn2 += loss_seg_ppn2
                    total_acc_ppn1 += acc_ppn1
                    total_acc_ppn2 += acc_ppn2
                    total_loss += (loss_seg + loss_seg_ppn1 + loss_seg_ppn2).float()
                    total_acc += acc
                # else:
                #     print("No particles !")

        ppn_results = {
            'ppn_acc': total_acc,
            'ppn_loss': total_loss,
            'loss_class': total_class,
            'loss_distance': total_distance,
            'loss_ppn1': total_loss_ppn1,
            'loss_ppn2': total_loss_ppn2,
            'acc_ppn1': total_acc_ppn1,
            'acc_ppn2': total_acc_ppn2
        }
        ppn_results_keys = list(ppn_results.keys())
        for key in ppn_results_keys:
            ppn_results[key + '_count'] = ppn_count
        return ppn_results
