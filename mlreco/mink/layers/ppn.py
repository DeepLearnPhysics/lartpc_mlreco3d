import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.ppn import define_ppn12


class AddLabels(nn.Module):
    def __init__(self):
        super(AddLabels, self).__init__()

    def forward(self, attention, label):
        features = attention.F.new().resize_(1).expand_as(attention.F).fill_(1.0)
        features = features * attention.F
        coords = attention.C
        output = ME.SparseTensor(
            coords=coords, feats=features,
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

class Selection(nn.Module):
    '''
    Inputs:
        - logits (ME.SparseTensor): SparseTensor with N x 2 PPN
            score feature map.

    Returns:
        - out (ME.SparseTensor): SparseTensor where coordinates with score
        less than the threshold is pruned.
    '''
    def __init__(self, threshold=0.8):
        super(Selection, self).__init__()
        self.threshold = threshold
        self.softmax = nn.Softmax(dim=1)
        self.prune = ME.MinkowskiPruning()

    def forward(self, logits):
        with torch.no_grad():
            mask = (self.softmax(logits.F)[:, 1] > self.threshold).cpu()
            out = self.prune(logits, mask)
            return out


def get_target(out, target_key, kernel_size=1):
    with torch.no_grad():
        target = torch.zeros(len(out), dtype=torch.bool)
        cm = out.coords_man
        strided_target_key = cm.stride(
            target_key, out.tensor_stride[0], force_creation=True)
        ins, outs = cm.get_kernel_map(
            out.coords_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1)
        for curr_in in ins:
            target[curr_in] = 1
    return target


class PPN(nn.Module):
    '''
    MinkowskiEngine implemnetation of PPN.
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
                    has_bias=False,
                    dimension=self._dimension))
        self.ppn1_conv = nn.Sequential(*self.ppn1_conv)
        self.ppn1_scores = ME.MinkowskiConvolution(
            nPlanes[self.ppn1_stride-self._num_strides], 1,
            kernel_size=3,
            has_bias=False,
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
                    dimension=self._dimension))
        self.unpool1 = nn.Sequential(*self.unpool1)

        middle_filters = (self.ppn2_stride+1)*m
        self.ppn2_conv = []
        for i in range(self._ppn_num_conv):
            self.ppn2_conv.append(ME.MinkowskiConvolution(
                middle_filters, middle_filters,
                kernel_size=3, has_bias=False, dimension=self._dimension))
        self.ppn2_conv = nn.Sequential(*self.ppn2_conv)
        self.ppn2_scores = ME.MinkowskiConvolution(
            middle_filters, 1,
            kernel_size=3,
            has_bias=False,
            dimension=self._dimension)

        self.unpool2 = []
        self.unpool_norm2 = nn.ModuleList()
        for i in range(self.ppn2_stride):
            self.unpool2.append(
                ME.MinkowskiConvolutionTranspose(
                    nPlanes[self.ppn2_stride-self._num_strides+i],
                    (nPlanes[self.ppn2_stride-self._num_strides+i+1] \
                        if i != self.ppn2_stride-1 else 1),
                    kernel_size=downsample[0], stride=downsample[1],
                    dimension=self._dimension))
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
                ME.MinkowskiConvolution(nPlanes[0], nPlanes[1],
                    kernel_size=3, has_bias=False, dimension=self._dimension))
        self.ppn3_conv = nn.Sequential(*self.ppn3_conv)
        self.ppn3_pixel_pred = ME.MinkowskiConvolution(nPlanes[0], self._dimension,
            kernel_size=3, has_bias=False, dimension=self._dimension)
        self.ppn3_scores = ME.MinkowskiConvolution(nPlanes[0], 2,
            kernel_size=3, has_bias=False, dimension=self._dimension)
        self.ppn3_type = ME.MinkowskiConvolution(nPlanes[0], self._num_classes,
            kernel_size=3, has_bias=False, dimension=self._dimension)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.selection1 = Selection()
        self.selection2 = Selection()

        self.prune = ME.MinkowskiPruning()

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
        x = self.ppn1_conv(feature_map1)
        ppn1_scores = self.ppn1_scores(x)
        mask = (self.sigmoid(ppn1_scores.F) > 0.8).cpu()
        x = self.prune(x, mask)
        for i, layer in enumerate(self.unpool1):
            x = self.unpool_norm1[i](x)
            x = layer(x)
        # print(x, x.C.shape)

        if self.training and label is not None:
            with torch.no_grad():
                pass
                # attention = self.add_labels1(attention, \
                # torch.cat([label[:, :-2]/2**self.ppn2_stride, label[:, -2][:, None]], dim=1).long())

        # Feature map 2 = intermediate
        # target_key = x.coords_man.create_coords_key(
        #     feature_map2.C,
        #     force_creation=True,
        #     allow_duplicate_coords=True)
        # target = get_target(attention, target_key)
        print(feature_map2, feature_map2.C.shape)
        mask = (self.sigmoid(x.F) > 0.8).cpu()
        y = self.prune(x, mask)
        y = self.ppn2_conv(y)
        ppn2_scores = self.ppn2_scores(y)
        mask = (self.sigmoid(ppn2_scores.F) > 0.8).cpu()
        y = self.prune(feature_map2, mask)
        y = self.selection2(ppn2_scores)
        attention2 = self.unpool2(y)
        if self.training and label is not None:
            with torch.no_grad():
                attention2 = self.add_labels2(attention2, label[:,:-1].long())

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
