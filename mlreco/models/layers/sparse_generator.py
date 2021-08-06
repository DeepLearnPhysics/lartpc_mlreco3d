mport torch
import torch.nn as nn

import MinkowskiEngine as ME

from mlreco.models.layers.activation_normalization_factories import activations_dict, activations_construct, normalizations_construct
from mlreco.models.layers.network_base import MENetworkBase
from mlreco.models.layers.blocks import ResNetBlock, ConvolutionBlock
from scipy.special import logit

# from torch_geometric.nn import BatchNorm, LayerNorm, MessageNorm

class SparseGenerator(MENetworkBase):

    def __init__(self, cfg, name='sparse_generator'):
        super(SparseGenerator, self).__init__(cfg)
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        # self.nPlanes = [(2**i) * self.num_filters for i in range(self.depth)]
        self.nPlanes = [16, 16, 32, 32, 64, 64, 128, 128, 256]
        print(self.nPlanes)
        assert len(self.nPlanes) == self.depth
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        print("Final Tensor Shape = ", final_tensor_shape)
        self.resolution = self.model_config.get('resolution', 1024)
        self.threshold = logit(self.model_config.get('threshold', 0.0))
        print(self.threshold)
        self.layer_limit = self.model_config.get('layer_limit', -1)
        if self.layer_limit < 0:
            self.layer_limit = len(self.nPlanes) + 1

        self.linear = nn.Sequential(
            normalizations_construct(self.norm, self.latent_size, **self.norm_args),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolutionTranspose(
                in_channels=self.latent_size,
                out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape,
                stride=final_tensor_shape,
                dimension=self.D,
                bias=self.allow_bias,
                generate_new_coords=True)
        )
        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        self.layer_cls = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(normalizations_construct(self.norm, self.nPlanes[i+1], **self.norm_args))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D,
                bias=self.allow_bias,
                generate_new_coords=True))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i],
                                     self.nPlanes[i],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args,
                                     normalization=self.norm,
                                     normalization_args=self.norm_args,
                                     has_bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            self.layer_cls.append(
                ME.MinkowskiLinear(self.nPlanes[i], 1, bias=self.allow_bias)
            )
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)
        self.layer_cls = nn.Sequential(*self.layer_cls)


        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
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


    def forward(self, latent, target_key):

        out_cls, targets = [], []

        latent.set_tensor_stride(self.resolution)

        x = self.linear(latent)
        layer_count = 0

        for i, layer in enumerate(self.decoding_conv):
            print(layer_count)
            if layer_count >= self.layer_limit:
                break
            x = layer(x)
            x = self.decoding_block[i](x)
            x_cls = self.layer_cls[i](x)
            target = self.get_target(x, target_key)
            targets.append(target)
            out_cls.append(x_cls)
            layer_count += 1
            keep = (x_cls.F > self.threshold).cpu().squeeze()
            if self.training:
                keep += target
            if keep.sum() > 0:
                x = self.pruning(x, keep.cpu())
            else:
                break

        return {
            'reconstruction': x,
            'out_cls': out_cls,
            'targets': targets}


class SparseGeneratorSimple(MENetworkBase):

    def __init__(self, cfg, name='sparse_generator'):
        super(SparseGeneratorSimple, self).__init__(cfg)
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        # self.nPlanes = [(2**i) * self.num_filters for i in range(self.depth)]
        self.nPlanes = [16, 16, 32, 32, 64, 64, 128, 128, 256]
        print(self.nPlanes)
        assert len(self.nPlanes) == self.depth
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        print("Final Tensor Shape = ", final_tensor_shape)
        self.resolution = self.model_config.get('resolution', 1024)
        self.threshold = logit(self.model_config.get('threshold', 0.0))
        print(self.threshold)
        self.layer_limit = self.model_config.get('layer_limit', -1)
        if self.layer_limit < 0:
            self.layer_limit = len(self.nPlanes) + 1

        self.linear = nn.Sequential(
            normalizations_construct(self.norm, self.latent_size, **self.norm_args),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolutionTranspose(
                in_channels=self.latent_size,
                out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape,
                stride=final_tensor_shape,
                dimension=self.D,
                bias=self.allow_bias,
                generate_new_coords=True)
        )
        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        self.layer_cls = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(normalizations_construct(self.norm, self.nPlanes[i+1], **self.norm_args))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D,
                bias=self.allow_bias,
                generate_new_coords=True))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ConvolutionBlock(self.nPlanes[i],
                                     self.nPlanes[i],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args,
                                     normalization=self.norm,
                                     normalization_args=self.norm_args,
                                     has_bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            self.layer_cls.append(
                ME.MinkowskiLinear(self.nPlanes[i], 1, bias=self.allow_bias)
            )
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)
        self.layer_cls = nn.Sequential(*self.layer_cls)


        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
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


    def forward(self, latent, target_key):

        out_cls, targets = [], []

        latent.set_tensor_stride(self.resolution)

        x = self.linear(latent)
        layer_count = 0

        for i, layer in enumerate(self.decoding_conv):
            print(layer_count)
            if layer_count >= self.layer_limit:
                break
            x = layer(x)
            x = self.decoding_block[i](x)
            x_cls = self.layer_cls[i](x)
            target = self.get_target(x, target_key)
            targets.append(target)
            out_cls.append(x_cls)
            layer_count += 1
            keep = (x_cls.F > self.threshold).cpu().squeeze()
            if self.training:
                keep += target
            if keep.sum() > 0:
                x = self.pruning(x, keep.cpu())
            else:
                break

        return {
            'reconstruction': x,
            'out_cls': out_cls,
            'targets': targets}
