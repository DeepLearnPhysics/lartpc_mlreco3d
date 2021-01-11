import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.normalizations import MinkowskiPixelNorm
from collections import defaultdict
from mlreco.mink.layers.factories import activations_construct, normalizations_construct


class EqualizedConvTranspose(nn.Module):
    '''
    Equalized Convolution Transpose from the ProGAN paper.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 has_bias=True,
                 generate_new_coords=False,
                 dimension=3):
        super(EqualizedConvTranspose, self).__init__()
        self.m = ME.MinkowskiConvolutionTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            has_bias=has_bias,
            generate_new_coords=generate_new_coords,
            dimension=dimension
        )
        nn.init.normal_(self.m.kernel)
        if has_bias:
            nn.init.zeros_(self.m.bias)
        with torch.no_grad():
            fan_in = torch.prod(self.m.kernel_size) * self.in_channels
            self.scale = torch.sqrt(2.0) / torch.sqrt(fan_in)
        
    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        y = ME.SparseTensor(
            x.F * self.scale,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)
        return self.m(y)


class EqualizedConv(nn.Module):
    '''
    Equalized Convolution from the ProGAN paper.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 has_bias=True,
                 dimension=3):
        super(EqualizedConv, self).__init__()
        self.m = ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            has_bias=has_bias,
            dimension=dimension
        )
        nn.init.normal_(self.m.kernel)
        if has_bias:
            nn.init.zeros_(self.m.bias)
        with torch.no_grad():
            fan_in = torch.prod(self.m.kernel_size) * self.in_channels
            self.scale = torch.sqrt(2.0) / torch.sqrt(fan_in)
        
    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        y = ME.SparseTensor(
            x.F * self.scale,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)
        return self.m(y)


class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, has_bias=True):
        super(EqualizedLinear, self).__init__()
        self.m = ME.MinkowskiLinear(in_features, out_features, has_bias=has_bias)
        nn.init.normal_(self.m.linear.weight)
        if has_bias:
            nn.init.zeros_(self.m.linear.bias)
        with torch.no_grad():
            fan_in = torch.prod(self.m.kernel_size) * self.in_channels
            self.scale = torch.sqrt(2.0) / torch.sqrt(fan_in)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        y = ME.SparseTensor(
            x.F * self.scale,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)
        return self.m(y)


class MiniBatchStdDev(nn.Module):
    '''
    TODO
    '''
    pass


class ProGANGenInitialBlock(nn.Module):
    '''
    Initial upsampling layer that transforms latent vector to 4x4x4 image.
    The layer design is taken from the ProGAN paper. 

    reference:
    https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/modules.py

    Args:
        - in_channels: number of input channels
        - out_channels: number of output channels
    '''
    def __init__(self, in_channels, out_channels, dimension=3, 
                 has_bias=True,
                 use_eql=False,
                 norm_name='pixel_norm',
                 norm_args={},
                 activation_name='lrelu', 
                 activation_args=dict(negative_slope=0.33)):
        super(GenInitialBlock, self).__init__()

        ConvBlock = EqualizedConv if use_eql else ME.MinkowskiConvolution
        ConvTBlock = EqualizedConvTranspose if use_eql else ME.MinkowskiConvolutionTranspose

        self.conv_1 = ConvTBlock(
            in_channels, out_channels, 
            kernel_size=4, stride=1, has_bias=has_bias,
            generate_new_coords=True, dimension=dimension)
        
        self.conv_2 = ConvBlock(
            out_channels, out_channels, 
            kernel_size=3, stride=1, has_bias=has_bias, dimension=dimension)

        self.norm_1 = normalizations_construct(norm_name, **norm_args)
        self.norm_2 = normalizations_construct(norm_name, **norm_args)

        self.act_1 = activations_construct(activation_name, **activation_args)
        self.act_2 = activations_construct(activation_name, **activation_args)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        y = self.norm_1(x)
        y = self.act1(self.conv_1(y))
        y = self.norm_2(y)
        y = self.act2(self.conv_2(y))
        return y


class ProGANGenBlock(nn.Module):
    '''
    Generator convolution block from the ProGAN paper.
    Performs one upsampling (scale=2) with generate_new_coords=True.

    reference:
    https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/modules.py

    '''
    def __init__(self, in_channels, out_channels, dimension=3, 
                 has_bias=True,
                 use_eql=False,
                 norm_name='pixel_norm',
                 norm_args={},
                 activation_name='lrelu', 
                 activation_args=dict(negative_slope=0.33)):
        super(ProGANGenBlock, self).__init__()

        ConvBlock = EqualizedConv if use_eql else ME.MinkowskiConvolution
        ConvTBlock = EqualizedConvTranspose if use_eql else ME.MinkowskiConvolutionTranspose

        self.upsampling = ConvTBlock(
            in_channels, out_channels, kernel_size=2, stride=2,
            dimension=dimension, has_bias=has_bias, generate_new_coords=True)

        self.conv_1 = ConvBlock(
            out_channels, out_channels, 
            kernel_size=3, stride=1, has_bias=has_bias, dimension=dimension)

        self.conv_2 = ConvBlock(
            out_channels, out_channels, 
            kernel_size=3, stride=1, has_bias=has_bias, dimension=dimension)

        self.norm_1 = normalizations_construct(norm_name, **norm_args)
        self.norm_2 = normalizations_construct(norm_name, **norm_args)
        self.norm_3 = normalizations_construct(norm_name, **norm_args)
        # Construct each separately, handling those activations that have a parameter (ex.PReLU)
        self.act_1 = activations_construct(activation_name, **activation_args)
        self.act_2 = activations_construct(activation_name, **activation_args)
        self.act_3 = activations_construct(activation_name, **activation_args)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        y = self.norm_1(self.act_1(self.upsampling(x)))
        y = self.norm_2(self.act_2(self.conv_1(y)))
        y = self.norm_3(self.act_3(self.conv_2(y)))
        return y
        

class ProGANDiscBlock(nn.Module):
    '''
    Discriminator convolution block from the ProGAN paper.
    Performs one downsampling (scale=2) via AveragePooling

    reference:
    https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/modules.py
    '''
    def __init__(self, in_channels, out_channels, dimension=3, 
                 has_bias=True,
                 use_eql=False,
                 norm_name='pixel_norm',
                 norm_args={},
                 activation_name='lrelu', 
                 activation_args=dict(negative_slope=0.33)):
        super(ProGANGenBlock, self).__init__()
        self.dowmsampling = ME.MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=dimension)

        ConvBlock = EqualizedConv if use_eql else ME.MinkowskiConvolution

        self.conv_1 = ConvBlock(
            in_channels, out_channels, 
            kernel_size=3, stride=1, has_bias=has_bias, dimension=dimension)

        self.conv_2 = ConvBlock(
            out_channels, out_channels, 
            kernel_size=3, stride=1, has_bias=has_bias, dimension=dimension)

        self.norm_1 = normalizations_construct(norm_name, **norm_args)
        self.norm_2 = normalizations_construct(norm_name, **norm_args)

        self.act_1 = activations_construct(activation_name, **activation_args)
        self.act_2 = activations_construct(activation_name, **activation_args)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        y = self.norm_1(self.act_1(self.conv_1(x)))
        y = self.norm_2(self.act_2(self.conv_2(y)))
        y = self.downsampling(y)
        return y


class ProGANDiscFinalBlock(nn.Module):
    '''
    ProGAN discriminator final layer block.

    reference:
    https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/modules.py
    '''
    def __init__(self, in_channels, out_channels, dimension=3,
                 has_bias=True,
                 use_eql=False,
                 norm_name='pixel_norm',
                 norm_args={},
                 activation_name='lrelu', 
                 activation_args=dict(negative_slope=0.33)):
        super(ProGANDiscFinalBlock, self).__init__()

        ConvBlock = EqualizedConv if use_eql else ME.MinkowskiConvolution
        self.conv_1 = ConvBlock(
            in_channels + 1, in_channels,
            kernel_size=3, stride=1, has_bias=has_bias, dimension=dimension)

        self.conv_2 = ConvBlock(
            in_channels, out_channels,
            kernel_size=4, stride=4, has_bias=has_bias, dimension=dimension)

        LinBlock = EqualizedLinear if use_eql else ME.MinkowskiLinear

        self.linear = LinBlock(out_channels, 1, has_bias=has_bias)
        # TODO: Batch Discriminator

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # y = self.batch_discriminator
        y = self.norm_1(self.act_1(self.conv_1(x)))
        y = self.norm_2(self.act_2(self.conv_2(y)))
        y = self.linear(y)
        return y
