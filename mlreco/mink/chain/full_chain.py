import numpy as np
import torch
import torch.nn as nn

# MinkowskiEngine Backend
import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from mlreco.mink.layers.blocks import ResNetBlock, CascadeDilationBlock, SPP, ASPP
from .factories import cnn_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.ppn import PPN

from pprint import pprint


class FullChainCNN1(MENetworkBase):
    '''
    Architecture:
      1) Shared Encoder
      2) Segmentation Decoder
      3) Instance Decoder
      4) CNN Node Encoder
      5) Geo/CNN Edge Encoder
    '''
    def __init__(self, cfg, name='full_chain_cnn'):
        super(FullChainCNN1, self).__init__(cfg)
        # print(cfg)

        # Encoder
        self.encoder_cfg = cfg['encoder']
        # print("Encoder Config")
        # pprint(self.encoder_cfg)
        self.encoder_name = self.encoder_cfg['name']
        # print(self.encoder_name)
        self.encoder = cnn_construct(self.encoder_name, self.encoder_cfg)
        # print(self.encoder)

        # Segmentation Decoder
        self.segment_decoder_cfg = cfg['seg_decoder']
        # print("Segment Decoder Config")
        # pprint(self.segment_decoder_cfg)
        self.segment_decoder_name = self.segment_decoder_cfg['name']
        self.segment_decoder = cnn_construct(
            self.segment_decoder_name, self.segment_decoder_cfg)
        # print(self.segment_decoder)

        # Instance Decoder
        self.instance_decoder_cfg = cfg['ins_decoder']
        # print("Instance Decoder Config")
        # pprint(self.instance_decoder_cfg)
        self.instance_decoder_name = self.instance_decoder_cfg['name']
        self.instance_decoder = cnn_construct(
            self.instance_decoder_name, self.instance_decoder_cfg)
        # PPN
        self.ppn = PPN(cfg)
        # Seediness


    def forward(self, input):

        x = ME.SparseTensor(coords=input[:, :4], feats=input[:, -1].view(-1, 1))
        res_encoder = self.encoder(x)
        encoderTensors = res_encoder['encoderTensors']
        finalTensor = res_encoder['finalTensor']

        seg_decoderTensors = self.segment_decoder(finalTensor, encoderTensors)
        ins_decoderTensors = self.instance_decoder(finalTensor, encoderTensors)

        features_ppn = res_encoder['features_ppn']
        features_ppn2 = [finalTensor] + seg_decoderTensors

        ppn_input = {
            'ppn_feature_enc': features_ppn,
            'ppn_feature_dec': features_ppn2
        }

        ppn_output = self.ppn(ppn_input)

        seg_features = seg_decoderTensors[-1]
        ins_features = ins_decoderTensors[-1]

        res = {
            'seg_features': [seg_features],
            'seg_decoderTensors': [seg_decoderTensors],
            'ins_decoderTensors': [ins_decoderTensors],
            'finalTensor': [finalTensor],
            'ins_features': [ins_features],
            'ppn_output': {},
            'seediness': []
        }

        return res


class FullChainCNN2(MENetworkBase):
    '''
    Architecture:
      1) Shared Encoder
      2) Segmentation Decoder
      3) Instance Decoder
      4) CNN Node Encoder
      5) MLP Edge Feature Constructor (Not CNN Encoder)
    '''
    def __init__(self, cfg, name='full_chain_cnn'):
        super(FullChainCNN2, self).__init__(cfg)
        self.model_config = cfg[name]

        # Encoder
        self.encoder_cfg = self.model_config['encoder']
        self.encoder_name = self.encoder_cfg['name']
        self.encoder = cnn_construct(self.encoder_name, cfg)

        # Segmentation Decoder
        self.segment_decoder_cfg = self.model_config['seg_decoder']
        self.segment_decoder_name = self.segment_decoder_cfg['name']
        self.segment_decoder = cnn_construct(self.segment_decoder_name, cfg)

        # Instance Decoder
        self.instance_decoder_cfg = self.model_config['ins_decoder']
        self.instance_decoder_name = self.instance_decoder_cfg['name']
        self.instance_decoder = cnn_construct(self.instance_decoder_name, cfg)

        # Node Encoder
        self.node_encoder_cfg = self.model_config['node_encoder']
        self.node_encoder_name = self.node_encoder_cfg['name']
        self.node_encoder = node_encoder_construct(self.node_encoder_name, cfg)

        # PPN

        # Seediness


    def forward(self, input):

        x = ME.SparseTensor(coords=input[:, :4], feats=input[:, -1].view(-1, 1))
        res_encoder = self.encoder(x)
        encoderTensors = res_encoder['encoderTensors']
        finalTensor = res_encoder['finalTensor']

        seg_decoderTensors = self.segment_decoder(finalTensor, encoderTensors)
        ins_decoderTensors = self.instance_decoder(finalTensor, encoderTensors)

        segmentation = seg_decoderTensors[-1]
        embeddings = ins_decoderTensors[-1]

        res = {
            'segmentation': [segmentation],
            'embeddings': [embeddings]
        }

        return res
