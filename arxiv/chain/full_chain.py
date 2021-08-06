import numpy as np
import torch
import torch.nn as nn
import time

# MinkowskiEngine Backend
import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from mlreco.models.layers.blocks import ResNetBlock, CascadeDilationBlock, SPP, ASPP
from .factories import cnn_construct
from mlreco.models.layers.configuration import setup_cnn_configuration
from mlreco.models.layers.ppn import PPN

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
        print("Encoder # Parameters = ", 
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
        # print(self.encoder)

        # Segmentation Decoder
        self.segment_decoder_cfg = cfg['seg_decoder']
        # print("Segment Decoder Config")
        # pprint(self.segment_decoder_cfg)
        self.segment_decoder_name = self.segment_decoder_cfg['name']
        self.segment_decoder = cnn_construct(
            self.segment_decoder_name, self.segment_decoder_cfg)
        print("Segmentation Decoder # Parameters = ", 
            sum(p.numel() for p in self.segment_decoder.parameters() if p.requires_grad))
        # print(self.segment_decoder)

        # Instance Decoder
        self.instance_decoder_cfg = cfg['ins_decoder']
        # print("Instance Decoder Config")
        # pprint(self.instance_decoder_cfg)
        self.instance_decoder_name = self.instance_decoder_cfg['name']
        self.instance_decoder = cnn_construct(
            self.instance_decoder_name, self.instance_decoder_cfg)
        print("Instance Decoder # Parameters = ", 
            sum(p.numel() for p in self.instance_decoder.parameters() if p.requires_grad))

        # PPN
        self.ppn = PPN(cfg)
        print("PPN # Parameters = ", 
            sum(p.numel() for p in self.ppn.parameters() if p.requires_grad))
        # Seediness
        # print(cfg.keys())
        # OutputLayers 
        self.segmentation = ME.MinkowskiLinear(
            self.segment_decoder_cfg['uresnet_decoder']['num_filters'], 
            self.segment_decoder_cfg['uresnet_decoder']['num_classes'])
        self.embedding = ME.MinkowskiLinear(
            self.instance_decoder_cfg['uresnet_decoder']['num_filters'], 
            cfg['embedding_dim'] + cfg['sigma_dim'])

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):

        x = ME.SparseTensor(coords=input[:, :4], feats=input[:, -1].view(-1, 1))
        start = time.time()
        res_encoder = self.encoder(x)
        end = time.time()
        print("Encoder Time = {:.4f}".format(end - start))
        encoderTensors = res_encoder['encoderTensors']
        finalTensor = res_encoder['finalTensor']

        start = time.time()
        seg_decoderTensors = self.segment_decoder(finalTensor, encoderTensors)
        end = time.time()
        print("Segment Decoder Time = {:.4f}".format(end - start))
        start = time.time()
        ins_decoderTensors = self.instance_decoder(finalTensor, encoderTensors)
        end = time.time()
        print("Instance Decoder Time = {:.4f}".format(end - start))

        features_ppn = res_encoder['features_ppn']
        features_ppn2 = [finalTensor] + seg_decoderTensors

        ppn_input = {
            'ppn_feature_enc': features_ppn,
            'ppn_feature_dec': features_ppn2
        }

        # start = time.time()
        # ppn_output = self.ppn(ppn_input)
        # end = time.time()
        # print("PPN Time = {:.4f}".format(end - start))

        seg_features = seg_decoderTensors[-1]
        ins_features = ins_decoderTensors[-1]

        embeddings = self.embedding(ins_features)
        segmentation = self.segmentation(seg_features)

        res = {
            'segmentation': [segmentation.F],
            'embeddings': [self.tanh(embeddings.F[:, :3])],
            'margins': [torch.exp(embeddings.F[:, 3:])]
            # 'seediness': [None]
        }

        # for key, val in ppn_output.items():
        #     res[key] = val

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
