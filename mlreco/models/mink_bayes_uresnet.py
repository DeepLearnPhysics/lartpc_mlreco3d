import torch
import torch.nn as nn
import MinkowskiEngine as ME

import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.mink.layers.factories import (activations_dict, 
                                          activations_construct, 
                                          normalizations_construct)
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.bayes.encoder import BayesianEncoder
from mlreco.bayes.decoder import BayesianDecoder


class BayesianUResNet(MENetworkBase):

    def __init__(self, cfg, name='bayesian_uresnet'):
        super(BayesianUResNet, self).__init__(cfg)
        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)

        self.encoder = BayesianEncoder(cfg)
        self.decoder = BayesianDecoder(cfg)

        self.classifier = ME.MinkowskiLinear(self.encoder.num_filters, self.num_classes)

    def forward(self, input):

        out = defaultdict(list)
        for igpu, x in enumerate(input):
            x = ME.SparseTensor(coordinates=x[:, :4],
                                features=x[:, -1].view(-1, 1))
            res_encoder = self.encoder.encoder(x)
            print([t.F.shape for t in res_encoder['encoderTensors']])
            decoderTensors = self.decoder(res_encoder['finalTensor'], res_encoder['encoderTensors'])
            feats = decoderTensors[-1]
            logits = self.classifier(feats)
            out['segmentation'].append(logits.F)
        return out
