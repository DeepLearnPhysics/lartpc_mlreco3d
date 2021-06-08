import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F
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
        self.num_samples = self.model_config.get('num_samples', 20)

        self.encoder = BayesianEncoder(cfg)
        self.decoder = BayesianDecoder(cfg)

        self.mode = self.model_config.get('mode', 'standard')

        self.classifier = ME.MinkowskiLinear(self.encoder.num_filters, self.num_classes)


    def mc_forward(self, input, num_samples=None):

        res = defaultdict(list)

        if num_samples is None:
            num_samples = self.num_samples

        for m in self.modules():
            if m.__class__.__name__ == 'Dropout':
                m.train()

        for igpu, x in enumerate(input):

            num_voxels = x.shape[0]

            device = x.device

            x_sparse = ME.SparseTensor(coordinates=x[:, :4],
                                features=x[:, -1].view(-1, 1))

            pvec = torch.zeros((num_voxels, self.num_classes)).to(device)
            logits = torch.zeros((num_voxels, self.num_classes)).to(device)

            for i in range(num_samples):
                res_encoder = self.encoder.encoder(x_sparse)
                decoderTensors = self.decoder(
                    res_encoder['finalTensor'], res_encoder['encoderTensors'])
                feats = decoderTensors[-1]
                out = self.classifier(feats)
                logits += out.F
                pvec += F.softmax(out.F, dim=1)

            logits /= num_samples
            softmax_probs = pvec / num_samples

            res['softmax'].append(softmax_probs)
            res['segmentation'].append(logits)

        return res
        


    def standard_forward(self, input):

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


    def forward(self, input):

        if self.mode == 'mc_dropout':
            return self.mc_forward(input)
        else:
            return self.standard_forward(input)