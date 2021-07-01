import torch
import torch.nn as nn
import MinkowskiEngine as ME
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.factories import activations_construct, normalizations_construct
import math

from MinkowskiEngine.utils.init import _calculate_fan_in_and_fan_out, _calculate_correct_fan

def selu_normal_(tensor, a=0, mode='fan_in'):
    fan = _calculate_correct_fan(tensor, mode)
    std = 1.0 / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


class VGG16(MENetworkBase):
    
    def __init__(self, cfg, name='simplenet'):
        super(VGG16, self).__init__(cfg)

        self.features = []
        self.classifier = []

        self.features.append(
            ME.MinkowskiConvolution(1, 16, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(16, 16, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(16, 32, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(32, 32, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(32, 64, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(64, 64, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(64, 128, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(128, 128, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(128, 256, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(256, 256, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(256, 256, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))
        
        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(256, 512, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))
        
        self.features.append(
            ME.MinkowskiConvolution(512, 512, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))
        
        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(512, 512, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(512, 512, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))
        
        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(512, 1024, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(1024, 1024, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))
        
        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(1024, 1024, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))

        self.features.append(
            ME.MinkowskiConvolution(1024, 1024, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))
        
        self.features.append(ME.MinkowskiSELU())

        self.features.append(
            ME.MinkowskiConvolution(1024, 1024, kernel_size=3, 
                                    dimension=self.D, 
                                    bias=self.allow_bias))

        self.features.append(ME.MinkowskiSELU())
        self.features.append(ME.MinkowskiMaxPooling(2, 2, dimension=self.D))
        
        self.pool = ME.MinkowskiGlobalMaxPooling()

        self.features = nn.Sequential(*self.features)

        self.classifier.extend(
            [
                nn.Linear(1024, 512),
                nn.SELU(),
                nn.AlphaDropout(0.5),
                nn.Linear(512, 5)
            ]
        )
        self.classifier = nn.Sequential(*self.classifier)

        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                selu_normal_(m.kernel, mode="fan_in")

            if isinstance(m, nn.Linear):
                selu_normal_(m.weight, mode="fan_in")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        
    def forward(self, input_tensor):

        x = ME.SparseTensor(coordinates=input_tensor[:, :4],
                    features=input_tensor[:, -1].view(-1, 1))
        
        out = self.features(x)
        print(out.tensor_stride)
        flattened = self.pool(out)
        logits = self.classifier(flattened.F)
        return logits