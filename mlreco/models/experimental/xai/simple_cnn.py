import torch
import torch.nn as nn
import MinkowskiEngine as ME

class ImageClassificationWrapper(nn.Module):

    def __init__(self, model):
        super(ImageClassificationWrapper, self).__init__()
        self.model = model

    def forward(self, x, output_name='logits'):

        res = self.model.forward(x)
        return res[output_name][0]
