# Dummy model that only passes the input to the output, used for testing purposes.

import torch
import torch.nn as nn

class FileIOPlaceHolder(nn.Module):
    def __init__(self, cfg, name='file_io'):
        super(FileIOPlaceHolder, self).__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        out = {
            'x': x
        }
        return out
    
class FileIOPlaceHolderLoss(nn.Module):
    
    def __init__(self, cfg):
        super(FileIOPlaceHolderLoss, self).__init__()
        self.cfg = cfg
        
    def forward(self, result, labels):
        
        output = {
            'accuracy': 0.0,
            'loss': 0.0
        }
        
        return output