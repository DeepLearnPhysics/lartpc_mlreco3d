import torch
import numpy as np


class SPICE(nn.Module):
    '''
    Driver class for cnn based pixel clustering. 
    '''
    def __init__(self, cfg, name='spice'):
        super(SPICE, self).__init__()