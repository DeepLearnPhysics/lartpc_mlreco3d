from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch.utils.data import Sampler

class RandomSequenceSampler(Sampler):

    def __init__(self,data_size,batch_size):
        self._data_size  = int(data_size)
        if self._data_size < 0:
            print(self.__class__.__name__,'received negative data size',data_size)
            raise ValueError
        
        self._batch_size = int(batch_size)
        if self._batch_size < 0 or self._batch_size > self._data_size:
            print(self.__class__.__name__,'received invalid batch size',batch_size,'for data size',self._data_size)
            raise ValueError
        
    def __len__(self):
        return int(self._data_size/self._batch_size)

    def __iter__(self):
        starts = torch.randint(high=self._data_size - self._batch_size,
                               size=(len(self),))
        return iter(np.concatenate([np.arange(start,start+self._batch_size) for start in starts]))

    @staticmethod
    def create(ds,cfg):
        return RandomSequenceSampler(len(ds),cfg['batch_size'])
