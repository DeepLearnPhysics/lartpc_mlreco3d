import numpy as np
import torch
from dataclasses import dataclass

from .globals import *
from .volumes import VolumeBoundaries


class Unwrapper:
    '''
    Break down the input and output dictionaries into individual events.

    Need to account for: multi-gpu, minibatching, multiple outputs, batches.
    '''
    def __init__(self, num_gpus, batch_size, rules={}, boundaries=None, remove_batch_col=False):
        '''
        Translate rule arrays and boundaries into instructions.

        Parameters
        ----------
        batch_size : int
             Number of events in the batch
        rules : dict
             Dictionary which contains a set of unwrapping rules for each
             output key of the reconstruction chain. If there is no rule
             associated with a key, the list is concatenated.
        boundaries : list
             List of detector volume boundaries
        remove_batch_col : bool
             Remove column which specifies batch ID from the unwrapped tensors
        '''
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.remove_batch_col = remove_batch_col
        self.merger = VolumeBoundaries(boundaries) if boundaries else None
        self.rules = self._parse_rules(rules)
        self.masks, self.offsets = {}, {}

    def __call__(self, data_blob, result_blob):
        '''
        Main unwrapping function. Loops over the data and result keys
        and applies the unwrapping rules. Returns the unwrapped versions
        of the two dictionaries

        Parameters
        ----------
        data_blob : dict
            Dictionary of array of array of minibatch data [key][num_minibatch][num_device]
        result_blob : dict
            Results dictionary, output of trainval.forward [key][num_minibatch*num_device]
        '''
        self._build_batch_masks(data_blob, result_blob)
        data_unwrapped, result_unwrapped = {}, {}
        for k, v in data_blob.items():
            data_unwrapped[k] = self._unwrap(k, v)
        for k, v in result_blob.items():
            result_unwrapped[k] = self._unwrap(k, v)

        return data_unwrapped, result_unwrapped

    @dataclass
    class Rule:
        method  : str = None
        ref_key : str = None

    def _parse_rules(self, rules):
        '''
        Translate rule arrays into Rule objects. Do the
        necessary checks to ensure rule sanity.

        Parameters
        ----------
        rules : dict
             Dictionary which contains a set of unwrapping rules for each
             output key of the reconstruction chain. If there is no rule
             associated with a key, the list is concatenated.
        '''
        parsed_rules = {}
        for key, rule in rules.items():
            parsed_rules[key] = self.Rule(*rule)
            if not parsed_rules[key].ref_key:
                parsed_rules[key].ref_key = key

            assert parsed_rules[key].method in ['done', 'scalar', 'tensor', 'tensor_list', 'edge_tensor']

        return parsed_rules


    def _build_batch_masks(self, data_blob, result_blob):
        '''
        For all the returned data objects that require a batch mask:
        build it and store it.

        Parameters
        ----------
        data_blob : dict
            Dictionary of array of array of minibatch data [key][num_minibatch][num_device]
        result_blob : dict
            Results dictionary, output of trainval.forward [key][num_minibatch*num_device]
        '''
        self.masks, self.offsets = {}, {}
        for key, value in data_blob.items():
            if isinstance(value[0], np.ndarray):
                self.masks[key] = [self._batch_masks(value[g]) for g in range(self.num_gpus)]
                if key not in self.rules:
                    self.rules[key] = self.Rule('tensor', key)
        for key in result_blob.keys():
            if key in self.rules and self.rules[key].method in ['tensor', 'tensor_list']:
                ref_key = self.rules[key].ref_key
                assert ref_key in self.masks or ref_key in result_blob, 'Must provide the reference tensor to unwrap'
                assert self.rules[key].method == self.rules[ref_key].method, 'Reference must be of same type'
                if ref_key not in self.masks:
                    if self.rules[key].method == 'tensor':
                        self.masks[ref_key] = [self._batch_masks(result_blob[ref_key][g]) for g in range(self.num_gpus)]
                    elif self.rules[key].method == 'tensor_list':
                        self.masks[ref_key] = [[self._batch_masks(v) for v in result_blob[ref_key][g]] for g in range(self.num_gpus)]
            elif key in self.rules and self.rules[key].method == 'edge_tensor':
                assert len(self.rules[key].ref_key) == 2, 'Must provide a reference to the edge_index and the node batch ids'
                for ref_key in self.rules[key].ref_key:
                    assert ref_key in result_blob, 'Must provide reference tensor to unwrap'
                ref_edge, ref_node = self.rules[key].ref_key
                if ref_edge not in self.masks:
                    edge_index, batch_ids = result_blob[ref_edge], result_blob[ref_node]
                    self.masks[ref_edge] = [self._batch_masks(batch_ids[g][edge_index[g][:,0]]) for g in range(self.num_gpus)]
                    self.offsets[ref_edge] = [np.cumsum([np.sum(batch_ids[g][:,BATCH_COL] == b-1) for b in range(self.batch_size)]) for g in range(self.num_gpus)]

    def _batch_masks(self, tensor):
        '''
        Makes a list of masks for each batch for a specific tensor.

        Parameters
        ----------
        tensor : np.ndarray
             Tensor with a batch ID column
        '''
        # Identify how many volumes we are dealing with
        num_volumes = self.merger.num_volumes() if self.merger else 1

        # Create batch masks
        batch_masks = []
        for b in range(self.batch_size):
            batch_mask = []
            for v in range(num_volumes):
                batch_mask.extend(np.where(tensor[:, BATCH_COL] == b*num_volumes+v)[0])
            batch_masks.append(batch_mask)

        return batch_masks

    def _unwrap(self, key, data):
        '''
        Routes set of data to the appropriate unwrapping scheme

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : list
            Data product
        '''
        if key not in self.rules or self.rules[key].method in [None, 'done', 'scalar']:
            return self._concatenate(data)
        else:
            ref_key = self.rules[key].ref_key
            if self.rules[key].method == 'tensor':
                return [data[g][mask] for g in range(self.num_gpus) for mask in self.masks[ref_key][g]]
            elif self.rules[key].method == 'tensor_list':
                return [[d[self.masks[ref_key][g][i][b]] for i, d in enumerate(data[g])] for g in range(self.num_gpus) for b in range(self.batch_size)]
            elif self.rules[key].method == 'edge_tensor':
                return [data[g][mask]-(key==ref_key[0])*self.offsets[ref_key[0]][g][i] for g in range(self.num_gpus) for i, mask in enumerate(self.masks[ref_key[0]][g])]

    def _concatenate(self, data):
        '''
        Simply concatenates the lists coming from each GPU

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : list
            Data product
        '''
        if isinstance(data[0], (int, float)):
            if len(data) == 1:
                return [data[0] for i in range(self.batch_size)]
            elif len(data) == self.batch_count:
                return data
            else:
                raise ValueError('Only accept scalar arrays of size 1 or batch_size: '+\
                                 f'{len(data)} != {self.batch_size}')
        if isinstance(data[0], list):
            concat_data = []
            for d in data:
                concat_data += d
            return concat_data
        elif isinstance(data[0], np.ndarray):
            return np.concatenate(data)
        else:
            raise TypeError('Unexpected data type', type(data[0]))
