import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .globals import *
from .volumes import VolumeBoundaries


class Unwrapper:
    '''
    Tools to break down the input and output dictionaries into individual events.
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
        self.num_volumes = self.merger.num_volumes() if self.merger else 1
        self.rules = self._parse_rules(rules)

    def __call__(self, data_blob, result_blob):
        '''
        Main unwrapping function. Loops over the data and result keys
        and applies the unwrapping rules. Returns the unwrapped versions
        of the two dictionaries

        Parameters
        ----------
        data_blob : dict
            Dictionary of array of array of minibatch data [key][num_gpus][batch_size]
        result_blob : dict
            Results dictionary, output of trainval.forward [key][num_gpus][batch_size]
        '''
        self._build_batch_masks(data_blob, result_blob)
        data_unwrapped, result_unwrapped = {}, {}
        for key, value in data_blob.items():
            data_unwrapped[key] = self._unwrap(key, value)
        for key, value in result_blob.items():
            result_unwrapped[key] = self._unwrap(key, value)

        return data_unwrapped, result_unwrapped

    @dataclass
    class Rule:
        '''
        Simple dataclass which stores the relevant
        unwrapping rule attributes for a speicific
        data product human-readable names.

        Attributes
        ----------
        method : str
            Unwrapping scheme
        ref_key : str
            Key of the data product that supplies the batch mapping
        done : bool
            True if the unwrapping is done by the model internally
        translate : tuple
            List of column indices that correspond to coordinates to correct
        '''
        method    : str = None
        ref_key   : str = None
        done      : bool = False
        translate : bool = False

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
        valid_methods = [None, 'scalar', 'list', 'tensor', 'tensor_list', 'edge_tensor', 'index_tensor', 'index_list']
        parsed_rules = {}
        for key, rule in rules.items():
            parsed_rules[key] = self.Rule(*rule)
            if not parsed_rules[key].ref_key:
                parsed_rules[key].ref_key = key

            assert parsed_rules[key].method in valid_methods,\
                    f'Unwrapping method {parsed_rules[key].method} for {key} not valid'

        return parsed_rules

    def _build_batch_masks(self, data_blob, result_blob):
        '''
        For all the returned data objects that require a batch mask:
        build it and store it. Also store the index offsets within that
        batch, wherever necessary to unwrap.

        Parameters
        ----------
        data_blob : dict
            Dictionary of array of array of minibatch data [key][num_gpus][batch_size]
        result_blob : dict
            Results dictionary, output of trainval.forward [key][num_gpus][batch_size]
        '''
        comb_blob = dict(data_blob, **result_blob)
        self.masks, self.offsets = {}, {}
        for key in comb_blob.keys():
            # Skip outputs with no rule
            if key not in self.rules:
                continue

            # For tensors and lists of tensors, build one mask per reference tensor
            if not self.rules[key].done and self.rules[key].method in ['tensor', 'tensor_list']:
                ref_key = self.rules[key].ref_key
                if ref_key not in self.masks:
                    assert ref_key in comb_blob, f'Must provide reference tensor ({ref_key}) to unwrap {key}'
                    assert self.rules[key].method == self.rules[ref_key].method, f'Reference ({ref_key}) must be of same type as {key}'
                    if self.rules[key].method == 'tensor':
                        self.masks[ref_key] = [self._batch_masks(comb_blob[ref_key][g]) for g in range(self.num_gpus)]
                    elif self.rules[key].method == 'tensor_list':
                        self.masks[ref_key] = [[self._batch_masks(v) for v in comb_blob[ref_key][g]] for g in range(self.num_gpus)]

            # For edge tensors, build one mask from each tensor (must figure out batch IDs of edges)
            elif self.rules[key].method == 'edge_tensor':
                assert len(self.rules[key].ref_key) == 2, 'Must provide a reference to the edge_index and the node batch ids'
                for ref_key in self.rules[key].ref_key:
                    assert ref_key in comb_blob, f'Must provide reference tensor ({ref_key}) to unwrap {key}'
                ref_edge, ref_node = self.rules[key].ref_key
                edge_index, batch_ids = comb_blob[ref_edge], comb_blob[ref_node]
                if not self.rules[key].done and ref_edge not in self.masks:
                    self.masks[ref_edge] = [self._batch_masks(batch_ids[g][edge_index[g][:,0]]) for g in range(self.num_gpus)]
                if ref_node not in self.offsets:
                    self.offsets[ref_node] = [self._batch_offsets(batch_ids[g]) for g in range(self.num_gpus)]

            # For an index tensor, only need to record the batch offsets within the wrapped tensor
            elif self.rules[key].method == 'index_tensor':
                ref_key = self.rules[key].ref_key
                assert ref_key in comb_blob, f'Must provide reference tensor ({ref_key}) to unwrap {key}'
                if not self.rules[key].done and ref_key not in self.masks:
                    self.masks[ref_key] = [self._batch_masks(comb_blob[ref_key][g]) for g in range(self.num_gpus)]
                if ref_key not in self.offsets:
                    self.offsets[ref_key] = [self._batch_offsets(comb_blob[ref_key][g]) for g in range(self.num_gpus)]

            # For lists of tensor indices, only need to record the offsets within the wrapped tensor
            elif self.rules[key].method == 'index_list':
                assert len(self.rules[key].ref_key) == 2, 'Must provide a reference to indexed tensor and the index batch ids'
                for ref_key in self.rules[key].ref_key:
                    assert ref_key in comb_blob, f'Must provide reference tensor ({ref_key}) to unwrap {key}'
                ref_tensor, ref_index = self.rules[key].ref_key
                if not self.rules[key].done and ref_index not in self.masks:
                    self.masks[ref_index] = [self._batch_masks(comb_blob[ref_index][g]) for g in range(self.num_gpus)]
                if ref_tensor not in self.offsets:
                    self.offsets[ref_tensor] = [self._batch_offsets(comb_blob[ref_tensor][g]) for g in range(self.num_gpus)]

    def _batch_masks(self, tensor):
        '''
        Makes a list of masks for each batch entry, for a specific tensor.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor with a batch ID column

        Returns
        -------
        list
            List of batch masks
        '''
        # Create batch masks
        masks = []
        for b in range(self.batch_size*self.num_volumes):
            if len(tensor.shape) == 1:
                masks.append(np.where(tensor == b)[0])
            else:
                masks.append(np.where(tensor[:, BATCH_COL] == b)[0])

        return masks

    def _batch_offsets(self, tensor):
        '''
        Computes the index of the first element in a tensor
        for each entry in the batch.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor with a batch ID column

        Returns
        -------
        np.ndarray
            Array of batch offsets
        '''
        # Compute batch offsets
        offsets = np.zeros(self.batch_size*self.num_volumes, np.int64)
        for b in range(1, self.batch_size*self.num_volumes):
            if len(tensor.shape) == 1:
                offsets[b] = offsets[b-1] + np.sum(tensor == b-1)
            else:
                offsets[b] = offsets[b-1] + np.sum(tensor[:, BATCH_COL] == b-1)

        return offsets

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
        # Scalars and lists are trivial to unwrap
        if key not in self.rules or self.rules[key].method in [None, 'scalar', 'list']:
            unwrapped = self._concatenate(data)
        else:
            ref_key = self.rules[key].ref_key
            unwrapped = []
            for g in range(self.num_gpus):
                for b in range(self.batch_size):
                    # Tensor unwrapping
                    if self.rules[key].method == 'tensor':
                        tensors = []
                        for v in range(self.num_volumes):
                            if not self.rules[key].done:
                                tensor = data[g][self.masks[ref_key][g][b*self.num_volumes+v]]
                                if key == ref_key:
                                    if len(tensor.shape) == 2:
                                        tensor[:, BATCH_COL] = v
                                    else:
                                        tensor[:] = v
                                if self.rules[key].translate:
                                    if v > 0:
                                        tensor[:, COORD_COLS] = self.merger.translate(tensor[:,COORD_COLS], v)
                                tensors.append(tensor)
                            else:
                                tensors.append(data[g][b*self.num_volumes+v])
                        unwrapped.append(np.concatenate(tensors))

                    # Tensor list unwrapping
                    elif self.rules[key].method == 'tensor_list':
                        tensors = []
                        for i, d in enumerate(data[g]):
                            subtensors = []
                            for v in range(self.num_volumes):
                                subtensor = d[self.masks[ref_key][g][i][b*self.num_volumes+v]]
                                if key == ref_key:
                                    if len(subtensor.shape) == 2:
                                        subtensor[:, BATCH_COL] = v
                                    else:
                                        subtensor[:] = v
                                if self.rules[key].translate:
                                    if v > 0:
                                        subtensor[:, COORD_COLS] = self.merger.translate(subtensor[:,COORD_COLS], v)
                                subtensors.append(subtensor)
                            tensors.append(np.concatenate(subtensors))
                        unwrapped.append(tensors)

                    # Edge tensor unwrapping
                    elif self.rules[key].method == 'edge_tensor':
                        ref_edge, ref_node = ref_key
                        tensors = []
                        for v in range(self.num_volumes):
                            if not self.rules[key].done:
                                tensor = data[g][self.masks[ref_edge][g][b*self.num_volumes+v]]
                                offset = (key == ref_edge) * self.offsets[ref_node][g][b*self.num_volumes]
                            else:
                                tensor = data[g][b*self.num_volumes+v]
                                offset = (key == ref_edge) *\
                                        (self.offsets[ref_node][g][b*self.num_volumes+v]-self.offsets[ref_node][g][b*self.num_volumes])
                            tensors.append(tensor + offset)
                        unwrapped.append(np.concatenate(tensors))

                    # Index tensor unwrapping
                    elif self.rules[key].method == 'index_tensor':
                        tensors = []
                        for v in range(self.num_volumes):
                            if not self.rules[key].done:
                                offset = self.offsets[ref_key][g][b*self.num_volumes]
                                tensors.append(data[self.masks[ref_key][g][b*self.num_volumes+v]] - offset)
                            else:
                                offset = self.offsets[ref_key][g][b*self.num_volumes+v]-self.offsets[ref_key][g][b*self.num_volumes]
                                tensors.append(data[g][b*self.num_volumes+v] + offset)

                        unwrapped.append(np.concatenate(tensors))

                    # Index list unwrapping
                    elif self.rules[key].method == 'index_list':
                        ref_tensor, ref_index = ref_key
                        index_list = []
                        for v in range(self.num_volumes):
                            if not self.rules[key].done:
                                offset = self.offsets[ref_tensor][g][b*self.num_volumes]
                                for i in self.masks[ref_index][g][b*self.num_volumes+v]:
                                    index_list.append(data[g][i] - offset)
                            else:
                                offset = self.offsets[ref_tensor][g][b*self.num_volumes+v]-self.offsets[ref_tensor][g][b*self.num_volumes]
                                for index in data[g][b*self.num_volumes+v]:
                                    index_list.append(index + offset)

                        index_list_nb    = np.empty(len(index_list), dtype=object)
                        index_list_nb[:] = index_list
                        unwrapped.append(index_list_nb)

        return unwrapped

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
                return [data[g] for g in range(self.num_gpus) for i in range(self.batch_size)]
            else:
                return data
            # elif len(data) == self.batch_count:
            #     return data
            # else:
            #     raise ValueError('Only accept scalar arrays of size 1 or batch_size: '+\
            #                      f'{len(data)} != {self.batch_size}')
        if isinstance(data[0], list):
            concat_data = []
            for d in data:
                concat_data += d
            return concat_data
        elif isinstance(data[0], np.ndarray):
            return np.concatenate(data)
        else:
            raise TypeError('Unexpected data type', type(data[0]))


def prefix_unwrapper_rules(rules, prefix):
    '''
    Modifies the default rules of a module to account for
    a prefix being added to its standard set of outputs

    Parameters
    ----------
    rules : dict
        Dictionary which contains a set of unwrapping rules for each
        output key of a given module in the reconstruction chain.
    prefix : str
        Prefix to add in front of all output names

    Returns
    -------
    dict
        Dictionary of rules containing the appropriate names
    '''
    prules = {}
    for key, value in rules.items():
        pkey = f'{prefix}_{key}'
        prules[pkey] = deepcopy(rules[key])
        if len(value) > 1:
            if isinstance(value[1], str):
                prules[pkey][1] = f'{prefix}_{value[1]}'
            else:
                for i in range(len(value[1])):
                    prules[pkey][1][i] = f'{prefix}_{value[1][i]}'

    return prules
