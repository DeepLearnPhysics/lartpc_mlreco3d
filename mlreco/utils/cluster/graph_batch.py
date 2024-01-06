from typing import List, AnyStr
import numpy as np

import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat

import torch_geometric
from torch_geometric.data import Data, Batch

class GraphBatch(Batch):
    '''
    The <get_example> method from torch_geometric.data.Batch class.
    © Copyright 2021, Matthias Fey.

    This is a temporary member function addition, as <get_example> was added
    after the recent release 1.6.3 (Dec. 2 2020)
    '''

    def __init__(self, **kwargs):
        super(GraphBatch, self).__init__(**kwargs)

    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        keys = list(set(data_list[0].keys) - set(exclude_keys))
        assert 'batch' not in keys and 'ptr' not in keys

        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []
        batch['ptr'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, Tensor) and item.dim() == 0:
                    cat_dim = None
                cat_dims[key] = cat_dim

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                    device = item.device()

                batch[key].append(item)  # Append item to the attribute list.

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size, ), i, dtype=torch.long,
                                       device=device))

            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def get_example(self, idx: int) -> Data:
        r"""Reconstructs the :class:`torch_geometric.data.Data` object at index
        :obj:`idx` from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if isinstance(self.x, torch.Tensor):
            x_mask  = torch.nonzero(self.batch == idx).flatten()
            x_offset = x_mask[0] if len(x_mask) else 0
            e_mask  = torch.nonzero(self.batch[self.edge_index[0]] == idx).flatten()
        else:
            x_mask = np.where(self.batch == idx)[0]
            x_offset = 0
            e_mask = np.where(self.batch[self.edge_index[0]] == idx)[0]

        data = Data()
        data.x = self.x[x_mask]
        data.pos = self.pos[x_mask]
        data.edge_index = self.edge_index[:,e_mask] - x_offset
        data.edge_attr = self.edge_attr[e_mask]
        if hasattr(self, 'edge_truth') and self.edge_truth is not None:
            data.edge_truth = self.edge_truth[e_mask]

        return data

    def add_node_features(self, node_feats, name : AnyStr, dtype=None):
        if hasattr(self, name):
            print('''GraphBatch already has attribute : {}, setting attribute \
                to new value.''')
            assert (getattr(self, name).shape == node_feats.shape)
            setattr(self, name, node_feats)
        else:
            device = self.x.device
            feats = node_feats
            if not isinstance(node_feats, Tensor):
                assert dtype is not None
                feats = Tensor(node_feats).to(device, dtype=dtype)
            setattr(self, name, feats)
            self.__slices__[name] = self.__slices__['x']
            self.__cat_dims__[name] = self.__cat_dims__['x']
            self.__cumsum__[name] = self.__cumsum__['x']


    def add_edge_features(self, edge_feats, name : AnyStr, dtype=None):

        device = self.edge_attr.device
        feats = edge_feats
        if not isinstance(edge_feats, Tensor):
            assert dtype is not None
            feats = Tensor(edge_feats).to(device, dtype=dtype)
        setattr(self, name, feats)
        self.__slices__[name] = self.__slices__['edge_index']
        self.__cat_dims__[name] = self.__cat_dims__['x']
        self.__cumsum__[name] = self.__cumsum__['x']
