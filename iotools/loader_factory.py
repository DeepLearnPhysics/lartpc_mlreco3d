from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data
from torch.utils.data import DataLoader

def loader_factory(name, batch_size,
                   shuffle=True,
                   num_workers=1,
                   collate_fn=None,
                   sampler=None,
                   **args):
    import iotools.datasets
    import iotools.collates
    import iotools.samplers
    ds = getattr(iotools.datasets,name)(**args)
    if collate_fn is not None:
        collate_fn = getattr(iotools.collates,collate_fn)
        loader = DataLoader(ds,
                            batch_size  = batch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers,
                            collate_fn  = collate_fn)
    else:
        loader = DataLoader(ds,
                            batch_size  = batch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers)
    return loader,ds.data_keys()

def loader_from_config(cfg):

    params = cfg['iotool']
    batch_size   = int(params['batch_size'])
    shuffle      = True if not 'shuffle' in params     else bool(params['shuffle'    ])
    num_workers  = 1    if not 'num_workers' in params else int (params['num_workers'])
    collate_fn   = None if not 'collate_fn' in params  else str (params['collate_fn' ])

    import iotools.datasets
    import iotools.collates
    import iotools.samplers
    ds_cfg = cfg['iotool']['dataset']
    ds = getattr(iotools.datasets,ds_cfg['name']).create(ds_cfg)
    sampler = None
    if 'sampler' in cfg['iotool']:
        sam_cfg = cfg['iotool']['sampler']
        sampler = getattr(iotools.samplers,sam_cfg['name']).create(ds,sam_cfg)
    if collate_fn is not None:
        collate_fn = getattr(iotools.collates,collate_fn)
        loader = DataLoader(ds,
                            batch_size  = batch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers,
                            collate_fn  = collate_fn)
    else:
        loader = DataLoader(ds,
                            batch_size  = batch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers)
    return loader,ds.data_keys()

    
    
