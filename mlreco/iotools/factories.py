from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import DataLoader


def loader_handmade(name, minibatch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=None,
                    sampler=None,
                    **args):
    import mlreco.iotools.collates
    import mlreco.iotools.samplers
    import mlreco.iotools.datasets
    ds = getattr(mlreco.iotools.datasets,name)(**args)
    if collate_fn is not None:
        collate_fn = getattr(mlreco.iotools.collates,collate_fn)
        loader = DataLoader(ds,
                            minibatch_size  = minibatch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers,
                            collate_fn  = collate_fn)
    else:
        loader = DataLoader(ds,
                            minibatch_size  = minibatch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers)
    return loader,ds.data_keys()

def dataset_factory(cfg,event_list=None):
    import mlreco.iotools.datasets
    params = cfg['iotool']['dataset']
    if event_list is not None:
        params['event_list'] = str(list(event_list))
    return getattr(mlreco.iotools.datasets, params['name']).create(params)

def loader_factory(cfg,event_list=None):
    params = cfg['iotool']
    minibatch_size = int(params['minibatch_size'])
    shuffle      = True if not 'shuffle' in params     else bool(params['shuffle'    ])
    num_workers  = 1    if not 'num_workers' in params else int (params['num_workers'])
    collate_fn   = None if not 'collate_fn' in params  else str (params['collate_fn' ])

    if not int(params['batch_size']) % int(params['minibatch_size']) == 0:
        print('iotools.batch_size (',params['batch_size'],'must be divisble by iotools.minibatch_size',params['minibatch_size'])
        raise ValueError
    
    import mlreco.iotools.collates
    import mlreco.iotools.samplers

    ds = dataset_factory(cfg,event_list)
    sampler = None
    if 'sampler' in cfg['iotool']:
        sam_cfg = cfg['iotool']['sampler']
        sam_cfg['minibatch_size']=cfg['iotool']['minibatch_size']
        sampler = getattr(mlreco.iotools.samplers,sam_cfg['name']).create(ds,sam_cfg)
    if collate_fn is not None:
        collate_fn = getattr(mlreco.iotools.collates,collate_fn)
        loader = DataLoader(ds,
                            batch_size  = minibatch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers,
                            collate_fn  = collate_fn)
    else:
        loader = DataLoader(ds,
                            batch_size  = minibatch_size,
                            shuffle     = shuffle,
                            sampler     = sampler,
                            num_workers = num_workers)
    return loader
