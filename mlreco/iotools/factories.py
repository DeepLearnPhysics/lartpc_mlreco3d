from copy import deepcopy
from torch.utils.data import DataLoader


def dataset_factory(cfg, event_list=None):
    """
    Instantiates dataset based on type specified in configuration under
    `iotool.dataset.name`. The name must match the name of a class under
    `mlreco.iotools.datasets`.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expects a field `iotool`.
    event_list: list, optional
        List of tree idx.

    Returns
    -------
    dataset: torch.utils.data.Dataset

    Note
    ----
    Currently the choice is limited to `LArCVDataset` only.
    """
    import mlreco.iotools.datasets
    params = cfg['iotool']['dataset']
    if event_list is not None:
        params['event_list'] = str(list(event_list))
    return getattr(mlreco.iotools.datasets, params['name']).create(params)


def loader_factory(cfg, event_list=None):
    """
    Instantiates a DataLoader based on configuration.

    Dataset comes from `dataset_factory`.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expects a field `iotool`.
    event_list: list, optional
        List of tree idx.

    Returns
    -------
    loader : torch.utils.data.DataLoader
    """
    params = cfg['iotool']
    minibatch_size = int(params['minibatch_size'])
    shuffle      = False if not 'shuffle' in params     else bool(params['shuffle'    ])
    num_workers  = 1     if not 'num_workers' in params else int (params['num_workers'])
    collate_fn   = None  if not 'collate_fn' in params  else str (params['collate_fn' ])
    collate_kwargs = {}

    if collate_fn is None:
        collate_params = params.get('collate', {})
        collate_fn = None if not 'collate_fn' in collate_params else str(collate_params['collate_fn'])
        collate_kwargs = {k:v for k, v in collate_params.items() if k != 'collate_fn'}

    if not int(params['batch_size']) % int(params['minibatch_size']) == 0:
        print('iotools.batch_size (',params['batch_size'],'must be divisble by iotools.minibatch_size',params['minibatch_size'])
        raise ValueError

    import mlreco.iotools.collates
    import mlreco.iotools.samplers
    from functools import partial

    ds = dataset_factory(cfg,event_list)
    sampler = None
    if 'sampler' in cfg['iotool']:
        sam_cfg = cfg['iotool']['sampler']
        sam_cfg['minibatch_size']=cfg['iotool']['minibatch_size']
        sampler = getattr(mlreco.iotools.samplers,sam_cfg['name']).create(ds,sam_cfg)
    if collate_fn is not None:
        collate_fn = partial(getattr(mlreco.iotools.collates,collate_fn), **collate_kwargs)
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


def reader_factory(cfg):
    """
    Instantiates writer based on type specified in configuration under
    `iotool.reader.name`. The name must match the name of a class under
    `mlreco.iotools.readers`.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expects a field `iotool`.

    Returns
    -------
    reader

    Note
    ----
    Currently the choice is limited to `HDF5Reader` only.
    """
    import mlreco.iotools.readers
    params = deepcopy(cfg)
    name   = params.pop('name')
    reader = getattr(mlreco.iotools.readers, name)(**params)
    return reader


def writer_factory(cfg):
    """
    Instantiates writer based on type specified in configuration under
    `iotool.writer.name`. The name must match the name of a class under
    `mlreco.iotools.writers`.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expects a field `iotool`.

    Returns
    -------
    writer

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    """
    import mlreco.iotools.writers
    params = deepcopy(cfg)
    name   = params.pop('name')
    writer = getattr(mlreco.iotools.writers, name)(**params)
    return writer
