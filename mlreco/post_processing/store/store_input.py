import os
from mlreco.utils import CSVData


def get_coords(row, data_dim, tree_index):
    if data_dim == 2:
        coords_labels = ('idx', 'x', 'y')
        coords = (tree_index, row[0], row[1])
    elif data_dim == 3:
        coords_labels = ('idx', 'x', 'y', 'z')
        coords = (tree_index, row[0], row[1], row[2])
    else:
        raise Exception("data_dim must be 2 or 3, got %d" % data_dim)
    return coords_labels, coords


def store_input(cfg, data_blob, res, logdir, iteration):
    """
    Store input data blob.

    Parameters
    -------------
    threshold: float, optional
        Default: 0.
    input_data: str, optional
    particles_label: str, optional
    segment_label: str, optional
    clusters_label: str, optional
    cluster3d_mcst_true: str, optional
    store_method: str, optional
        Can be `per-iteration` or `per-event`
    """
    method_cfg = cfg['post_processing']['store_input']

    if (method_cfg is not None and not method_cfg.get('input_data', 'input_data') in data_blob) or (method_cfg is None and 'input_data' not in data_blob): return

    threshold = 0. if method_cfg is None else method_cfg.get('threshold',0.)
    data_dim = 3 if method_cfg is None else method_cfg.get('data_dim', 3)

    index      = data_blob.get('index', None)
    input_dat  = data_blob.get('input_data' if method_cfg is None else method_cfg.get('input_data', 'input_data'), None)
    label_ppn  = data_blob.get('particles_label' if method_cfg is None else method_cfg.get('particles_label', 'particles_label'), None)
    label_seg  = data_blob.get('segment_label' if method_cfg is None else method_cfg.get('segment_label', 'segment_label'), None)
    label_cls  = data_blob.get('clusters_label' if method_cfg is None else method_cfg.get('clusters_label', 'clusters_label'), None)
    label_mcst = data_blob.get('cluster3d_mcst_true' if method_cfg is None else method_cfg.get('cluster3d_mcst_true', 'cluster3d_mcst_true'), None)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout=CSVData(os.path.join(logdir, 'input-iter-%07d.csv' % iteration))

    if input_dat is None: return

    for data_index,tree_index in enumerate(index):

        if not store_per_iteration:
            fout=CSVData(os.path.join(logdir, 'input-event-%07d.csv' % tree_index))

        mask = input_dat[data_index][:,-1] > threshold


        # type 0 = input data
        for row in input_dat[data_index][mask]:
            coords_labels, coords = get_coords(row, data_dim, tree_index)
            fout.record(coords_labels + ('type','value'), coords + (0,row[data_dim+1]))
            fout.write()

        # type 1 = Labels for PPN
        if label_ppn is not None:
            for row in label_ppn[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],1,row[4]))
                fout.write()
        # 2 = UResNet labels
        if label_seg is not None:
            for row in label_seg[data_index][mask]:
                coords_labels, coords = get_coords(row, data_dim, tree_index)
                fout.record(coords_labels + ('type','value'),coords + (2,row[data_dim+1]))
                fout.write()
        # type 15 = group id, 16 = semantic labels, 17 = energy
        if label_cls is not None:
            for row in label_cls[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],15,row[5]))
                fout.write()
            for row in label_cls[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],16,row[6]))
                fout.write()
            for row in label_cls[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],17,row[4]))
                fout.write()
        # type 18 = cluster3d_mcst_true
        if label_mcst is not None:
            for row in label_mcst[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],19,row[4]))
                fout.write()

        if not store_per_iteration: fout.close()

    if store_per_iteration: fout.close()
