import numpy as np
import scipy
import os
from mlreco.utils import CSVData

def store_uresnet(cfg, data_blob, res, logdir, iteration):
    # UResNet prediction
    if not 'segmentation' in res: return

    method_cfg = cfg['post_processing']['store_uresnet']

    index        = data_blob['index']
    segment_data = res['segmentation']
    input_data   = data_blob.get('input_data' if method_cfg is None else method_cfg.get('input_data', 'input_data'), None)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout=CSVData(os.path.join(logdir, 'uresnet-segmentation-iter-%07d.csv' % iteration))

    for data_idx, tree_idx in enumerate(index):

        if not store_per_iteration:
            fout=CSVData(os.path.join(logdir, 'uresnet-segmentation-event-%07d.csv' % tree_idx))

        predictions = np.argmax(segment[data_idx],axis=1)
        for row in predictions:
            event = input_data[i]
            fout.record(('idx','x', 'y', 'z', 'type', 'value'),
                        (idx,event[0], event[1], event[2], 4, row))
            fout.write()

        if not store_per_iteration: fout.close()

    if store_per_iteration: fout.close()
