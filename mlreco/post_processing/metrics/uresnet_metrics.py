import numpy as np
import scipy
import os
from mlreco.utils import CSVData


def uresnet_metrics(cfg, module_cfg, data_blob, res, logdir, iteration):
    import torch
    # UResNet prediction
    if not 'segmentation' in res: return

    method_cfg = cfg['post_processing']['uresnet_metrics']

    index        = data_blob['index']
    segment_data = res['segmentation']
    # input_data   = data_blob.get('input_data' if method_cfg is None else method_cfg.get('input_data', 'input_data'), None)
    segment_label = data_blob.get('segment_label' if method_cfg is None else method_cfg.get('segment_label', 'segment_label'), None)
    num_classes = 5 if method_cfg is None else method_cfg.get('num_classes', 5)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout=CSVData(os.path.join(logdir, 'uresnet-metrics-iter-%07d.csv' % iteration))

    for data_idx, tree_idx in enumerate(index):

        if not store_per_iteration:
            fout=CSVData(os.path.join(logdir, 'uresnet-metrics-event-%07d.csv' % tree_idx))

        predictions = np.argmax(segment_data[data_idx],axis=1)
        label = segment_label[data_idx][:, -1]
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        acc = (predictions == label).sum() / float(len(label))
        class_acc = []
        pix = []
        for c1 in range(num_classes):
            for c2 in range(num_classes):
                class_mask = label == c1
                class_acc.append((predictions[class_mask] == c2).sum() / float(np.count_nonzero(class_mask)))
                pix.append(np.count_nonzero((label == c1) & (predictions == c2)))
        fout.record(('idx', 'acc') + tuple(['confusion_%d_%d' % (c1, c2) for c1 in range(num_classes) for c2 in range(num_classes)]) + tuple(['num_pix_%d_%d' % (c1, c2) for c1 in range(num_classes) for c2 in range(num_classes)]),
                    (tree_idx, acc) + tuple(class_acc) + tuple(pix))
        fout.write()


        if not store_per_iteration: fout.close()

    if store_per_iteration: fout.close()
