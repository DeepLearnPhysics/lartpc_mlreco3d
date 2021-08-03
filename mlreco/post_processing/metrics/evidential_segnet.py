import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData

from scipy.special import softmax as softmax_func
from scipy.stats import entropy

import torch


def evidential_segnet_metrics(cfg, processor_cfg, data_blob, result, logdir, iteration):

    labels = data_blob['segment_label'][0].cpu().numpy()
    index = data_blob['index']
    # logits = result['logits'][0]
    if processor_cfg['mode'] != 'mc_dropout':
        softmax = softmax_func(result['segmentation'][0], axis=1)
    else:
        softmax = result['softmax'][0]
        segmentation = result['segmentation'][0]

    softmax = result['expected_probability'][0]
    uncertainty = result['uncertainty'][0].squeeze()
    pred = np.argmax(softmax, axis=1)
    index = np.asarray(index)
    batch_index = labels[:, 0]

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'evidential-segnet-metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        batch_mask = batch_index == batch_id
        input_batch = data_blob['input_data'][0][batch_mask]
        label_batch = labels[labels[:, 0].astype(int) == batch_id][:, -1].astype(int)
        pred_batch = pred[batch_mask].squeeze()
        softmax_batch = softmax[batch_mask]
        uncertainty_batch = uncertainty[batch_mask]
        
        entropy_batch = entropy(softmax_batch, axis=1)

        for i in range(input_batch.shape[0]):

            fout.record(('Index', 'Truth', 'Prediction', 'Entropy', 'Uncertainty'),
                        (int(event_id), int(label_batch[i]), 
                         int(pred_batch[i]), entropy_batch[i], uncertainty_batch[i]))
            fout.write()

    fout.close()