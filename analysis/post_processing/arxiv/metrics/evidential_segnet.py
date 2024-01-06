import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData

from scipy.special import softmax as softmax_func
from scipy.stats import entropy

import torch


def evidential_segnet_metrics(cfg, processor_cfg, data_blob, result, logdir, iteration):

    labels = data_blob['segment_label'][0]
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

    min_samples = processor_cfg['min_samples']

    if iteration:
        append = True
    else:
        append = False

    fout_voxel = CSVData(os.path.join(logdir, 'evidential-segnet-metrics-voxels.csv'), append=append)
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

        df = np.concatenate([np.ones((label_batch.shape[0], 1)) * event_id,
                             label_batch.reshape(-1, 1), 
                             pred_batch.reshape(-1, 1), 
                             softmax_batch, 
                             uncertainty_batch.reshape(-1, 1), 
                             entropy_batch.reshape(-1, 1)], axis=1)

        columns = ['Index', 'Truth', 'Prediction', 'p0', 'p1', 'p2', 'p3', 'p4', 'Uncertainty', 'Entropy']

        df = pd.DataFrame(df, columns=columns)
        avg_entropy = df['Entropy'].to_numpy().mean()
        max_entropy = df['Entropy'].to_numpy().max()

        accuracy = np.sum(df['Truth'] == df['Prediction']) / float(df.shape[0])

        fout.record(('Index', 'Mean Entropy', 'Max Entropy', 'Accuracy'),
                    (int(event_id), avg_entropy, max_entropy, accuracy))
        fout.write()
        
        for c in np.unique(label_batch.astype(int)):
            df_slice = df.query('Truth == {}'.format(c))
            num_total_voxels = df_slice.shape[0]
            num_samples = np.ceil(num_total_voxels * 0.05).astype(int)
            if min_samples > num_total_voxels:
                samples = df_slice
            else:
                samples = df_slice.sample(num_samples)
            fout_voxel.record(samples)

    fout.close()