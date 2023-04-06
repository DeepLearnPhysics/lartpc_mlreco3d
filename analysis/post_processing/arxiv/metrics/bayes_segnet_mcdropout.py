import numpy as np
import pandas as pd
import os

from mlreco.utils import CSVData

from scipy.special import softmax as softmax_func
from scipy.stats import entropy

def bayes_segnet_mcdropout(cfg, 
                           processor_cfg, 
                           data_blob, 
                           result, 
                           logdir, 
                           iteration):

    labels = data_blob['segment_label'][0]
    index = data_blob['index']
    # logits = result['logits'][0]
    if processor_cfg['mode'] != 'mc_dropout':
        softmax = softmax_func(result['segmentation'][0], axis=1)
    else:
        softmax = result['softmax'][0]
        segmentation = result['segmentation'][0]

    pred = np.argmax(result['segmentation'][0], axis=1)
    index = np.asarray(index)
    batch_index = data_blob['input_data'][0][:, 0].astype(int)

    if iteration:
        append = True
    else:
        append = False

    min_samples = processor_cfg['min_samples']

    fout = CSVData(
        os.path.join(logdir, 'bayes-segnet-metrics.csv'), append=append)

    fout_voxel = CSVData(
        os.path.join(logdir, 'bayes-segnet-metrics-voxels.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        batch_mask = batch_index == batch_id
        input_batch = data_blob['input_data'][0][batch_mask]
        label_mask = labels[:, 0].astype(int) == batch_id
        label_batch = labels[label_mask][:, -1].astype(int)
        pred_batch = pred[batch_mask].squeeze()
        softmax_batch = softmax[batch_mask]
        
        entropy_batch = entropy(softmax_batch, axis=1)

        df = np.concatenate([np.ones((label_batch.shape[0], 1)) * event_id,
                             label_batch.reshape(-1, 1), 
                             pred_batch.reshape(-1, 1), 
                             softmax_batch, 
                             entropy_batch.reshape(-1, 1)], axis=1)
        
        columns = ['Index', 'Truth', 'Prediction', 
                   'p0', 'p1', 'p2', 'p3', 'p4', 'Entropy']

        df = pd.DataFrame(df, columns=columns)
        avg_entropy = df['Entropy'].mean()
        median_entropy = df['Entropy'].median()

        accuracy = np.sum(df['Truth'] == df['Prediction']) / float(df.shape[0])

        fout.record(('Index', 'Mean Entropy', 'Median Entropy', 'Accuracy'),
                    (int(event_id), avg_entropy, median_entropy, accuracy))
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