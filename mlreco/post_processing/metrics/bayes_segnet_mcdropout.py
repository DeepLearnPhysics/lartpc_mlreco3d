import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData

from scipy.special import softmax as softmax_func
from scipy.stats import entropy

import torch

def bayes_segnet_mcdropout(cfg, processor_cfg, data_blob, result, logdir, iteration):

    labels = data_blob['segment_label'][0].cpu().numpy()
    index = data_blob['index']
    # logits = result['logits'][0]
    if processor_cfg['mode'] != 'mc_dropout':
        softmax = softmax_func(result['segmentation'][0], axis=1)
    else:
        softmax = result['softmax'][0]
        segmentation = result['segmentation'][0]

    pred = np.argmax(softmax, axis=1)
    index = np.asarray(index)
    batch_index = data_blob['input_data'][0][:, 0].cpu().numpy().astype(int)

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'bayes-segnet-metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        batch_mask = batch_index == batch_id
        input_batch = data_blob['input_data'][0][batch_mask]
        label_batch = labels[labels[:, 0].astype(int) == batch_id][:, -1].astype(int)
        pred_batch = pred[batch_mask].squeeze()
        softmax_batch = softmax[batch_mask]

        accuracy = np.sum(pred_batch == label_batch) / pred_batch.shape[0]

        wrongs = np.where(pred_batch != label_batch)[0]
        wrong_probs = softmax_batch[wrongs]
        corrects = np.where(pred_batch == label_batch)[0]
        correct_probs = softmax_batch[corrects]

        num_samples = wrongs.shape[0]
        
        perm = np.random.randint(0, correct_probs.shape[0], num_samples)
        
        ent_correct = entropy(correct_probs[perm], axis=1)
        ent_wrong = entropy(wrong_probs, axis=1)

        for i in range(num_samples):

            fout.record(('Index', 'Accuracy', 'entropy_wrong', 'entropy_correct'),
                        (int(event_id), float(accuracy), ent_wrong[i], ent_correct[i]))
            fout.write()

    fout.close()