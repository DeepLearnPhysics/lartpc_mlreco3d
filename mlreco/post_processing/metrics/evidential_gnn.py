from mlreco.utils.gnn.cluster import get_cluster_label
import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData

from scipy.special import softmax as softmax_func
from scipy.stats import entropy

import torch


def evidential_gnn_metrics(cfg, processor_cfg, data_blob, result, logdir, iteration):

    clust_label = torch.Tensor(data_blob['clust_label'][0])
    clusts = result['clusts']
    index = data_blob['index']

    num_batches = len(clusts)
    assert num_batches == len(result['node_pred'])

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'evidential-segnet-metrics.csv'), append=append)

    for batch_id, evidence in enumerate(result['node_pred_type']):

        batch_index = clust_label[:, 3] == batch_id
        labels_batch = clust_label[batch_index]

        event_particle_labels = get_cluster_label(labels_batch, clusts[batch_id], column=7)
        concentration = evidence + 1.0
        S = np.sum(concentration, axis=1)
        uncertainty = evidence.shape[1] / S
        p = concentration / S.reshape(-1, 1)

        valid = np.nonzero(event_particle_labels > -1)[0]
        num_valid = valid.shape[0]


        p_valid = p[valid]
        truth_valid = event_particle_labels[valid]
        pred_valid = np.argmax(p_valid, axis=1)
        
        entropy_event = entropy(p_valid, axis=1)
        uncertainty_event = uncertainty[valid]

        for i in range(num_valid):

            fout.record(('Index', 'Truth', 'Prediction', 'Entropy', 'Uncertainty'),
                        (int(index[batch_id]), int(truth_valid[i]), 
                         int(pred_valid[i]), entropy_event[i], uncertainty_event[i]))
            fout.write()

    fout.close()