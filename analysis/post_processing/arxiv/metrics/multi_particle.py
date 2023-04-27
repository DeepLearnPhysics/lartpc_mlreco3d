import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label

from scipy.special import softmax
from scipy.stats import entropy

import torch

def multi_particle(cfg, processor_cfg, data_blob, result, logdir, iteration):

    output = pd.DataFrame(columns=['p0', 'p1', 'p2', 'p3',
        'p4', 'prediction', 'truth', 'index', 'entropy'])

    index = data_blob['index']
    logits = result['logits']
    clusts = result['clusts']
    
    labels = get_cluster_label(data_blob['input_data'][0], clusts, 9)
    primary_labels = get_cluster_label(data_blob['input_data'][0], clusts, 15)

    logits = np.vstack(logits)

    pred = np.argmax(logits, axis=1)
    index = np.asarray(index)

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'multi-particle-metrics.csv'), append=append)

    for i in range(len(labels)):

        logit_batch = logits[i]
        pred = np.argmax(logit_batch)
        label_batch = labels[i]

        probs = softmax(logit_batch)
        ent = entropy(probs)

        fout.record(('Index', 'Truth', 'Prediction', 
                    'p0', 'p1', 'p2', 'p3', 'p4', 'entropy', 'is_primary'),
                    (int(i), int(label_batch), int(pred),
                     probs[0], probs[1], probs[2], probs[3], probs[4], ent, int(primary_labels[i])))
        fout.write()

    fout.close()