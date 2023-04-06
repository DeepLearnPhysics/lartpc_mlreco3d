import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData

from scipy.special import softmax
from scipy.stats import entropy

import torch

def single_particle(cfg, processor_cfg, data_blob, result, logdir, iteration):

    output = pd.DataFrame(columns=['p0', 'p1', 'p2', 'p3',
        'p4', 'prediction', 'truth', 'index', 'entropy'])

    labels = data_blob['label'][0][:, 0]
    index = data_blob['index']
    logits = result['logits'][0]
    pred = np.argmax(logits, axis=1)
    index = np.asarray(index)

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'single-particle-metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        logit_batch = logits[batch_id]
        pred = np.argmax(logit_batch)
        label_batch = labels[batch_id]

        probs = softmax(logit_batch)
        ent = entropy(probs)

        fout.record(('Index', 'Truth', 'Prediction', 
                    'p0', 'p1', 'p2', 'p3', 'p4', 'entropy'),
                    (int(event_id), int(label_batch), int(pred),
                     probs[0], probs[1], probs[2], probs[3], probs[4], ent))
        fout.write()

    fout.close()