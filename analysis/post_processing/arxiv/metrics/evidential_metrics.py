import numpy as np
import os
from mlreco.utils import CSVData
from scipy.stats import entropy


def evidential_metrics(cfg, 
                       processor_cfg, 
                       data_blob, 
                       result, 
                       logdir, 
                       iteration):

    labels = data_blob['label'][0][:, 0]
    index = data_blob['index']
    # logits = result['logits'][0]
    softmax = result['expected_probability'][0]
    uncertainty = result['uncertainty'][0].squeeze()
    pred = np.argmax(softmax, axis=1)
    index = np.asarray(index)

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'evidential_metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        probs = softmax[batch_id]
        pred = np.argmax(probs)
        label_batch = labels[batch_id]
        ent = entropy(probs)
        unc = uncertainty[batch_id]

        fout.record(('Index', 
                     'Truth', 'Prediction', 
                     'p0', 'p1', 'p2', 'p3', 'p4', 
                     'uncertainty', 'entropy'),
                    (int(event_id), 
                     int(label_batch), int(pred),
                     probs[0], probs[1], probs[2], probs[3], probs[4], 
                     unc, ent))
        fout.write()

    fout.close()