import numpy as np
import os
from mlreco.utils import CSVData
from scipy.special import softmax as softmax_func
from scipy.stats import entropy

from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label


def default_gnn_metrics(cfg, 
                        processsor_cfg, 
                        data_blob, 
                        result, 
                        logdir, 
                        iteration):

    clust_label = data_blob['cluster_label']
    clusts = result['clusts']
    index = data_blob['index']

    num_batches = len(clusts)
    assert num_batches == len(result['node_pred'])

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'default-gnn-metrics.csv'), append=append)

    for batch_id, logits in enumerate(result['node_pred_type']):

        labels_batch = clust_label[batch_id]

        event_particle_labels = get_cluster_label(labels_batch, 
                                                  clusts[batch_id], 
                                                  column=9)

        valid = np.nonzero(event_particle_labels > -1)[0]
        num_valid = valid.shape[0]

        probas = softmax_func(logits, axis=1)
        entropy_event = entropy(probas, axis=1)

        proba_valid = probas[valid]
        truth_valid = event_particle_labels[valid]
        pred_valid = np.argmax(logits[valid], axis=1)
        entropy_valid = entropy_event[valid]

        for i in range(num_valid):

            fout.record(('Index', 
                         'Truth', 'Prediction', 'Entropy',
                         'p0', 'p1', 'p2', 'p3', 'p4'),
                        (int(index[batch_id]), 
                         int(truth_valid[i]), int(pred_valid[i]), entropy_valid[i], 
                         proba_valid[i][0], proba_valid[i][1], proba_valid[i][2], proba_valid[i][3], proba_valid[i][4]))
            fout.write()

    fout.close()


def evidential_gnn_metrics(cfg, 
                           processor_cfg, 
                           data_blob, 
                           result, 
                           logdir, 
                           iteration):

    clust_label = data_blob['cluster_label']
    clusts = result['clusts']
    index = data_blob['index']

    num_batches = len(clusts)
    assert num_batches == len(result['node_pred']) == len(clust_label)

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'evidential-gnn-metrics.csv'), append=append)

    for batch_id, evidence in enumerate(result['node_pred_type']):

        labels_batch = clust_label[batch_id]

        event_particle_labels = get_cluster_label(labels_batch, 
                                                  clusts[batch_id], 
                                                  column=9)
        concentration = evidence + 1.0
        S = np.sum(concentration, axis=1)
        uncertainty = evidence.shape[1] / S
        p = concentration / S.reshape(-1, 1)

        valid = np.nonzero(event_particle_labels > -1)[0]
        num_valid = valid.shape[0]

        if num_valid < 1:
            continue

        p_valid = p[valid]
        truth_valid = event_particle_labels[valid]
        pred_valid = np.argmax(p_valid, axis=1)
        
        entropy_event = entropy(p_valid, axis=1)
        uncertainty_event = uncertainty[valid]

        for i in range(num_valid):

            fout.record(('Index', 'loss', 
                         'Truth', 'Prediction', 
                         'Entropy', 'Uncertainty', 'Strength',
                         'p0', 'p1', 'p2', 'p3', 'p4'),
                        (int(index[batch_id]), 
                         int(truth_valid[i]), int(pred_valid[i]), 
                         entropy_event[i], uncertainty_event[i], S[valid][i], 
                         p_valid[i][0], p_valid[i][1], p_valid[i][2], p_valid[i][3], p_valid[i][4]))
            fout.write()

    fout.close()