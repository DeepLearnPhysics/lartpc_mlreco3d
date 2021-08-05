import numpy as np
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

    labels = data_blob['segment_label'][0].cpu().numpy()
    index = data_blob['index']
    # logits = result['logits'][0]
    if processor_cfg['mode'] != 'mc_dropout':
        softmax = softmax_func(result['segmentation'][0], axis=1)
    else:
        softmax = result['softmax'][0]
        segmentation = result['segmentation'][0]

    pred = np.argmax(result['segmentation'][0], axis=1)
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
        label_mask = labels[:, 0].astype(int) == batch_id
        label_batch = labels[label_mask][:, -1].astype(int)
        pred_batch = pred[batch_mask].squeeze()
        softmax_batch = softmax[batch_mask]
        
        entropy_batch = entropy(softmax_batch, axis=1)

        perm = np.arange(input_batch.shape[0])
        perm = np.random.permutation(perm)
        num_voxels = int(np.floor(input_batch.shape[0] * 0.1))
        perm = perm[:num_voxels]

        input_batch_selected = input_batch[perm]
        label_batch_selected = label_batch[perm]
        pred_batch_selected = pred_batch[perm]
        entropy_batch_selected = entropy_batch[perm]

        for i in range(input_batch_selected.shape[0]):

            fout.record(('Index', 
                         'Truth', 
                         'Prediction', 
                         'Entropy'),
                        (int(event_id), 
                         int(label_batch_selected[i]), 
                         int(pred_batch_selected[i]), 
                         entropy_batch_selected[i]))
            fout.write()

    fout.close()