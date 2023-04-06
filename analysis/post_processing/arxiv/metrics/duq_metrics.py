import numpy as np
import os
from mlreco.utils import CSVData
from scipy.stats import entropy


def duq_metrics(cfg,
                processor_cfg,
                data_blob,
                result,
                logdir,
                iteration):
    import umap
    
    labels = data_blob['label'][0][:, 0]
    index = data_blob['index']

    score = result['score'][0]
    pred = np.argmax(score, axis=1)
    probability = (score + 1e-6) / np.sum(score + 1e-6, axis=1, keepdims=True)
    embedding = result['embedding'][0]
    centroids = result['centroids'][0]
    uncertainty = np.linalg.norm(centroids.reshape(1, -1, 5) - embedding, axis=1)
    uncertainty = uncertainty[np.arange(pred.shape[0]), pred]

    np.save(os.path.join(logdir, 'centroids'), centroids)

    print(centroids)

    pred_entropy = entropy(probability, axis=1)
    latent = np.zeros((embedding.shape[0], 2, embedding.shape[2]))

    for c in range(embedding.shape[2]):
        reduced = umap.UMAP(n_components=2).fit_transform(embedding[:, :, c])
        latent[:, :, c] = reduced

    latent = latent[np.arange(embedding.shape[0]), :, pred]

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'duq-singlep-metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        latent_batch = latent[batch_id]
        labels_batch = labels[batch_id]

        p = probability[batch_id]
        unc = uncertainty[batch_id]
        ent = pred_entropy[batch_id]

        fout.record(('Index', 'Truth', 'Prediction',
                    'p0', 'p1', 'p2', 'p3', 'p4', 'uncertainty', 'entropy',
                    'x', 'y'),
                    (int(event_id), int(labels_batch), int(pred[batch_id]),
                     p[0], p[1], p[2], p[3], p[4], unc, ent,
                     latent_batch[0], latent_batch[1]))
        fout.write()

    fout.close()
