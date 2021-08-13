import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn import metrics
from mlreco.utils import CSVData

def instance_clustering(cfg, data_blob, res, logdir, iteration):
    """
    Simple DBSCAN on uresnet clustering output for instance segmentation

    Parameters
    ----------
    data_blob: dict
        Input dictionary returned by iotools
    res: dict
        Results from the network, dictionary using `analysis_keys`
    cfg: dict
        Configuration
    idx: int
        Iteration number

    Input
    -----
    Requires the following analysis keys
    - `segmentation`: output of UResNet segmentation (scores)
    - `clustering`: coordinates in hyperspace, also output of the network
    Requires the following data blob keys
    - `input_data`
    - `segment_label` UResNet 5 classes label
    - `cluster_label`

    Output
    ------
    Writes 2 CSV files:
    - `instance_clustering-*` with the clustering predictions (point type 0 =
    event data, point type 1 = predictions, point type 2 = T-SNE visualizations)
    - `instance_clustering_metrics-*` with some event-wise metrics such as AMI and ARI.
    """

    method_cfg = cfg['post_processing']['instance_clustering']
    model_cfg  = cfg['model']['modules']['uresnet_clustering']

    tsne = TSNE(n_components = 2 if method_cfg is None else method_cfg.get('tsne_dim',2))
    compute_tsne = False if method_cfg is None else method_cfg.get('compute_tsne',False)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout_cluster,fout_metric=None,None
    if store_per_iteration:
        fout_cluster=CSVData(os.path.join(logdir, 'instance-clustering-iter-%07d.csv' % iteration))
        fout_metrics=CSVData(os.path.join(logdir, 'instance-clustering-metrics-iter-%07d.csv' % iteration))

    model_cfg = cfg['model']['modules']['uresnet_clustering']
    data_dim = model_cfg.get('data_dim', 3)
    depth = model_cfg.get('num_strides', 5)
    num_classes = model_cfg.get('num_classes', 5)

    # Loop over batch index
    for batch_index, event_data in enumerate(data_blob['input_data']):

        event_index = data_blob['index'][batch_index]

        if not store_per_iteration:
            fout_cluster=CSVData(os.path.join(logdir, 'instance-clustering-iter-%07d.csv' % event_index))
            fout_metrics=CSVData(os.path.join(logdir, 'instance-clustering-metrics-iter-%07d.csv' % event_index))

        event_segmentation = res['segmentation'][batch_index]
        event_label = data_blob['segment_label'][batch_index]
        event_cluster_label = data_blob['cluster_label'][batch_index]
        max_depth = len(event_cluster_label)
        for d, event_feature_map in enumerate(res['cluster_feature'][batch_index]):
            coords = event_feature_map[:, :data_dim]
            perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            coords = coords[perm]
            class_label = event_label[-(d+1+max_depth-depth)]
            cluster_count = 0
            for class_ in range(num_classes):
                class_index = class_label[:, -1] == class_
                if np.count_nonzero(class_index) == 0:
                    continue
                clusters_label = event_cluster_label[-(d+1+max_depth-depth)][class_index]
                embedding = event_feature_map[perm][class_index]
                # DBSCAN in high dimension embedding
                predicted_clusters = DBSCAN(eps=20, min_samples=1).fit(embedding).labels_
                predicted_clusters += cluster_count  # To avoid overlapping id
                cluster_count += len(np.unique(predicted_clusters))

                # Cluster similarity metrics
                ARI = metrics.adjusted_rand_score(clusters_label[:, -1], predicted_clusters)
                AMI = metrics.adjusted_mutual_info_score(clusters_label[:, -1], predicted_clusters)
                fout_metrics.record(('class', 'batch_id', 'AMI', 'ARI', 'idx'),
                                    (class_, batch_index, AMI, ARI, event_index))
                fout_metrics.write()

                for i, point in enumerate(clusters_label):
                    fout_cluster.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id', 'idx'),
                                        (1, point[0], point[1], point[2], batch_index, d, -1, class_label[class_index][i, -1], clusters_label[i, -1], predicted_clusters[i], event_index))
                    fout_cluster.write()
                # TSNE to visualize embedding
                if compute_tsne and embedding.shape[0] > 1:
                    new_embedding = tsne.fit_transform(embedding)
                    for i, point in enumerate(new_embedding):
                        fout_cluster.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id', 'idx'),
                                            (2, point[0], point[1], -1, batch_index, d, -1, class_label[class_index][i, -1], clusters_label[i, -1], predicted_clusters[i], event_index))
                        fout_cluster.write()

        # Record in CSV everything
        perm = np.lexsort((event_data[:, 2], event_data[:, 1], event_data[:, 0]))
        event_data = event_data[perm]
        event_segmentation = event_segmentation[perm]
        # Point in data and semantic class predictions/true information
        for i, point in enumerate(event_data):
            fout_cluster.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id', 'idx'),
                                (0, point[0], point[1], point[2], batch_index, point[4], np.argmax(event_segmentation[i]), event_label[0][:,-1][i], -1, -1, event_index))
            fout_cluster.write()

        if not store_per_iteration:
            fout_cluster.close()
            fout_metrics.close()
    if store_per_iteration:
        fout_cluster.close()
        fout_metrics.close()
