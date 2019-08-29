import numpy as np
from mlreco.utils import utils
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn import metrics


def instance_clustering(data_blob, res, cfg, idx, compute_tsne=False):
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
    csv_logger = utils.CSVData("%s/instance_clustering-%.07d.csv" % (cfg['trainval']['log_dir'], idx))
    csv_logger2 = utils.CSVData("%s/instance_clustering_metrics-%.07d.csv" % (cfg['trainval']['log_dir'], idx))

    model_cfg = cfg['model']['modules']['uresnet_clustering']
    segmentation_all = res['segmentation'][0]  # (N, 5)
    clustering_all = res['clustering'][0]

    data_all = data_blob['input_data'][0][0]
    idx_all = data_blob['index'][0][0]
    label_all = data_blob['segment_label'][0][0]
    clusters_label_all = data_blob['cluster_label'][0][0]

    data_dim = model_cfg.get('data_dim', 3)
    batch_ids = np.unique(data_all[:, data_dim])
    depth = model_cfg.get('num_strides', 5)
    max_depth = len(clusters_label_all)
    num_classes = model_cfg.get('num_classes', 5)
    tsne = TSNE(n_components=2)
    # Loop over batch index
    for b in batch_ids:
        batch_index = data_all[:, data_dim] == b
        event_index = idx_all[int(b)][0]
        event_data = data_all[batch_index]
        event_segmentation = segmentation_all[batch_index]
        event_label = label_all[0][batch_index][:, -1]

        for d, feature_map in enumerate(clustering_all):
            event_feature_map = feature_map[feature_map[:, data_dim] == b]
            coords = event_feature_map[:, :data_dim]
            perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            coords = coords[perm]
            class_label = label_all[-(d+1+max_depth-depth)][label_all[-(d+1+max_depth-depth)][:, -2] == b]
            cluster_count = 0
            for class_ in range(num_classes):
                class_index = class_label[:, -1] == class_
                if np.count_nonzero(class_index) == 0:
                    continue
                clusters_label = clusters_label_all[-(d+1+max_depth-depth)][class_index]
                embedding = event_feature_map[perm][class_index]
                # DBSCAN in high dimension embedding
                predicted_clusters = DBSCAN(eps=20, min_samples=1).fit(embedding).labels_
                predicted_clusters += cluster_count  # To avoid overlapping id
                cluster_count += len(np.unique(predicted_clusters))

                # Cluster similarity metrics
                ARI = metrics.adjusted_rand_score(clusters_label[:, -1], predicted_clusters)
                AMI = metrics.adjusted_mutual_info_score(clusters_label[:, -1], predicted_clusters)
                csv_logger2.record(('class', 'batch_id', 'AMI', 'ARI', 'idx'),
                                   (class_, b, AMI, ARI, event_index))
                csv_logger2.write()

                for i, point in enumerate(clusters_label):
                    csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id', 'idx'),
                                      (1, point[0], point[1], point[2], point[3], d, -1, class_label[class_index][i, -1], clusters_label[i, -1], predicted_clusters[i], event_index))
                    csv_logger.write()
                # TSNE to visualize embedding
                if compute_tsne and embedding.shape[0] > 1:
                    new_embedding = tsne.fit_transform(embedding)
                    for i, point in enumerate(new_embedding):
                        csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id', 'idx'),
                                          (2, point[0], point[1], -1, clusters_label[i, 3], d, -1, class_label[class_index][i, -1], clusters_label[i, -1], predicted_clusters[i], event_index))
                        csv_logger.write()

        # Record in CSV everything
        perm = np.lexsort((event_data[:, 2], event_data[:, 1], event_data[:, 0]))
        event_data = event_data[perm]
        event_segmentation = event_segmentation[perm]
        # Point in data and semantic class predictions/true information
        for i, point in enumerate(event_data):
            csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id', 'idx'),
                              (0, point[0], point[1], point[2], point[3], point[4], np.argmax(event_segmentation[i]), event_label[i], -1, -1, event_index))
            csv_logger.write()

    csv_logger.close()
    csv_logger2.close()
