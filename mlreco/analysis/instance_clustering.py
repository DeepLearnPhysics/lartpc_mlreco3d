import numpy as np
from mlreco.utils import utils
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


def instance_clustering(data_blob, res, cfg, idx):
    """
    Simple thresholding on uresnet clustering output for instance segmentation
    """
    csv_logger = utils.CSVData("%s/instance_clustering-%.07d.csv" % (cfg['training']['log_dir'], idx))

    model_cfg = cfg['model']

    segmentation_all = res['segmentation'][0]  # (N, 5)
    # predictions_all = np.argmax(segmentation_all, axis=1)
    # encoding_all = res['encoding'][0]  # len = depth + 1
    decoding_all = res['decoding'][0]  # len = depth

    data_all = data_blob['input_data'][0][0]
    label_all = data_blob['segment_label'][0][0]
    clusters_label_all = data_blob['cluster_label'][0][0]

    data_dim = 3  # model_cfg['data_dim']
    batch_ids = np.unique(data_all[:, data_dim])
    depth = 5
    max_depth = len(clusters_label_all)
    num_classes = 5
    # Loop over batch index
    for b in batch_ids:
        batch_index = data_all[:, data_dim] == b
        event_data = data_all[batch_index]
        event_segmentation = segmentation_all[batch_index]
        event_label = label_all[0][batch_index][:, -1]

        for d, feature_map in enumerate(decoding_all):
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
                embedding = event_feature_map[class_index]
                # DBSCAN in high dimension embedding
                predicted_clusters = DBSCAN(eps=5, min_samples=1).fit(embedding).labels_
                predicted_clusters += cluster_count  # To avoid overlapping id
                cluster_count += len(np.unique(predicted_clusters))
                for i, point in enumerate(clusters_label):
                    csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id'),
                                      (1, point[0], point[1], point[2], point[3], d, -1, class_label[class_index][i, -1], clusters_label[i, -1], predicted_clusters[i]))
                    csv_logger.write()
                # TSNE to visualize embedding
                print('Embedding size:', embedding.shape[1])
                if embedding.shape[0] > 1:
                    print(d, class_, 'Starting TSNE')
                    new_embedding = TSNE(n_components=2).fit_transform(embedding)
                    for i, point in enumerate(new_embedding):
                        csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id'),
                                          (2, point[0], point[1], -1, clusters_label[i, 3], d, -1, class_label[class_index][i, -1], clusters_label[i, -1], predicted_clusters[i]))
                        csv_logger.write()
                    print('Done')

        # Record in CSV everything
        perm = np.lexsort((event_data[:, 2], event_data[:, 1], event_data[:, 0]))
        event_data = event_data[perm]
        event_segmentation = event_segmentation[perm]
        # Point in data and semantic class predictions/true information
        for i, point in enumerate(event_data):
            csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'true_cluster_id', 'predicted_cluster_id'),
                              (0, point[0], point[1], point[2], point[3], point[4], np.argmax(event_segmentation[i]), event_label[i], -1, -1))
            csv_logger.write()

    csv_logger.close()
