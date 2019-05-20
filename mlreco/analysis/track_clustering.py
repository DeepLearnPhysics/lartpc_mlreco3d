import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from mlreco.utils import utils


def nms_numpy(im_proposals, im_scores, threshold, size):
    dim = im_proposals.shape[-1]
    coords = []
    for d in range(dim):
        coords.append(im_proposals[:, d] - size)
    for d in range(dim):
        coords.append(im_proposals[:, d] + size)
    coords = np.array(coords)

    areas = np.ones_like(coords[0])
    areas = np.prod(coords[dim:] - coords[0:dim] + 1, axis=0)

    order = im_scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx = np.maximum(coords[:dim, i][:, np.newaxis], coords[:dim, order[1:]])
        yy = np.minimum(coords[dim:, i][:, np.newaxis], coords[dim:, order[1:]])
        w = np.maximum(0.0, yy - xx + 1)
        inter = np.prod(w, axis=0)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def track_clustering(data_blob, res, cfg, idx):
    # Create output CSV
    csv_logger = utils.CSVData("%s/track_clustering-%.07d.csv" % (cfg['training']['log_dir'], idx))

    model_cfg = cfg['model']
    clusters = res['clusters'][0]  # (N1, 6)
    points = res['points'][0]  # (N, 5)
    segmentation = res['segmentation'][0]  # (N, 5)
    # FIXME N1 >= N because some points might belong to several clusters?
    clusters_label = data_blob['clusters_label'][0][0]  # (N1, 5)
    particles_label = data_blob['particles_label'][0][0]  # (N_gt, 5)
    data = data_blob['input_data'][0][0]
    segmentation_label = data_blob['segment_label'][0][0]

    # print(points)

    data_dim = 3  # model_cfg['data_dim']
    batch_ids = np.unique(data[:, data_dim])
    # print(segmentation[: 10], batch_ids)
    score_threshold = 0.6
    threshold_association = 3
    exclusion_radius = 5
    for b in batch_ids:
        event_clusters = clusters[clusters[:, data_dim] == b]
        batch_index = points[:, data_dim] == b
        event_points = points[batch_index][:, :-2]
        event_scores = points[batch_index][:, -2:]
        event_data = data[:, :data_dim][batch_index]
        event_segmentation = segmentation[batch_index]
        event_clusters_label = clusters_label[clusters_label[:, data_dim] == b]
        event_particles_label = particles_label[particles_label[:, data_dim] == b]

        anchors = (event_data + 0.5)
        event_points = event_points + anchors

        if event_points.shape[0] > 0:
            score_index = event_scores[:, 1] > score_threshold
            event_points = event_points[score_index]
            event_scores = event_scores[score_index]
        # 0) DBScan predicted pixels
        # print(event_points[:10])
        if event_points.shape[0] > 0:
            db = DBSCAN(eps=1.0, min_samples=5).fit(event_points).labels_
            print(np.unique(db))
            dbscan_points = []
            for label in np.unique(db):
                dbscan_points.append(event_points[db == label].mean(axis=0))
            dbscan_points = np.stack(dbscan_points)
            print(dbscan_points.shape)

            # 1) Break algorithm
            print(len(event_clusters), np.unique(event_clusters[:, -1]))
            cluster_ids = np.unique(event_clusters[:, -1])
            final_clusters = []
            for c in cluster_ids:
                # print("Cluster ", c)
                cluster = event_clusters[event_clusters[:, -1] == c][:, :data_dim]
                d = cdist(dbscan_points, cluster)
                # print(d.shape)
                index = d.min(axis=1) < threshold_association
                cluster_points = dbscan_points[index]
                # print(cluster_points)
                new_d = d[index.reshape((-1,)), :]
                # print(new_d.shape)
                new_index = (new_d > exclusion_radius).all(axis=0)
                new_cluster = cluster[new_index]
                remaining_cluster = cluster[~new_index]
                # print(new_cluster.shape)
                db2 = DBSCAN(eps=1.0, min_samples=1).fit(new_cluster).labels_
                # print(db2)
                new_cluster_ids = np.unique(db2)
                new_clusters = []
                for c2 in new_cluster_ids:
                    new_clusters.append([new_cluster[db2 == c2]])
                d3 = cdist(remaining_cluster, new_cluster)
                remaining_db = db2[d3.argmin(axis=1)]
                for i, c in enumerate(remaining_cluster):
                    new_clusters[remaining_db[i]].append(c[None, :])
                for i in range(len(new_clusters)):
                    new_clusters[i] = np.concatenate(new_clusters[i], axis=0)
                # print(new_clusters)
                final_clusters.extend(new_clusters)
        else:
            final_clusters = []
            cluster_idx = np.unique(event_clusters[:, -1])
            for c in cluster_idx:
                final_clusters.append(event_clusters[event_clusters[:, -1] == c][:, :data_dim])
            # FIXME is this right?


        # 2) Compute cluster efficiency/purity
        # ie associate final clusters after breaking with true clusters
        label_cluster_ids = np.unique(event_clusters_label[:, -1])
        true_clusters = []
        for c in label_cluster_ids:
            true_clusters.append(event_clusters_label[event_clusters_label[:, -1] == c][:, :-2])

        # Match each predicted cluster to a true cluster
        matches = []
        overlaps = []
        for predicted_cluster in final_clusters:
            overlap = []
            for true_cluster in true_clusters:
                overlap_pixel_count = np.count_nonzero((cdist(predicted_cluster, true_cluster)<1).any(axis=0))
                overlap.append(overlap_pixel_count)
            overlap = np.array(overlap)
            if overlap.max() > 0:
                matches.append(overlap.argmax())
                overlaps.append(overlap.max())
            else:
                matches.append(-1)
                overlaps.append(0)

        # Compute cluster purity/efficiency
        purity, efficiency = [], []
        npix_predicted, npix_true = [], []
        for i, predicted_cluster in enumerate(final_clusters):
            if matches[i] > -1:
                matched_cluster = true_clusters[matches[i]]
                purity.append(overlaps[i] / predicted_cluster.shape[0])
                efficiency.append(overlaps[i] / matched_cluster.shape[0])
                npix_predicted.append(predicted_cluster.shape[0])
                npix_true.append(matched_cluster.shape[0])

        print("Purity: ", purity)
        print("Efficiency: ", efficiency)
        print("Match indices: ", matches)
        print("Overlaps: ", overlaps)
        print("Npix predicted: ", npix_predicted)
        print("Npix true: ", npix_true)

        # Record in CSV everything
        for i, point in enumerate(data[batch_index]):
            csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'cluster_id', 'type'),
                              (0, point[0], point[1], point[2], point[3], point[4], np.argmax(event_segmentation[i]), segmentation_label[segmentation_label[:, data_dim] == b][i, -1], -1, -1 ))
            csv_logger.write()
        for c, cluster in enumerate(final_clusters):
            for point in cluster:
                csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'type'),
                                  (1, point[0], point[1], point[2], b, c, -1, -1, -1, -1))
                csv_logger.write()
        for c, cluster in enumerate(true_clusters):
            for point in cluster:
                csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'type'),
                                  (2, point[0], point[1], point[2], b, c, -1, -1, -1, -1))
        for point in event_points:
            csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'type'),
                              (3, point[0], point[1], point[2], b, -1, -1, -1, -1, -1))
            csv_logger.write()
        for point in event_clusters_label:
            csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'type'),
                              (4, point[0], point[1], point[2], b, point[4], -1, -1, -1, -1))
            csv_logger.write()
        for point in event_particles_label:
            csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'type', 'cluster_id', 'value', 'predicted_class', 'true_class'),
                              (5, point[0], point[1], point[2], b, point[4], -1, -1, -1, -1))
            csv_logger.write()
        # for point in segmentation:
        #     csv_logger.record(('point_type', 'x', 'y', 'z', 'batch_id', 'class_id'),
        #                       (6, point[0], point[1], point[2], b, np.argmax()))
        #     csv_logger.write()
        csv_logger.close()
