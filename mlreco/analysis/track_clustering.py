import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from mlreco.utils import utils
import scipy


def track_clustering(data_blob, res, cfg, idx, debug=False):
    """
    Track clustering on PPN+UResNet output.

    Parameters
    ----------
    data_blob: dict
        The input data dictionary from iotools.
    res: dict
        The output of the network, formatted using `analysis_keys`.
    cfg: dict
        Configuration.
    debug: bool, optional
        Whether to print some stats or not in the stdout.

    Notes
    -----
    Based on
    - semantic segmentation output
    - point position and type predictions
    In addition, includes a break algorithm to break track clusters into
    smaller clusters based on predicted points.
    Stores all points and informations in a CSV file.
    """
    # Create output CSV
    csv_logger = utils.CSVData("%s/track_clustering-%.07d.csv" % (cfg['trainval']['log_dir'], idx))

    model_cfg = cfg['model']
    clusters = res['clusters'][0]  # (N1, 7) from dbscan
    points = res['points'][0]  # (N, 5+c) ppn predictions 3 coords + 2 scores + classes scores for point type prediction
    segmentation = res['segmentation'][0]  # (N, 5) uresnet predictions
    mask = res['mask'][0]  # (N, 2)
    # FIXME N1 >= N because some points might belong to several clusters?
    # N1 == N if we use `parse_cluster3d_clean` instead of `parse_cluster3d`
    clusters_label = data_blob['clusters_label'][0][0]  # (N1, 5)
    particles_label = data_blob['particles_label'][0][0]  # (N_gt, 5)
    data = data_blob['input_data'][0][0]  # shape (N, 5)
    idx = data_blob['index'][0][0]
    segmentation_label = data_blob['segment_label'][0][0]  # shape (N, 5)

    data_dim = 3  # model_cfg['data_dim']
    batch_ids = np.unique(data[:, data_dim])
    score_threshold = 0.6
    threshold_association = 3
    exclusion_radius = 5
    type_threshold = 2
    # Loop over batch index
    for b in batch_ids:
        event_clusters = clusters[clusters[:, data_dim] == b]
        batch_index = data[:, data_dim] == b
        event_index = idx[int(b)][0]
        event_points = points[batch_index][:, :data_dim]
        event_scores = points[batch_index][:, data_dim:data_dim+2]
        event_data = data[:, :data_dim][batch_index]
        event_segmentation = segmentation[batch_index]
        event_clusters_label = clusters_label[clusters_label[:, data_dim] == b]
        event_particles_label = particles_label[particles_label[:, data_dim] == b]
        event_mask = mask[batch_index]
        anchors = (event_data + 0.5)
        event_points = event_points + anchors

        dbscan_points = []
        predicted_points = []

        # 0) Postprocessing on predicted pixels
        # Apply selection mask from PPN2 + score thresholding
        scores = scipy.special.softmax(event_scores, axis=1)
        event_mask = ((~(event_mask == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
        # Now loop through semantic classes and look at ppn+uresnet predictions
        uresnet_predictions = np.argmax(event_segmentation[event_mask], axis=1)
        num_classes = event_segmentation.shape[1]
        ppn_type_predictions = np.argmax(scipy.special.softmax(points[batch_index][event_mask][:, 5:], axis=1), axis=1)
        for c in range(num_classes):
            uresnet_points = uresnet_predictions == c
            ppn_points = ppn_type_predictions == c
            # We want to keep only points of type X within 2px of uresnet prediction of type X
            d = scipy.spatial.distance.cdist(event_points[event_mask][ppn_points], event_data[event_mask][uresnet_points])
            ppn_mask = (d < type_threshold).any(axis=1)
            # dbscan_points stores coordinates only
            # predicted_points stores everything for each point
            dbscan_points.append(event_points[event_mask][ppn_points][ppn_mask])
            pp = points[batch_index][event_mask][ppn_points][ppn_mask]
            pp[:, :3] += anchors[event_mask][ppn_points][ppn_mask]
            predicted_points.append(pp)
        dbscan_points = np.concatenate(dbscan_points, axis=0)
        predicted_points = np.concatenate(predicted_points, axis=0)

        # 0.5) Remove points for delta rays
        point_types = np.argmax(predicted_points[:, -5:], axis=1)
        dbscan_points = dbscan_points[point_types != 3]

        # 1) Break algorithm
        # Using PPN point predictions, the idea is to mask an area around
        # each point associated with a given predicted cluster. Dbscan
        # then tells us whether this predicted cluster should be broken in
        # two or more smaller clusters. Pixel that were masked are then
        # assigned to the closest cluster among the newly formed clusters.
        if dbscan_points.shape[0] > 0:  # If PPN predicted some points
            cluster_ids = np.unique(event_clusters[:, -1])
            final_clusters = []
            # Loop over predicted clusters
            for c in cluster_ids:
                # Find predicted points associated to this predicted cluster
                cluster = event_clusters[event_clusters[:, -1] == c][:, :data_dim]
                d = cdist(dbscan_points, cluster)
                index = d.min(axis=1) < threshold_association
                new_d = d[index.reshape((-1,)), :]
                # Now mask around these points
                new_index = (new_d > exclusion_radius).all(axis=0)
                # Main body of the cluster (far way from the points)
                new_cluster = cluster[new_index]
                # Cluster part around the points
                remaining_cluster = cluster[~new_index]
                # FIXME this might eliminate too small clusters?
                # put a threshold here? sometimes clusters with 1px only
                if new_cluster.shape[0] == 0:
                    continue
                # Now dbscan on the main body of the cluster to find if we need
                # to break it or not
                db2 = DBSCAN(eps=exclusion_radius, min_samples=cfg['model']['modules']['dbscan']['minPoints']).fit(new_cluster).labels_
                # All points were garbage
                if (len(new_cluster[db2 == -1]) == len(new_cluster)):
                    continue
                # These are going to be the new bodies of predicted clusters
                new_cluster_ids = np.unique(db2)
                new_clusters = []
                for c2 in new_cluster_ids:
                    if c2 > -1:
                        new_clusters.append([new_cluster[db2 == c2]])
                # If some points were left by dbscan, put them in remaining
                # cluster and assign them to closest cluster
                if len(new_cluster[db2 == -1]) > 0:
                    print(len(new_cluster[db2 == -1]), len(new_cluster))
                    remaining_cluster = np.concatenate([remaining_cluster, new_cluster[db2 == -1]], axis=0)
                    # effectively remove them from new_cluster for the argmin
                    new_cluster[db2 == -1] = 100000
                # Now assign remaining pixels in remaining_cluster based on
                # their distance to the new clusters.
                # First we find which point of new_cluster was closest
                d3 = cdist(remaining_cluster, new_cluster)
                # Then we find what is the corresponding new cluster id of this
                # closest pixel
                remaining_db = db2[d3.argmin(axis=1)]
                # Now append each pixel of remaining_cluster to correct new
                # cluster
                for i, c in enumerate(remaining_cluster):
                    new_clusters[remaining_db[i]].append(c[None, :])
                # Turn everything into np arrays
                for i in range(len(new_clusters)):
                    new_clusters[i] = np.concatenate(new_clusters[i], axis=0)
                final_clusters.extend(new_clusters)
        else:  # no predicted points: no need to break, keep predicted clusters
            final_clusters = []
            cluster_idx = np.unique(event_clusters[:, -1])
            for c in cluster_idx:
                final_clusters.append(event_clusters[event_clusters[:, -1] == c][:, :data_dim])

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

        if debug:
            print("Purity: ", purity)
            print("Efficiency: ", efficiency)
            print("Match indices: ", matches)
            print("Overlaps: ", overlaps)
            print("Npix predicted: ", npix_predicted)
            print("Npix true: ", npix_true)

        # Record in CSV everything
        # Point in data and semantic class predictions/true information
        for i, point in enumerate(data[batch_index]):
            csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'cluster_id', 'point_type', 'idx'),
                              (0, point[0], point[1], point[2], point[3], point[4], np.argmax(event_segmentation[i]), segmentation_label[segmentation_label[:, data_dim] == b][i, -1], -1, -1, event_index))
            csv_logger.write()
        # Predicted clusters
        for c, cluster in enumerate(final_clusters):
            for point in cluster:
                csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type', 'idx'),
                                  (1, point[0], point[1], point[2], b, c, -1, -1, -1, -1, event_index))
                csv_logger.write()
        # True clusters
        for c, cluster in enumerate(true_clusters):
            for point in cluster:
                csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type', 'idx'),
                                  (2, point[0], point[1], point[2], b, c, -1, -1, -1, -1, event_index))
                csv_logger.write()
        # for point in event_points:
        #     csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type'),
        #                       (3, point[0], point[1], point[2], b, -1, -1, -1, -1, -1))
        #     csv_logger.write()
        # for point in event_clusters_label:
        #     csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type'),
        #                       (4, point[0], point[1], point[2], b, point[4], -1, -1, -1, -1))
        #     csv_logger.write()
        # True PPN points
        for point in event_particles_label:
            csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'point_type', 'cluster_id', 'value', 'predicted_class', 'true_class', 'idx'),
                              (5, point[0], point[1], point[2], b, point[4], -1, -1, -1, -1, event_index))
            csv_logger.write()
        # Predicted PPN points
        for point in predicted_points:
            csv_logger.record(('type', 'x', 'y', 'z', 'batch_id', 'predicted_class', 'value', 'true_class', 'cluster_id', 'point_type', 'idx'),
                              (6, point[0], point[1], point[2], b, np.argmax(point[-5:]), -1, -1, -1, -1, event_index))
            csv_logger.write()
        csv_logger.close()
