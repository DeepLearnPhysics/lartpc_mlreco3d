from mlreco.utils import CSVData
import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import scipy
from mlreco.utils.dbscan import dbscan_types, dbscan_points
from sklearn.metrics import adjusted_rand_score
from mlreco.utils.track_clustering import track_clustering as clustering
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


def track_clustering(cfg, data_blob, res, logdir, iteration):
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
    method_cfg = cfg['post_processing']['track_clustering']
    dbscan_cfg  = cfg['model']['modules']['dbscan']
    data_dim = int(dbscan_cfg['data_dim'])
    min_samples = int(dbscan_cfg['minPoints'])

    debug                 = bool(method_cfg.get('debug',None))
    score_threshold       = float(method_cfg.get('score_threshold',0.6))
    # To associate PPN predicted points and clusters
    # threshold_association = float(method_cfg.get('threshold_association',3))
    # To mask around PPN points
    exclusion_radius      = float(method_cfg.get('exclusion_radius',5))
    type_threshold        = float(method_cfg.get('type_threshold',2))
    record_voxels         = bool(method_cfg.get('record_voxels', False))
    clustering_method     = str(method_cfg.get('clustering_method', 'masked_dbscan'))
    tracks_only           = bool(method_cfg.get('tracks_only', True))
    tracks_mixed          = bool(method_cfg.get('tracks_mixed', False))
    eps                   = float(method_cfg.get('eps', 1.9))

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        if record_voxels:
            fout=CSVData(os.path.join(logdir, 'track-clustering-iter-%07d.csv' % iteration))
        fout2=CSVData(os.path.join(logdir, 'track-clustering2-iter-%07d.csv' % iteration))
        fout3=CSVData(os.path.join(logdir, 'track-clustering3-iter-%07d.csv' % iteration))

    # Loop over batch index
    #for b in batch_ids:
    for batch_index, data in enumerate(data_blob['input_data']):

        if not store_per_iteration:
            if record_voxels:
                fout=CSVData(os.path.join(logdir, 'track-clustering-event-%07d.csv' % event_index))
            fout2=CSVData(os.path.join(logdir, 'track-clustering2-event-%07d.csv' % event_index))
            fout2=CSVData(os.path.join(logdir, 'track-clustering3-iter-%07d.csv' % iteration))

        #event_clusters = res['final'][batch_index]

        event_index    = data_blob['index'][batch_index]
        event_data     = data[:,:data_dim]
        event_clusters_label  = data_blob['clusters_label'][batch_index]
        event_particles_label = data_blob['particles_label'][batch_index]
        event_particles_asis  = data_blob['particles'][batch_index]
        event_segmentation    = res['segmentation'][batch_index]
        event_segmentation_label = data_blob['segment_label'][batch_index]
        points         = res['points'][batch_index]
        event_xyz      = points[:, :data_dim]
        event_scores   = points[:, data_dim:data_dim+2]
        event_mask     = res['mask_ppn2'][batch_index]
        #print(event_clusters_label.shape, event_segmentation.shape)
        anchors = (event_data + 0.5)
        event_xyz = event_xyz + anchors
        num_classes = event_segmentation.shape[1]
        print("\n", batch_index, event_index)

        # 0) Postprocessing on predicted pixels
        ppn = uresnet_ppn_type_point_selector(data, res, entry=batch_index, type_threshold=2, score_threshold=0.5)
        #predicted_points = np.stack(ppn[0], axis=0)
        coords_points = ppn[:, :data_dim]
        #print(ppn[ppn[:, 0] == 570.8106])

        # 0.5) Remove points for delta rays
        # point_types = np.argmax(predicted_points[:, -5:], axis=1)
        coords_points = coords_points[ppn[:, -3]<0.5]
        ppn = ppn[ppn[:, -3]<0.5]

        # -1) Make event preliminary clusters per semantic class
        #print("label", event_clusters_label[:5])
        predictions = event_segmentation.argmax(axis=1)
        event_clusters = []
        clusters_count = 0
        predicted_cluster_labels = -1 * np.ones((event_segmentation.shape[0],))

        def process(mask, clusters_count, c=0):
            if np.count_nonzero(mask) == 0:
                return clusters_count
            if c == 0 or c == 1:
                clusters = clustering(event_data[mask][:, :data_dim], coords_points, method=clustering_method, eps=eps, min_samples=5, mask_radius=exclusion_radius)
            else: # pure dbscan
                if tracks_only:
                    return clusters_count
                else:
                    clusters = DBSCAN(eps=1.9, min_samples=5).fit(event_data[mask][:, :data_dim]).labels_
            masked_predicted_cluster_labels = predicted_cluster_labels[mask]
            masked_predicted_cluster_labels[clusters>-1] = clusters[clusters>-1] + clusters_count
            predicted_cluster_labels[mask] = masked_predicted_cluster_labels
            #print(np.unique(predicted_cluster_labels[mask][clusters>-1]))
            clusters_count += clusters.max() + 1
            if np.count_nonzero(clusters > -1) == 0:
                return clusters_count
            ones = np.ones((event_data[mask][clusters > -1].shape[0], 1))
            predicted_class = ones * c
            true_class = ones * np.bincount(event_segmentation_label[mask][clusters > -1][:, -1].astype(int)).argmax()
            #print(c, clusters[clusters>-1][:, None] + clusters_count)
            event_clusters.append(np.concatenate([event_data[mask, :data_dim][clusters > -1], predicted_class, true_class, clusters[clusters>-1][:, None] + clusters_count], axis=1))
            clusters_count += clusters.max()+1 #len(np.unique(clusters[clusters>-1]))
            return clusters_count

        if tracks_mixed:
            clusters_count = process((predictions == 0) | (predictions == 1), clusters_count)
        else:
            for c in np.unique(predictions):
                mask = predictions == c
                clusters_count = process(mask, clusters_count, c=c)

        event_clusters = np.concatenate(event_clusters, axis=0)
        final_clusters = []
        for cluster_id in np.unique(event_clusters[:, -1]):
            final_clusters.append(event_clusters[event_clusters[:, -1] == cluster_id])

        #print(len(final_clusters))
        #print(np.unique(np.concatenate(event_clusters, axis=0)[:, -1]))

        # Compute ARI per class
        # Remove particles with num_voxels == 0
        #event_particles_asis = np.array(event_particles_asis)[np.array([p.num_voxels() > 0 for p in event_particles_asis])]
        # Now event_particles_asis and event_clusters_label are aligned, find semantics
        event_clusters_label_semantics = -1 * np.ones((event_clusters_label.shape[0],))
        pid = 0
        for idx, p in enumerate(event_particles_asis):
            #print("pdg ", p.pdg_code())
            if p.num_voxels() == 0:
                continue
            gt_type = -1
            if p.pdg_code() == 2212:
                gt_type = 0
            elif p.pdg_code() != 22 and p.pdg_code() != 11:
                gt_type = 1
            else:
                if tracks_only:
                    continue
                else:
                    if p.pdg_code() == 22:
                        gt_type = 2
                    else:
                        prc = p.creation_process()
                        if prc == "primary" or prc == "nCapture" or prc == "conv":
                            gt_type = 2 # em shower
                        elif prc == "muIoni" or prc == "hIoni":
                            gt_type = 3 # delta
                        elif prc == "muMinusCaptureAtRest" or prc == "muPlusCaptureAtRest" or prc == "Decay":
                            gt_type = 4 # michel
            event_clusters_label_semantics[event_clusters_label[:, -1] == idx] = gt_type
            #print(p.pdg_code(), gt_type, np.unique(event_clusters_label_semantics[event_clusters_label[:, -1] == pid]), np.count_nonzero(event_clusters_label[:, -1] == pid))
            pid += 1
            #print(gt_type, np.count_nonzero(event_clusters_label[:, -1] == idx))

        aris = []
        event_clusters_label2 = -1 * np.ones((predicted_cluster_labels.shape[0], event_clusters_label.shape[1]))
        def process_ari(mask, mask2, c=0):
            if np.count_nonzero(mask) == 0 or np.count_nonzero(mask2) == 0:
                return
            # Find true cluster ids of this semantic class
            #print(np.unique(event_clusters_label[:, :data_dim], axis=0).shape, event_segmentation.shape)
            #print(event_segmentation[:10],  np.unique(event_clusters_label[:, :data_dim], axis=0)[:10])
            class_clusters_label = event_clusters_label[mask2]
            print("is 12 here", np.count_nonzero(class_clusters_label[:, -1] == 12))
            #print(class_clusters_label.shape, event_segmentation_label[mask].shape, np.unique(class_clusters_label[:, -1], return_counts=True))
            d = cdist(event_segmentation_label[mask][:, :data_dim], class_clusters_label[:, :data_dim])
            keep_columns = np.where((d<1).any(axis=0))[0]
            #true_cluster_labels = class_clusters_label[keep_columns][d[:, keep_columns].argmin(axis=1)][:, -1]
            true_cluster_labels = class_clusters_label[d.argmin(axis=1)][:, -1]
            print("is 12 here", np.count_nonzero(true_cluster_labels == 12))
            #print("True", np.unique(true_cluster_labels, return_counts = True))
            #print("Pred", np.unique(predicted_cluster_labels[mask], return_counts=True))
            #mask_outliers = predicted_cluster_labels[mask] > -1
            event_clusters_label2[mask] = class_clusters_label[d.argmin(axis=1)]
            print("is 12 here", np.count_nonzero(event_clusters_label2[mask][:, -1] == 12))
            ari = adjusted_rand_score(true_cluster_labels, predicted_cluster_labels[mask])
            #print("ari", np.count_nonzero(mask), ari)
            fout3.record(('batch_id', 'idx', 'class', 'ari', 'num_voxels', 'num_true_clusters', 'num_pred_cluster'),
                        (batch_index, event_index, c, ari, np.count_nonzero(mask), len(np.unique(true_cluster_labels)), len(np.unique(predicted_cluster_labels[mask]))))
            fout3.write()
            aris.append(ari)

        #print("event_clusters_label", np.unique(event_clusters_label[:, -1], return_counts=True), np.unique(event_clusters_label_semantics, return_counts=True))
        if tracks_mixed:
            process_ari((event_segmentation_label[:, -1] == 0) | (event_segmentation_label[:, -1] == 1),
                        (event_clusters_label_semantics == 0) | (event_clusters_label_semantics == 1))
        else:
            for c in range(num_classes):
                if tracks_only & (c > 1):
                    continue
                mask = event_segmentation_label[:, -1] == c
                process_ari(mask, (event_clusters_label_semantics == c), c=c)

        #ari_per_class = []
        #ari_per_class.append(adjusted_rand_score(event_clusters_label[:, -1]))

        # 2) Compute cluster efficiency/purity
        # ie associate final clusters after breaking with true clusters
        #print([(c, np.count_nonzero(event_clusters_label[:, -1] == c), np.unique(event_clusters_label_semantics[event_clusters_label[:, -1] == c])) for c in np.unique(event_clusters_label[:, -1])])
        event_clusters_label = event_clusters_label2
        label_cluster_ids = np.unique(event_clusters_label[:, -1])
        true_clusters = []
        true_class = []
        for c in label_cluster_ids:
            # print(label_cluster_ids.max())
            # if c == label_cluster_ids.max():
            #     continue
            true_clusters.append(event_clusters_label[event_clusters_label[:, -1] == c][:, :data_dim])
            true_class.append(event_segmentation_label[cdist([true_clusters[-1][0]], event_segmentation_label[:, :data_dim]).argmin(axis=1)[0], -1])
        #print(len(true_clusters))
        #print([cluster.shape[0] for cluster in true_clusters])
        # Match each predicted cluster to a true cluster
        matches = []
        overlaps = []
        matches_id = []
        for predicted_cluster in final_clusters:
            overlap = []
            for true_cluster in true_clusters:
                overlap_pixel_count = np.count_nonzero((cdist(predicted_cluster[:, :data_dim], true_cluster)<1).any(axis=0))
                overlap.append(overlap_pixel_count)
            overlap = np.array(overlap)
            #print(predicted_cluster[0, -1], predicted_cluster.shape[0], overlap)
            if overlap.max() > 0: # Found a match among true clusters
                matched_true_cluster = true_clusters[overlap.argmax()]
                matches.append(overlap.argmax())
                matches_id.append(int(matched_true_cluster[0, -1]))
                overlaps.append(overlap.max())
            else: # No matching true cluster
                matches.append(-1)
                matches_id.append(-1)
                overlaps.append(0)
        matches = np.array(matches)
        matches_id = np.array(matches_id)
        overlaps = np.array(overlaps)

        if debug:
            print("Purity: ", purity)
            print("Efficiency: ", efficiency)
            print("Match indices: ", matches)
            print("Overlaps: ", overlaps)
            print("Npix predicted: ", npix_predicted)
            print("Npix true: ", npix_true)

        # Record in CSV everything
        if record_voxels:
            # Point in data and semantic class predictions/true information
            for i, point in enumerate(data):
                fout.record(('type', 'x', 'y', 'z', 'batch_id', 'value', 'predicted_class', 'true_class', 'cluster_id', 'point_type', 'idx', 'match', 'overlap'),
                            (3, point[0], point[1], point[2], batch_index, point[4], np.argmax(event_segmentation[i]), event_segmentation_label[i, -1], -1, -1, event_index, -1, -1))
                fout.write()
            # Predicted clusters
            for c, cluster in enumerate(final_clusters):
                for point in cluster:
                    fout.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type', 'idx', 'match', 'overlap'),
                                (4, point[0], point[1], point[2], batch_index, c, -1, cluster[0, -3], cluster[0, -2], -1, event_index, matches[c], overlaps[c]))
                    fout.write()
            # True clusters
            for c, cluster in enumerate(true_clusters):
                for point in cluster:
                    fout.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type', 'idx', 'match', 'overlap'),
                                      (2, point[0], point[1], point[2], batch_index, c, -1, -1, true_class[c], -1, event_index, -1, -1))
                    fout.write()
            # for point in event_xyz:
            #     fout.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type'),
            #                       (3, point[0], point[1], point[2], batch_index, -1, -1, -1, -1, -1))
            #     fout.write()
            # for point in event_clusters_label:
            #     fout.record(('type', 'x', 'y', 'z', 'batch_id', 'cluster_id', 'value', 'predicted_class', 'true_class', 'point_type'),
            #                       (4, point[0], point[1], point[2], batch_index, point[4], -1, -1, -1, -1))
            #     fout.write()
            # True PPN points
            for point in event_particles_label:
                fout.record(('type', 'x', 'y', 'z', 'batch_id', 'point_type', 'cluster_id', 'value', 'predicted_class', 'true_class', 'idx', 'match', 'overlap'),
                                  (5, point[0], point[1], point[2], batch_index, point[4], -1, -1, -1, -1, event_index, -1, -1))
                fout.write()
            # Predicted PPN points
            for point in ppn:
                fout.record(('type', 'x', 'y', 'z', 'batch_id', 'predicted_class', 'value', 'true_class', 'cluster_id', 'point_type', 'idx', 'match', 'overlap'),
                                  (6, point[0], point[1], point[2], batch_index, point[-1], -1, -1, -1, -1, event_index, -1, -1))
                fout.write()

        # Record in any case cluster-wise information
        # for c, cluster in enumerate(final_clusters):
        #     fout2.record(('type', 'num_voxels', 'batch_id', 'idx', 'predicted_class', 'true_class', 'cluster_id', 'match', 'overlap'),
        #                 (0, cluster.shape[0], batch_index, event_index, cluster[0, -3], cluster[0, -2], cluster[0, -1], matches_id[c], overlaps[c]))
        #     fout2.write()
        # for c, cluster in enumerate(true_clusters):
        #     num_matches = np.count_nonzero(matches == cluster[0, -1])
        #     fout2.record(('type', 'num_voxels', 'batch_id', 'idx', 'predicted_class', 'true_class', 'cluster_id', 'match', 'overlap'),
        #                 (1, cluster.shape[0], batch_index, event_index, -1, true_class[c], cluster[0, -1], num_matches, -1 if num_matches == 0 else overlaps[matches == cluster[0, -1]].max()))
        #     fout2.write()

        # Compute cluster purity/efficiency
        purity, efficiency = [], []
        npix_predicted, npix_true = [], []
        class_predicted, class_true = [], []
        for i, predicted_cluster in enumerate(final_clusters):
            purity, efficiency = -1, -1
            npix_predicted, npix_true = -1, -1
            class_predicted, class_true = -1, -1
            overlap = -1
            if matches[i] > -1:
                matched_cluster = true_clusters[matches[i]]
                purity = overlaps[i] / predicted_cluster.shape[0]
                efficiency = overlaps[i] / matched_cluster.shape[0]
                npix_predicted = predicted_cluster.shape[0]
                npix_true = matched_cluster.shape[0]
                overlap = overlaps[i]
            fout2.record(('num_voxels_pred', 'num_voxels_true', 'batch_id', 'idx', 'predicted_class', 'true_class', 'cluster_id', 'overlap', 'purity', 'efficiency'),
                        (predicted_cluster.shape[0], npix_true, batch_index, event_index, predicted_cluster[0, -3], predicted_cluster[0, -2], predicted_cluster[0, -1], overlap, purity, efficiency))
            fout2.write()

        if not store_per_iteration:
            if record_voxels:
                fout.close()
            fout2.close()
            fout3.close()

    if store_per_iteration:
        if record_voxels:
            fout.close()
        fout2.close()
        fout3.close()
