import numpy as np
import os
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from mlreco.utils import CSVData
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels


def michel_reconstruction(cfg, module_cfg, data_blob, res, logdir, iteration):
    """
    Very simple algorithm to reconstruct Michel clusters from UResNet semantic
    segmentation output.

    Notes
    -----
    Assumes 3D

    Configuration
    -------------
    ```
      michel_reconstruction:
        store_method: per-iteration
        dbscan: False
        particles: particles_asis
    ```

    Output
    ------
    Writes 2 CSV files:
    - `michel_reconstruction-reco-*`
    - `michel_reconstruction-true-*`
    """
    coords_col = module_cfg.get('coords_col', (1, 4))
    particles_label = module_cfg.get('particles', 'particles_label')
    cluster_label   = module_cfg.get('cluster_label', 'cluster_label')
    segment_label   = module_cfg.get('segment_label', 'segment_label')
    # should_adapt_labels    = module_cfg.get('adapt_labels', False)

    # Create output CSV
    store_per_iteration = True
    if module_cfg is not None and module_cfg.get('store_method',None) is not None:
        assert(module_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = module_cfg['store_method'] == 'per-iteration'

    fout_reco,fout_true=None,None
    if store_per_iteration:
        fout_reco=CSVData(os.path.join(logdir, 'michel-reconstruction-reco-iter-%07d.csv' % iteration))
        fout_true=CSVData(os.path.join(logdir, 'michel-reconstruction-true-iter-%07d.csv' % iteration))
        fout_matched=CSVData(os.path.join(logdir, 'michel-reconstruction-matched-iter-%07d.csv' % iteration))

    # Loop over events
    for batch_id,data in enumerate(data_blob['input_data']):

        event_idx = data_blob['index'          ][batch_id]

        if not store_per_iteration:
            fout_reco=CSVData(os.path.join(logdir, 'michel-reconstruction-reco-event-%07d.csv' % event_idx))
            fout_true=CSVData(os.path.join(logdir, 'michel-reconstruction-true-event-%07d.csv' % event_idx))
            fout_matched=CSVData(os.path.join(logdir, 'michel-reconstruction-matched-event-%07d.csv' % event_idx))

        # from input/labels
        label       = data_blob[segment_label][batch_id][:,-1]
        #label_raw   = data_blob['sparse3d_pcluster_semantics'][batch_id]
        clusters    = data_blob[cluster_label][batch_id]
        # if should_adapt_labels:
        #     clusters = adapt_labels(res, data_blob[segment_label], data_blob[cluster_label])[batch_id]
        particles   = data_blob[particles_label][batch_id]
        true_ghost_mask = label < 5
        data_masked     = data[true_ghost_mask]
        label_masked    = label[true_ghost_mask]

        one_pixel = 5#2.8284271247461903

        # Retrieve semantic labels corresponding to clusters
        clusters_semantics = clusters[:, -1]

        # from network output
        segmentation = res['segmentation'][batch_id]
        predictions  = np.argmax(segmentation,axis=1)
        ghost_mask   = (np.argmax(res['ghost'][batch_id],axis=1) == 0)

        data_pred    = data[ghost_mask]  # coords
        label_pred   = label[ghost_mask]  # labels
        # FIXME this is temporary because of 2-class deghosting only, assumes we do perfect 5-class segmentation
        # after deghosting
        #predictions  = label_pred
        predictions = (np.argmax(segmentation,axis=1))[ghost_mask]
        segmentation = segmentation[ghost_mask]

        Michel_label = 2
        MIP_label = 1

        # 0. Retrieve coordinates of true and predicted Michels
        # MIP_coords = data[(label == 1).reshape((-1,)), ...][:, :3]
        # Michel_coords = data[(label == 4).reshape((-1,)), ...][:, :3]
        # Michel_particles = particles[particles[:, 4] == Michel_label]
        MIP_coords = data[label == MIP_label][:, coords_col[0]:coords_col[1]]
        # Michel_coords = data[label == Michel_label][:, :3]
        Michel_all = clusters[clusters_semantics == Michel_label]
        if Michel_all.shape[0] == 0:  # FIXME
            continue
        #print(Michel_all.shape)
        if module_cfg.get('dbscan', False):
            shower_label = 0
            lowE_label = 4
            #Michel_coords = clusters[(clusters_semantics == Michel_label) | (clusters_semantics == shower_label) | (clusters_semantics == lowE_label)][:, :3]
            shower_lowE = clusters[(clusters_semantics == shower_label) | (clusters_semantics == lowE_label)]
            if shower_lowE.shape[0] > 0:
                shower_lowE_clusters = DBSCAN(eps=one_pixel, min_samples=1).fit(shower_lowE[:, coords_col[0]:coords_col[1]]).labels_

                d = cdist(Michel_all[:, coords_col[0]:coords_col[1]], shower_lowE[:, coords_col[0]:coords_col[1]])
                select_shower = (d.min(axis=0) < module_cfg.get('threshold', 2.))
                select_shower_clusters = np.unique(shower_lowE_clusters[(shower_lowE_clusters>-1) & select_shower])
                fragments = []
                for shower_id in select_shower_clusters:
                    fragment = shower_lowE[shower_lowE_clusters == shower_id]
                    print(batch_id, "merging", fragment.shape[0])
                    #print(fragment[:, -3][:20])
                    temp = fragment[:, -3]
                    print(temp, Michel_all[d[:, shower_lowE_clusters == shower_id].min(axis=1).argmin(), -3])
                    fragment[:, -3] = Michel_all[d[:, shower_lowE_clusters == shower_id].min(axis=1).argmin(), -3]
                    print(np.count_nonzero(temp != fragment[:, -3]))
                    fragments.append(fragment)
                Michel_all = np.concatenate([Michel_all] + fragments, axis=0)
                print(Michel_all.shape)
            # Michel_true_clusters = DBSCAN(eps=2., min_samples=5).fit(Michel_coords).labels_
            # Michel_coords = Michel_coords[Michel_true_clusters>-1]
            # Michel_true_clusters = Michel_true_clusters[Michel_true_clusters>-1]
            # Michel_all = Michel_all[Michel_true_clusters>-1]

        Michel_coords = Michel_all[:, coords_col[0]:coords_col[1]]
        MIP_coords_pred = data_pred[(predictions == MIP_label).reshape((-1,)), ...][:, coords_col[0]:coords_col[1]]
        Michel_coords_pred = data_pred[(predictions == Michel_label).reshape((-1,)), ...][:, coords_col[0]:coords_col[1]]

        # 1. Find true particle information matching the true Michel cluster
        # Michel_true_clusters = DBSCAN(eps=one_pixel, min_samples=5).fit(Michel_coords).labels_
        # Michel_true_clusters = [Michel_coords[Michel_coords[:, -2] == gid] for gid in np.unique(Michel_coords[:, -2])]
        #print(clusters.shape, label.shape)
        Michel_true_clusters = Michel_all[:, 6].astype(np.int64)

        # Michel_start = Michel_particles[:, :3]
        for cluster in np.unique(Michel_true_clusters):
            # print("True", np.count_nonzero(Michel_true_clusters == cluster))
            # TODO sum_pix
            fout_true.record(('batch_id', 'iteration', 'event_idx',
                             'num_pix', 'sum_pix'),
                             (batch_id, iteration, event_idx,
                              np.count_nonzero(Michel_true_clusters == cluster),
                              Michel_all[Michel_true_clusters == cluster][:, -4].sum()))
            fout_true.write()

        # TODO how do we count events where there are no predictions but true?
        if MIP_coords_pred.shape[0] == 0 or Michel_coords_pred.shape[0] == 0:
            continue
        # print("Also predicted!")
        # 2. Compute true and predicted clusters
        MIP_clusters = DBSCAN(eps=one_pixel, min_samples=5).fit(MIP_coords_pred).labels_
        if np.count_nonzero(MIP_clusters>-1) == 0:
            continue
        Michel_pred_clusters = DBSCAN(eps=2*one_pixel, min_samples=5).fit(Michel_coords_pred).labels_
        Michel_pred_clusters_id = np.unique(Michel_pred_clusters[Michel_pred_clusters>-1])
        Michel_true_clusters_id = np.unique(Michel_true_clusters[Michel_true_clusters > -1])

        # Loop over predicted Michel clusters
        Michel_is_attached, Michel_is_edge = [], []
        for Michel_id in Michel_pred_clusters_id:
            current_index = Michel_pred_clusters == Michel_id
            # 3. Check whether predicted Michel is attached to a predicted MIP
            # and at the edge of the predicted MIP
            distances = cdist(Michel_coords_pred[current_index], MIP_coords_pred[MIP_clusters>-1])
            # is_attached = np.min(distances) < 2.8284271247461903
            is_attached = np.min(distances) < 5
            is_edge = False  # default
            # print("Min distance:", np.min(distances))
            if is_attached:
                Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
                MIP_id = MIP_clusters[MIP_clusters>-1][MIP_min]
                MIP_min_coords = MIP_coords_pred[MIP_clusters>-1][MIP_min]
                MIP_cluster_coords = MIP_coords_pred[MIP_clusters==MIP_id]
                ablated_cluster = MIP_cluster_coords[np.linalg.norm(MIP_cluster_coords-MIP_min_coords, axis=1)>15.0]
                if ablated_cluster.shape[0] > 0:
                    new_cluster = DBSCAN(eps=one_pixel, min_samples=5).fit(ablated_cluster).labels_
                    is_edge = len(np.unique(new_cluster[new_cluster>-1])) == 1
                else:
                    is_edge = True
            # print(is_attached, is_edge)
            Michel_is_attached.append(is_attached)
            Michel_is_edge.append(is_edge)

        Michel_is_attached = np.array(Michel_is_attached, dtype=np.bool)
        Michel_is_edge = np.array(Michel_is_edge, dtype=np.bool)

        candidates = np.isin(Michel_pred_clusters, Michel_pred_clusters_id[Michel_is_edge & Michel_is_attached])

        # Record all predicted Michel cluster
        for idx, Michel_id in enumerate(Michel_pred_clusters_id):
            current_index = Michel_pred_clusters == Michel_id
            fout_reco.record(('batch_id', 'iteration', 'event_idx', 'is_attached', 'is_edge',
                            'num_pix'),
                             (batch_id, iteration, event_idx, Michel_is_attached[idx], Michel_is_edge[idx],
                             np.count_nonzero(current_index)))
            fout_reco.write()

        if np.count_nonzero(candidates) == 0:
            continue

        for Michel_id in Michel_true_clusters_id:
            current_index = Michel_true_clusters == Michel_id
            distances = cdist(Michel_coords_pred[candidates], Michel_coords[current_index])
            closest_clusters = Michel_pred_clusters[candidates][np.argmin(distances, axis=0)]
            closest_clusters_matching = closest_clusters[(closest_clusters > -1) & (np.min(distances, axis=0)<one_pixel)]
            if len(closest_clusters_matching) == 0:
                continue
            # Index of Michel predicted cluster that overlaps the most
            closest_pred_id = np.bincount(closest_clusters_matching).argmax()
            if closest_pred_id < 0:
                continue
            closest_Michel_pred = Michel_coords_pred[Michel_pred_clusters == closest_pred_id]

            # Intersection
            michel_pred_num_pix_true = 0
            michel_pred_sum_pix_true = 0.
            for v in data_masked[clusters_semantics == Michel_label][current_index]:
                count = int(np.any(np.all(v[coords_col[0]:coords_col[1]] == closest_Michel_pred, axis=1)))
                michel_pred_num_pix_true += count
                if count > 0:
                    michel_pred_sum_pix_true += v[-1]

            michel_true_num_pix = particles[Michel_id].num_voxels() #np.count_nonzero(current_index)
            michel_true_sum_pix = data_masked[clusters_semantics == Michel_label][current_index, 4].sum()
            michel_pred_num_pix = np.count_nonzero(Michel_pred_clusters == closest_pred_id)
            michel_pred_sum_pix = data_pred[predictions == Michel_label][Michel_pred_clusters == closest_pred_id, 4].sum()
            michel_true_energy = particles[Michel_id].energy_init()
            michel_true_num_pix_cluster = np.count_nonzero(current_index)
            # for v in Michel_coords[Michel_true_clusters == closest_true_id]:
            #     if (v == data[label == Michel_label, coords_col[0]:coords_col[1]]).all(axis=1).any():
            #         michel_true_num_pix_cluster += 1

            # Record every predicted Michel cluster in CSV
            fout_matched.record(('batch_id', 'iteration', 'event_idx', 'pred_num_pix', 'pred_sum_pix',
                              'pred_num_pix_true', 'pred_sum_pix_true',
                              'true_num_pix', 'true_sum_pix',
                              'is_attached', 'is_edge', 'michel_true_energy', 'true_num_pix_cluster'),
                             (batch_id, iteration, event_idx, michel_pred_num_pix, michel_pred_sum_pix,
                              michel_pred_num_pix_true, michel_pred_sum_pix_true,
                              michel_true_num_pix, michel_true_sum_pix,
                              is_attached, is_edge, michel_true_energy, michel_true_num_pix_cluster))
            fout_matched.write()

        if not store_per_iteration:
            fout_reco.close()
            fout_true.close()
            fout_matched.close()

    if store_per_iteration:
        fout_reco.close()
        fout_true.close()
        fout_matched.close()
