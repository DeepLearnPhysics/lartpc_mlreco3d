import numpy as np
import os
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from mlreco.utils import CSVData


def find_edges(coords):
    point0 = coords[0]
    i = cdist([point0], coords).argmax(axis=1)[0]
    point1 = coords[i]
    j = cdist([point1], coords).argmax(axis=1)[0]
    return [i, j]


def is_at_edge(cluster_coords, point_coords, one_pixel=1, radius=10.0):
    """
    Determines whether the point with coordinates `point_coords` is at the edge
    of the cluster with coordinates `cluster_coords`.
    Removes a disc of radius `radius` around that point, runs DBSCAN and checks
    whether there is still only 1 cluster.

    Assumes: that DBSCAN run on cluster_coords will only find 1 cluster.

    cluster_coords: np.array (N, data_dim)
    point_coords: np.array (1, data_dim) or (data_dim,)
    """
    ablated_cluster = cluster_coords[np.linalg.norm(cluster_coords-point_coords, axis=1)>radius]
    if ablated_cluster.shape[0] > 0:
        new_cluster = DBSCAN(eps=one_pixel, min_samples=5).fit(ablated_cluster).labels_
        return len(np.unique(new_cluster[new_cluster>-1])) == 1
    else:
        return True


def michel_reconstruction_2d(cfg, data_blob, res, logdir, iteration):
    """
    Very simple algorithm to reconstruct Michel clusters from UResNet semantic
    segmentation output.

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

    Notes
    -----
    Assumes 2D

    Input
    -----
    Requires the following analysis keys:
    - `segmentation` output of UResNet
    Requires the following input keys:
    - `input_data`
    - `segment_label`
    - `particles_label` to get detailed information such as energy.
    - `clusters_label` from `cluster3d_mcst` for true clusters informations

    Output
    ------
    Writes 2 CSV files:
    - `michel_reconstruction-*`
    - `michel_reconstruction2-*`
    """
    method_cfg = cfg['post_processing']['michel_reconstruction_2d']

    # Create output CSV
    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'

    fout_reco,fout_true=None,None
    if store_per_iteration:
        fout_reco=CSVData(os.path.join(logdir, 'michel-reconstruction-reco-iter-%07d.csv' % iteration))
        fout_true=CSVData(os.path.join(logdir, 'michel-reconstruction-true-iter-%07d.csv' % iteration))

    # Loop over events
    for batch_id,data in enumerate(data_blob['input_data']):

        event_idx = data_blob['index'          ][batch_id]

        if not store_per_iteration:
            fout_reco=CSVData(os.path.join(logdir, 'michel-reconstruction-reco-event-%07d.csv' % event_idx))
            fout_true=CSVData(os.path.join(logdir, 'michel-reconstruction-true-event-%07d.csv' % event_idx))

        # from input/labels
        data        = data_blob['input_data'     ][batch_id]
        label       = data_blob['segment_label'  ][batch_id][:,-1]
        meta        = data_blob['meta'           ][batch_id]

        # clusters    = data_blob['clusters_label' ][batch_id]
        # particles   = data_blob['particles_label'][batch_id]

        # Michel_particles = particles[particles[:, 2] == 4]  # FIXME 3 or 4 in 2D? Also not sure if type is registered for Michel

        # from network output
        segmentation = res['segmentation'][batch_id]
        predictions  = np.argmax(segmentation,axis=1)
        Michel_label = 3
        MIP_label = 0

        data_dim = 2
        # 0. Retrieve coordinates of true and predicted Michels
        MIP_coords = data[(label == MIP_label).reshape((-1,)), ...][:, :data_dim]
        Michel_coords = data[(label == Michel_label).reshape((-1,)), ...][:, :data_dim]
        # MIP_coords = clusters[clusters[:, -1] == 1][:, :3]
        # Michel_coords = clusters[clusters[:, -1] == 4][:, :3]
        if Michel_coords.shape[0] == 0:  # FIXME
            continue
        MIP_coords_pred = data[(predictions == MIP_label).reshape((-1,)), ...][:, :data_dim]
        Michel_coords_pred = data[(predictions == Michel_label).reshape((-1,)), ...][:, :data_dim]

        # DBSCAN epsilon used for many things... TODO list here
        #one_pixel = 15#2.8284271247461903
        one_pixel_dbscan = 5
        one_pixel_is_attached = 2
        # 1. Find true particle information matching the true Michel cluster
        Michel_true_clusters = DBSCAN(eps=one_pixel_dbscan, min_samples=5).fit(Michel_coords).labels_
        MIP_true_clusters = DBSCAN(eps=one_pixel_dbscan, min_samples=5).fit(MIP_coords).labels_

        # compute all edges of true MIP clusters
        MIP_edges = []
        for cluster in np.unique(MIP_true_clusters[MIP_true_clusters>-1]):
            touching_idx = find_edges(MIP_coords[MIP_true_clusters == cluster])
            MIP_edges.append(MIP_coords[MIP_true_clusters == cluster][touching_idx[0]])
            MIP_edges.append(MIP_coords[MIP_true_clusters == cluster][touching_idx[1]])
        # Michel_true_clusters = [Michel_coords[Michel_coords[:, -2] == gid] for gid in np.unique(Michel_coords[:, -2])]
        # Michel_true_clusters = clusters[clusters[:, -1] == 4][:, -2].astype(np.int64)
        # Michel_start = Michel_particles[:, :data_dim]

        true_Michel_is_attached = {}
        true_Michel_is_edge = {}
        true_Michel_is_too_close = {}
        for cluster in np.unique(Michel_true_clusters):
            min_y = Michel_coords[Michel_true_clusters == cluster][:, 1].min()# * meta[-1] + meta[1]
            max_y = Michel_coords[Michel_true_clusters == cluster][:, 1].max()# * meta[-1] + meta[1]
            min_x = Michel_coords[Michel_true_clusters == cluster][:, 0].min()# * meta[-2] + meta[0]
            max_x = Michel_coords[Michel_true_clusters == cluster][:, 0].max()# * meta[-2] + meta[0]

            # Find coordinates of Michel pixel touching MIP edge
            Michel_edges_idx = find_edges(Michel_coords[Michel_true_clusters == cluster])
            distances = cdist(Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx], MIP_coords[MIP_true_clusters>-1])

            # Make sure true Michel is attached at edge of MIP
            Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
            is_attached = np.min(distances) < one_pixel_is_attached
            is_too_close = np.max(distances) < one_pixel_is_attached
            # Check whether the Michel is at the edge of a predicted MIP
            # From the MIP pixel closest to the Michel, remove all pixels in
            # a radius of 15px. DBSCAN what is left and make sure it is all in
            # one single piece.
            is_edge = False  # default
            if is_attached:
                # cluster id of MIP closest
                MIP_id = MIP_true_clusters[MIP_true_clusters>-1][MIP_min]
                # coordinates of closest MIP pixel in this cluster
                MIP_min_coords = MIP_coords[MIP_true_clusters>-1][MIP_min]
                # coordinates of the whole cluster
                MIP_cluster_coords = MIP_coords[MIP_true_clusters==MIP_id]
                is_edge = is_at_edge(MIP_cluster_coords, MIP_min_coords, one_pixel=one_pixel_dbscan, radius=15.0)
            true_Michel_is_attached[cluster] = is_attached
            true_Michel_is_edge[cluster] = is_edge
            true_Michel_is_too_close[cluster] = is_too_close

            # these are the coordinates of Michel edges
            edge1_x = Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx[0], 0]
            edge1_y = Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx[0], 1]
            edge2_x = Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx[1], 0]
            edge2_y = Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx[1], 1]

            # Find for each Michel edge the closest MIP pixels
            # Check for each of these whether they are at the edge of MIP
            # FIXME what happens if both are at the edge of a MIP?? unlikely
            closest_MIP_pixels = np.argmin(distances, axis=1)
            clusters_idx = MIP_true_clusters[closest_MIP_pixels]
            edge0 = is_at_edge(MIP_coords[MIP_true_clusters == clusters_idx[0]],
                               MIP_coords[closest_MIP_pixels[0]],
                               one_pixel=one_pixel_dbscan,
                               radius=10.0)
            edge1 = is_at_edge(MIP_coords[MIP_true_clusters == clusters_idx[1]],
                               MIP_coords[closest_MIP_pixels[1]],
                               one_pixel=one_pixel_dbscan,
                               radius=10.0)
            if edge0 and not edge1:
                touching_x = edge1_x
                touching_y = edge1_y
            elif not edge0 and edge1:
                touching_x = edge2_x
                touching_y = edge2_y
            else:
                if distances[0, closest_MIP_pixels[0]] < distances[1, closest_MIP_pixels[1]]:
                    touching_x = edge1_x
                    touching_y = edge1_y
                else:
                    touching_x = edge2_x
                    touching_y = edge2_y
            # touching_idx = np.unravel_index(np.argmin(distances), distances.shape)
            # touching_x = Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx][touching_idx[0], 0]
            # touching_y = Michel_coords[Michel_true_clusters == cluster][Michel_edges_idx][touching_idx[0], 1]
            #
            # if touching_x not in [edge1_x, edge2_x] or touching_y not in [edge1_y, edge2_y]:
            #     print('true', event_idx, touching_x, touching_y, edge1_x, edge1_y, edge2_x, edge2_y)
            #if event_idx == 127:
            #    print('true', touching_x, touching_y, edge1_x, edge1_y, edge2_x, edge2_y)
            fout_true.record(('batch_id', 'iteration', 'event_idx', 'num_pix',
                              'sum_pix', 'min_y', 'max_y', 'min_x', 'max_x',
                              'pixel_width', 'pixel_height', 'meta_min_x', 'meta_min_y',
                              'touching_x', 'touching_y',
                              'edge1_x', 'edge1_y', 'edge2_x', 'edge2_y',
                              'edge0', 'edge1',
                              'is_attached', 'is_edge', 'is_too_close', 'cluster_id'),
                             (batch_id, iteration, event_idx,
                              np.count_nonzero(Michel_true_clusters == cluster),
                              data[(label == Michel_label).reshape((-1,)), ...][Michel_true_clusters == cluster][:, -1].sum(),
                              # clusters[clusters[:, -1] == 4][Michel_true_clusters == cluster][:, -3].sum()
                              min_y, max_y, min_x, max_x,
                              meta[-2], meta[-1], meta[0], meta[1],
                              touching_x, touching_y,
                              edge1_x, edge1_y, edge2_x, edge2_y,
                              edge0, edge1,
                              is_attached, is_edge, is_too_close, cluster
                             ))
            fout_true.write()
        # e.g. deposited energy, creation energy
        # TODO retrieve particles information
        # if Michel_coords.shape[0] > 0:
        #     Michel_clusters_id = np.unique(Michel_true_clusters[Michel_true_clusters>-1])
        #     for Michel_id in Michel_clusters_id:
        #         current_index = Michel_true_clusters == Michel_id
        #         distances = cdist(Michel_coords[current_index], MIP_coords)
        #         is_attached = np.min(distances) < 2.8284271247461903
        #         # Match to MC Michel
        #         distances2 = cdist(Michel_coords[current_index], Michel_start)
        #         closest_mc = np.argmin(distances2, axis=1)
        #         closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]

        # TODO how do we count events where there are no predictions but true?
        if MIP_coords_pred.shape[0] == 0 or Michel_coords_pred.shape[0] == 0:
            continue

        #
        # 2. Compute true and predicted clusters
        #
        MIP_clusters = DBSCAN(eps=one_pixel_dbscan, min_samples=10).fit(MIP_coords_pred).labels_
        MIP_clusters_id = np.unique(MIP_clusters[MIP_clusters>-1])

        # If no predicted MIP then continue TODO how do we count this?
        if MIP_coords_pred[MIP_clusters>-1].shape[0] == 0:
            continue

        # MIP_edges = []
        # for cluster in MIP_clusters_id:
        #     touching_idx = find_edges(MIP_coords_pred[MIP_clusters == cluster])
        #     MIP_edges.append(MIP_coords_pred[MIP_clusters == cluster][touching_idx[0]])
        #     MIP_edges.append(MIP_coords_pred[MIP_clusters == cluster][touching_idx[1]])

        Michel_pred_clusters = DBSCAN(eps=one_pixel_dbscan, min_samples=5).fit(Michel_coords_pred).labels_
        Michel_pred_clusters_id = np.unique(Michel_pred_clusters[Michel_pred_clusters>-1])
        # print(len(Michel_pred_clusters_id))

        # Loop over predicted Michel clusters
        for Michel_id in Michel_pred_clusters_id:
            current_index = Michel_pred_clusters == Michel_id
            # 3. Check whether predicted Michel is attached to a predicted MIP
            # and at the edge of the predicted MIP
            Michel_edges_idx = find_edges(Michel_coords_pred[current_index])
            # distances_edges = cdist(Michel_coords_pred[current_index][Michel_edges_idx], MIP_edges)
            # distances = cdist(Michel_coords_pred[current_index], MIP_coords_pred[MIP_clusters>-1])
            distances = cdist(Michel_coords_pred[current_index][Michel_edges_idx], MIP_coords_pred[MIP_clusters>-1])
            Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
            is_attached = np.min(distances) < one_pixel_is_attached
            is_too_close = np.max(distances) < one_pixel_is_attached
            # Check whether the Michel is at the edge of a predicted MIP
            # From the MIP pixel closest to the Michel, remove all pixels in
            # a radius of 15px. DBSCAN what is left and make sure it is all in
            # one single piece.
            is_edge = False  # default
            if is_attached:
                # cluster id of MIP closest
                MIP_id = MIP_clusters[MIP_clusters>-1][MIP_min]
                # coordinates of closest MIP pixel in this cluster
                MIP_min_coords = MIP_coords_pred[MIP_clusters>-1][MIP_min]
                # coordinates of the whole cluster
                MIP_cluster_coords = MIP_coords_pred[MIP_clusters==MIP_id]
                is_edge = is_at_edge(MIP_cluster_coords, MIP_min_coords, one_pixel=one_pixel_dbscan, radius=15.0)

            michel_pred_num_pix_true, michel_pred_sum_pix_true = -1, -1
            michel_true_num_pix, michel_true_sum_pix = -1, -1
            michel_true_energy = -1
            touching_x, touching_y = -1, -1
            edge1_x, edge1_y, edge2_x, edge2_y = -1, -1, -1, -1
            true_is_attached, true_is_edge, true_is_too_close = -1, -1, -1
            closest_true_id = -1

            # Find point where MIP and Michel touches
            # touching_idx = np.unravel_index(np.argmin(distances), distances.shape)
            # touching_x = Michel_coords_pred[current_index][Michel_edges_idx][touching_idx[0], 0]
            # touching_y = Michel_coords_pred[current_index][Michel_edges_idx][touching_idx[0], 1]
            edge1_x = Michel_coords_pred[current_index][Michel_edges_idx[0], 0]
            edge1_y = Michel_coords_pred[current_index][Michel_edges_idx[0], 1]
            edge2_x = Michel_coords_pred[current_index][Michel_edges_idx[1], 0]
            edge2_y = Michel_coords_pred[current_index][Michel_edges_idx[1], 1]

            closest_MIP_pixels = np.argmin(distances, axis=1)
            clusters_idx = MIP_clusters[MIP_clusters>-1][np.argmin(distances, axis=1)]
            edge0 = is_at_edge(MIP_coords_pred[MIP_clusters == clusters_idx[0]],
                               MIP_coords_pred[MIP_clusters>-1][closest_MIP_pixels[0]],
                               one_pixel=one_pixel_dbscan,
                               radius=10.0)
            edge1 = is_at_edge(MIP_coords_pred[MIP_clusters == clusters_idx[1]],
                               MIP_coords_pred[MIP_clusters>-1][closest_MIP_pixels[1]],
                               one_pixel=one_pixel_dbscan,
                               radius=10.0)
            if edge0 and not edge1:
                touching_x = edge1_x
                touching_y = edge1_y
            elif not edge0 and edge1:
                touching_x = edge2_x
                touching_y = edge2_y
            else:
                if distances[0, closest_MIP_pixels[0]] < distances[1, closest_MIP_pixels[1]]:
                    touching_x = edge1_x
                    touching_y = edge1_y
                else:
                    touching_x = edge2_x
                    touching_y = edge2_y

            if is_attached and is_edge:
                # Distance from current Michel pred cluster to all true points
                distances = cdist(Michel_coords_pred[current_index], Michel_coords)
                closest_clusters = Michel_true_clusters[np.argmin(distances, axis=1)]
                closest_clusters_final = closest_clusters[(closest_clusters > -1) & (np.min(distances, axis=1)<one_pixel_dbscan)]
                if len(closest_clusters_final) > 0:
                    # print(closest_clusters_final, np.bincount(closest_clusters_final), np.bincount(closest_clusters_final).argmax())
                    # cluster id of closest true Michel cluster
                    # we take the one that has most overlap
                    # closest_true_id = closest_clusters_final[np.bincount(closest_clusters_final).argmax()]
                    closest_true_id = np.bincount(closest_clusters_final).argmax()
                    overlap_pixels_index = (closest_clusters == closest_true_id) & (np.min(distances, axis=1)<one_pixel_dbscan)
                    if closest_true_id > -1:
                        closest_true_index = label[predictions==Michel_label][current_index]==Michel_label
                        # Intersection
                        michel_pred_num_pix_true = 0
                        michel_pred_sum_pix_true = 0.
                        for v in data[(predictions==Michel_label).reshape((-1,)), ...][current_index]:
                            count = int(np.any(np.all(v[:data_dim] == Michel_coords[Michel_true_clusters == closest_true_id], axis=1)))
                            michel_pred_num_pix_true += count
                            if count > 0:
                                michel_pred_sum_pix_true += v[-1]

                        michel_true_num_pix = np.count_nonzero(Michel_true_clusters == closest_true_id)
                        # michel_true_sum_pix = clusters[clusters[:, -1] == 4][Michel_true_clusters == closest_true_id][:, -3].sum()
                        michel_true_sum_pix = data[(label == Michel_label).reshape((-1,)), ...][Michel_true_clusters == closest_true_id][:, -1].sum()

                        # Check whether true Michel is attached to MIP, otherwise exclude
                        true_is_attached = true_Michel_is_attached[closest_true_id]
                        true_is_edge = true_Michel_is_edge[closest_true_id]
                        true_is_too_close = true_Michel_is_too_close[closest_true_id]
                        # Register true energy
                        # Match to MC Michel
                        # FIXME in 2D Michel_start is no good
                        # distances2 = cdist(Michel_coords[Michel_true_clusters == closest_true_id], Michel_start)
                        # closest_mc = np.argmin(distances2, axis=1)
                        # closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]
                        # michel_true_energy = Michel_particles[closest_mc_id, 7]
                        michel_true_energy = -1


            # Record every predicted Michel cluster in CSV
            # Record min and max x in real coordinates
            min_y = Michel_coords_pred[current_index][:, 1].min()# * meta[-1] + meta[1]
            max_y = Michel_coords_pred[current_index][:, 1].max()# * meta[-1] + meta[1]
            min_x = Michel_coords_pred[current_index][:, 0].min()# * meta[-2] + meta[0]
            max_x = Michel_coords_pred[current_index][:, 0].max()# * meta[-2] + meta[0]
            fout_reco.record(('batch_id', 'iteration', 'event_idx', 'pred_num_pix', 'pred_sum_pix',
                              'pred_num_pix_true', 'pred_sum_pix_true',
                              'true_num_pix', 'true_sum_pix',
                              'is_attached', 'is_edge', 'michel_true_energy',
                              'min_y', 'max_y', 'min_x', 'max_x',
                              'pixel_width', 'pixel_height', 'meta_min_x', 'meta_min_y',
                              'touching_x', 'touching_y',
                              'edge1_x', 'edge1_y', 'edge2_x', 'edge2_y',
                              'edge0', 'edge1',
                              'true_is_attached', 'true_is_edge', 'true_is_too_close',
                              'is_too_close', 'closest_true_index'),
                             (batch_id, iteration, event_idx, np.count_nonzero(current_index),
                              data[(predictions==Michel_label).reshape((-1,)), ...][current_index][:, -1].sum(),
                              michel_pred_num_pix_true, michel_pred_sum_pix_true, michel_true_num_pix, michel_true_sum_pix,
                              is_attached, is_edge, michel_true_energy,
                              min_y, max_y, min_x, max_x,
                              meta[-2], meta[-1], meta[0], meta[1],
                              touching_x, touching_y,
                              edge1_x, edge1_y, edge2_x, edge2_y,
                              edge0, edge1,
                              true_is_attached, true_is_edge, true_is_too_close,
                              is_too_close, closest_true_id))
            fout_reco.write()

        if not store_per_iteration:
            fout_reco.close()
            fout_true.close()

    if store_per_iteration:
        fout_reco.close()
        fout_true.close()
