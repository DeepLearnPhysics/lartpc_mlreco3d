import numpy as np
import pandas as pd
from mlreco.post_processing import post_processing
from mlreco.utils.gnn.cluster import get_cluster_label, get_cluster_batch
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


def get_fragments(input_data, fragments, step=16, track_endpoints=None, spatial_size=768, fiducial=33, min_length=200):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)

    local_fragments = []
    residual_range = []
    endpoints = []
    cluster_id = []
    for idx, f in enumerate(fragments):
        coords = input_data[f, 1:4]
        new_coords = pca.fit_transform(coords)
        new_coords[:, 1:] = 0.

        #
        # Require a minimum length
        #
        if new_coords[:, 0].max() - new_coords[:, 0].min() < min_length:
            continue

        #
        # If we want to use rough residual range, then this
        # block makes sure we are starting from the endpoint.
        # We also recompute the residual range later to use dx estimations.
        #
        endpoint = None
        if track_endpoints is not None and len(track_endpoints) > 0:
            from scipy.spatial.distance import cdist
            distances = cdist(track_endpoints[:, 1:4], coords)
            if (distances.min(axis=0) < 2).any():
                endpoint_idx, pt_idx = np.unravel_index(np.argmin(distances), distances.shape)
                endpoint = new_coords[pt_idx, 0]
                real_endpoint = track_endpoints[endpoint_idx, 1:4]

        # If None could be found, it probably means
        # that the track is exiting the volume.
        # If it was found but it is too close to the
        # boundaries, we also skip it.
        if endpoint is None or (np.abs(real_endpoint) < fiducial).any() or (np.abs(spatial_size - real_endpoint) < fiducial).any():
            continue

        # Now we split the track into smaller fragments
        # of size `step` along the main PCA axis
        n = np.ceil((np.ceil(new_coords[:, 0].max()) - np.floor(new_coords[:, 0].min()))/step)
        bins = np.digitize(new_coords[:, 0], np.linspace(np.floor(new_coords[:, 0].min()), np.ceil(new_coords[:, 0].max()), int(n)))
        for b in np.unique(bins):
            local_fragments.append(f[bins == b])
            middle_pt = (new_coords[bins == b, 0].min() + new_coords[bins == b, 0].max()) / 2.
            residual_range.append(np.abs(middle_pt - endpoint))
            endpoints.append(real_endpoint)
            cluster_id.append(idx)

    if len(endpoints):
        endpoints = np.stack(endpoints, axis=0)

    return local_fragments, residual_range, endpoints, cluster_id


def compute_metrics(local_fragments, cluster_id, residual_range, input_data, endpoints, min_voxels=3):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)

    metrics = {
        "dQdx": [],
        "rough_residual_range": [],
        "dN": [],
        "dQ": [],
        "dx": []
    }

    final_fragments, final_endpoints, final_cluster_id = [], [], []
    for idx, f in enumerate(local_fragments):
        if len(f) < min_voxels:
            continue
        coords = input_data[f, 1:4]
        new_coords = pca.fit_transform(coords)
        dx = new_coords[:, 0].max() - new_coords[:, 0].min()
        dQ = input_data[f, 4].sum()
        metrics["dQdx"].append(dQ/dx)
        metrics["rough_residual_range"].append(residual_range[idx])
        metrics["dN"].append(len(f))
        metrics["dQ"].append(dQ)
        metrics["dx"].append(dx)
        final_fragments.append(f)
        final_endpoints.append(endpoints[idx])
        final_cluster_id.append(cluster_id[idx])
    if len(final_endpoints):
        final_endpoints = np.stack(final_endpoints, axis=0)
    return pd.DataFrame(metrics), final_fragments, final_endpoints, np.array(final_cluster_id)


def compute_residual_range(input_data, final_cluster_ids, track_endpoints, metrics, final_fragments, endpoints,
                          spatial_size=768, fiducial=33):
    from scipy.spatial.distance import cdist

    #true_cluster_ids = get_cluster_label(clust_label, final_fragments, column=6)
    fragments_batch_ids = get_cluster_batch(input_data, final_fragments)

    final_fragments = np.array([np.array(f, dtype=int) for f in final_fragments], dtype=object)
    residual_range = - np.ones((len(final_fragments),))
    # Loop over events in current batch
    for b in fragments_batch_ids:
        batch_mask = fragments_batch_ids == b
        # Loop over track clusters in this event
        for idx in np.unique(final_cluster_ids[batch_mask]):
            track_fragments = final_fragments[batch_mask][final_cluster_ids[batch_mask] == idx]
            endpoint = endpoints[batch_mask][final_cluster_ids[batch_mask] == idx][0]

            # Now order fragments per distance to endpoint
            # We want to start from the fragment closest to the endpoint
            # and go from there.
            init_perm = np.argsort(metrics["rough_residual_range"][batch_mask][final_cluster_ids[batch_mask] == idx].values)
            order = []
            total_d = 0 # Keep track of total distance by accumulating dx
            # Loop over segments, starting with the one closest to the endpoint.
            for f_idx, f in enumerate(track_fragments[init_perm]):
                distances = cdist(input_data[np.array(f).astype(int), 1:4], [endpoint])
                order.append(total_d + (distances.max() + distances.min())/2)
                total_d += (distances.max() + distances.min())/2
                # At each step we change the "endpoint" to the other end of the segment.
                endpoint = input_data[np.array(f).astype(int), 1:4][np.argmax(distances.reshape((-1,)))]

            # Now we want to re-order according to the new distances we just computed.
            perm = np.argsort(np.array(order)[np.argsort(init_perm)])

            current_distance = 0.
            for f_idx, f in enumerate(track_fragments[perm]):
                dx = metrics["dx"][batch_mask][final_cluster_ids[batch_mask] == idx].values[perm][f_idx]
                current_distance += dx

                where = np.where(batch_mask)[0][final_cluster_ids[batch_mask] == idx][perm][f_idx]
                residual_range[where] = current_distance

    metrics["residual_range"] = residual_range
    metrics = metrics[metrics["residual_range"] > -1]
    return metrics


@post_processing('muon-residual-range',
                ['input_data', 'clust_data', 'points_label'],
                ['particles', 'particles_seg', 'node_pred_type'])
def muon_residual_range(cfg, module_cfg, data_blob, res, logdir, iteration,
                        input_data=None, particles=None, particles_seg=None, node_pred_type=None,
                        points_label=None, clust_data=None, data_idx=None, clust_data_noghost=None,
                        **kwargs):
    """
    Compute residual range vs dQ/dx for track-like particles (muons, protons).
    """
    use_true_points = module_cfg.get('use_true_points', False)

    track_label = 1
    track_mask = particles_seg[data_idx] == track_label
    muon_mask = np.argmax(node_pred_type[data_idx], axis=1) == 2
    muon_particles = particles[data_idx][track_mask & muon_mask]

    step = 16 # about 5cm
    # Bin tracks into smaller fragments
    if use_true_points:
        track_endpoints = points_label[data_idx][(points_label[data_idx][:, -1] == 1) & (points_label[data_idx][:, 4] == 1)]
    else:
        ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res, entry=data_idx)
        #ppn_voxels = ppn[:, 1:4]
        ppn_endpoints = np.argmax(ppn[:, 13:])
        ppn_track_score = ppn[:, 8]
        #track_endpoints = ppn[(ppn_endpoints == 1) & (ppn_track_score > 0.5)]
        #track_endpoints = ppn[(ppn_track_score > 0.5)]
        track_endpoints = ppn
    local_fragments, residual_range, endpoints, cluster_id = get_fragments(input_data[data_idx], muon_particles, step=step, track_endpoints=track_endpoints)

    if len(local_fragments) == 0:
        return (), ()

    # Get true PID label to select true muons
    #true_cluster_id =  get_cluster_label(clust_label, local_fragments, column=6)
    #true_pdg = np.array([particles[idx].pdg_code() for idx in true_cluster_id])
    #muons = (true_pdg == 13) | (true_pdg == -13)

    # For each smaller fragment compute metrics
    result, final_fragments, final_endpoints, final_cluster_id = compute_metrics(np.array(local_fragments, dtype=object), cluster_id, np.array(residual_range), input_data[data_idx],  endpoints)

    if len(final_fragments) == 0:
        return (), ()

    # Refine residual range computation
    result = compute_residual_range(input_data[data_idx], final_cluster_id, track_endpoints, result, final_fragments, final_endpoints)

    row_names, row_values = [], []
    for idx in range(len(result['dQdx'])):
        tuple_names = ("residual_range", "rough_residual_range", "dQdx", "dN", "dx", "dQ")
        row_names.append(tuple_names)
        row_values.append(tuple([result[key][idx] for key in tuple_names]))

    return row_names, row_values
