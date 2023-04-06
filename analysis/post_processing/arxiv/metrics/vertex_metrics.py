import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from mlreco.post_processing import post_processing
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label
from mlreco.utils.vertex import predict_vertex, get_vertex


@post_processing(['vertex-candidates', 'vertex-distances', 'vertex-distances-others', 'vertex-distances-primaries'],
                ['input_data', 'seg_label', 'clust_data', 'particles', 'kinematics'],
                ['node_pred_vtx', 'inter_particles', 'inter_group_pred', 'particles_seg'])
def vertex_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                    seg_label=None, clust_data=None, particles=None, kinematics=None,
                    node_pred_vtx=None, inter_particles=None, inter_group_pred=None,
                    data_idx=None, input_data=None, particles_seg=None, **kwargs):
    spatial_size        = module_cfg.get('spatial_size', 768)
    vtx_col             = module_cfg.get('vtx_col', 9)
    vtx_positives_col   = module_cfg.get('vtx_positives_col', 12)
    primary_label       = module_cfg.get('primary_label', 1)
    # endpoint_label      = module_cfg.get('endpoint_label', 0)
    track_label         = module_cfg.get('track_label', 1)
    shower_label        = module_cfg.get('shower_label', 0)
    nu_col              = module_cfg.get('nu_col', 8)
    coords_col          = module_cfg.get('coords_col', (1, 4))
    attaching_threshold = module_cfg.get('attaching_threshold', 10)
    inter_threshold     = module_cfg.get('inter_threshold', 20)
    other_primaries_threshold = module_cfg.get('other_primaries_threshold', 10)
    other_primaries_gamma_threshold = module_cfg.get('other_primaries_gamma_threshold', 100)
    # fraction_bad_primaries = module_cfg.get('fraction_bad_primaries', 0.6)
    min_overlap_count = module_cfg.get('min_overlap_count', 10)
    pca_radius = module_cfg.get('pca_radius', 28)
    min_track_count = module_cfg.get('min_track_count', 2)
    min_voxel_count = module_cfg.get('min_voxel_count', 10)

    node_pred_vtx = node_pred_vtx[data_idx]
    original_node_pred_vtx = node_pred_vtx
    clusts = inter_particles[data_idx]
    # print(np.unique(data_blob['cluster_label'][data_idx][data_blob['cluster_label'][data_idx][:, 6] == 3, 9]))
    node_x_vtx = get_cluster_label(kinematics[data_idx], clusts, column=vtx_col)
    node_y_vtx = get_cluster_label(kinematics[data_idx], clusts, column=vtx_col+1)
    node_z_vtx = get_cluster_label(kinematics[data_idx], clusts, column=vtx_col+2)

    node_assn_vtx = np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1)
    node_assn_vtx = node_assn_vtx/spatial_size

    good_index = np.all(np.abs(node_assn_vtx) <= 1., axis=1)

    positives = []
    for c in clusts:
        positives.append(kinematics[data_idx][c, vtx_positives_col].max().item())
    original_positives = np.array(positives)
    positives = np.array(positives)

    n_clusts_vtx = (good_index).sum()
    n_clusts_vtx_positives = (good_index & positives.astype(bool)).sum()

    node_pred_vtx = node_pred_vtx[good_index]
    node_assn_vtx = node_assn_vtx[good_index]
    positives = original_positives[good_index]

    pred_positives = np.argmax(node_pred_vtx[:, 3:], axis=1)
    accuracy_positives = (pred_positives == positives).sum() / len(positives)
    # SMAPE metric
    accuracy_position = np.sum(1. - np.abs(node_pred_vtx[positives.astype(bool), :3]-node_assn_vtx[positives.astype(bool)])/(np.abs(node_assn_vtx[positives.astype(bool)]) + np.abs(node_pred_vtx[positives.astype(bool), :3])))/3.

    # Look at each interaction predicted
    vtx_candidates = []
    mask_ghost = np.argmax(res['ghost'][data_idx], axis=1) == 0
    masking = lambda ar: ar[mask_ghost] if 'ghost' in res else ar
    vtx_resolution = 0.
    row_candidates_names, row_candidates_values = [], []
    row_distances_names, row_distances_values = [], []
    row_distances_others_names, row_distances_others_values = [], []
    row_distances_primaries_names, row_distances_primaries_values = [], []
    for inter_idx in np.unique(inter_group_pred[data_idx]):
        inter_mask = inter_group_pred[data_idx] == inter_idx
        interaction = inter_particles[data_idx][inter_mask]
        ppn_candidates, c_candidates, vtx_candidate, vtx_std, ppn_candidates_old, distances, distances_others, distances_primaries = predict_vertex(inter_idx, data_idx, input_data, res,
                                                                            coords_col=coords_col, primary_label=primary_label,
                                                                            shower_label=shower_label, track_label=track_label,
                                                                            #endpoint_label=endpoint_label,
                                                                            attaching_threshold=attaching_threshold,
                                                                            inter_threshold=inter_threshold,
                                                                            return_distances=True,
                                                                            other_primaries_threshold=other_primaries_threshold,
                                                                            other_primaries_gamma_threshold=other_primaries_gamma_threshold,
                                                                            #fraction_bad_primaries=fraction_bad_primaries,
                                                                            pca_radius=pca_radius,
                                                                            min_voxel_count=min_voxel_count)
        inter_mask = inter_group_pred[data_idx] == inter_idx
        interaction = inter_particles[data_idx][inter_mask]
        primary_particles = np.argmax(original_node_pred_vtx[inter_mask][:, 3:], axis=1) == primary_label

        if len(distances):
            for x in np.hstack(distances):
                row_distances_names.append(("d",))
                row_distances_values.append((x,))
        if len(distances_others):
            distances_others = np.hstack(distances_others)
            row_distances_others_names.extend([("d",) for _ in distances_others])
            row_distances_others_values.extend([(x,) for x in distances_others])
        else:
            distances_others = np.empty((1,))
        if len(distances_primaries):
            #print('distance to primaries', distances_primaries)
            row_distances_primaries_names.extend([('d',) for _ in distances_primaries])
            row_distances_primaries_values.extend([(x,) for x in distances_primaries])
        else:
            distances_primaries = np.empty((1,))
        # No primary particle in interaction
        if len(clusts[inter_mask][primary_particles]) == 0:
            continue


        # Match to a true interaction

        is_nu = get_cluster_label(clust_data[data_idx], clusts[inter_mask][primary_particles], column=nu_col)
        #print(is_nu, len(clusts[inter_mask][primary_particles]))
        #print(clust_data[data_idx][clusts[inter_mask][0], nu_col])
        is_nu = is_nu[is_nu > -1]
        is_nu = np.argmax(np.bincount(is_nu.astype(int))) if len(is_nu) else -1

        print('Predicted interaction ', inter_idx, len(np.hstack(interaction)), is_nu, [len(x) for x in interaction])

        #print(is_nu)
        #print(inter_idx, len(clusts[inter_mask]))
        vtx_resolution = -1
        clust_assn_vtx = [-1, -1, -1]
        max_overlap = 0
        matched_inter_idx = -1
        if len(ppn_candidates):
            # Now evaluate vertex candidates for this interaction
            clust_assn_vtx = node_assn_vtx[positives.astype(bool) & (inter_mask[good_index])] * spatial_size
            clust_assn_vtx = clust_assn_vtx.mean(axis=0)
            print('Clust assn vtx', clust_assn_vtx)
            for true_inter_idx in np.unique(clust_data[data_idx][:, 7]):
                if true_inter_idx < 0:
                    continue
                true_inter_mask = clust_data[data_idx][:, 7] == true_inter_idx
                intersection = np.intersect1d(np.hstack(interaction), np.where(true_inter_mask)[0])
                print('check ', inter_idx, true_inter_idx, intersection)
                if len(intersection) > max_overlap and len(intersection) > min_overlap_count:
                    max_overlap = len(intersection)
                    matched_inter_idx = true_inter_idx
            print('We have candidates and overlap is ', max_overlap)
            if matched_inter_idx > -1:
                vtx = get_vertex(kinematics, clust_data, data_idx, matched_inter_idx, vtx_col=vtx_col)
                print("Vertex candidate and true = ", vtx_candidate, clust_assn_vtx, vtx)
            else:
                print("Vertex candidate and true = ", vtx_candidate, clust_assn_vtx, None)
            vtx_resolution = np.linalg.norm(clust_assn_vtx-vtx_candidate)
            print("resolution = ", vtx_resolution)
            vtx_candidates.append(vtx_candidate)
        row_candidates_names.append(("inter_id", "num_ppn_candidates", "num_ppn_associations", "vtx_resolution",
                                    "vtx_candidate_x", "vtx_candidate_y", "vtx_candidate_z",
                                    "vtx_true_x", "vtx_true_y", "vtx_true_z",
                                    "vtx_std_x", "vtx_std_y", "vtx_std_z",
                                    "num_primaries", "is_nu", "num_pix",
                                    "overlap", "distance_others_min", "distance_others_max",
                                    "distance_primaries_min", "distance_primaries_max",
                                    "matched_inter_id"))

        row_candidates_values.append((inter_idx, len(ppn_candidates), len(ppn_candidates_old), vtx_resolution,
                                    vtx_candidate[0], vtx_candidate[1], vtx_candidate[2],
                                    clust_assn_vtx[0], clust_assn_vtx[1], clust_assn_vtx[2],
                                    vtx_std[0], vtx_std[1], vtx_std[2],
                                    np.sum(primary_particles), is_nu, np.sum([len(c) for c in clusts[inter_mask][primary_particles]]),
                                    max_overlap, np.amin(distances_others), np.amax(distances_others),
                                    np.amin(distances_primaries), np.amax(distances_primaries),
                                    matched_inter_idx))

    return [(row_candidates_names, row_candidates_values),
            (row_distances_names, row_distances_values),
            (row_distances_others_names, row_distances_others_values),
            (row_distances_primaries_names, row_distances_primaries_values)]
