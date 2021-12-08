import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from mlreco.post_processing import post_processing
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label
from mlreco.utils.vertex import predict_vertex, get_vertex


@post_processing(['vertex-metrics', 'vertex-candidates', 'vertex-true'],
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
    endpoint_label      = module_cfg.get('endpoint_label', 0)
    track_label         = module_cfg.get('track_label', 1)
    shower_label        = module_cfg.get('shower_label', 0)
    nu_col              = module_cfg.get('nu_col', 8)
    coords_col          = module_cfg.get('coords_col', (1, 4))
    attaching_threshold = module_cfg.get('attaching_threshold', 2)
    inter_threshold     = module_cfg.get('inter_threshold', 10)

    node_pred_vtx = node_pred_vtx[data_idx]
    original_node_pred_vtx = node_pred_vtx
    clusts = inter_particles[data_idx]

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
    for inter_idx in np.unique(inter_group_pred[data_idx]):
        ppn_candidates, c_candidates, vtx_candidate, vtx_std = predict_vertex(inter_idx, data_idx, input_data, res,
                                                                            coords_col=coords_col, primary_label=primary_label,
                                                                            shower_label=shower_label, track_label=track_label,
                                                                            endpoint_label=endpoint_label,
                                                                            attaching_threshold=attaching_threshold,
                                                                            inter_threshold=inter_threshold)
        inter_mask = inter_group_pred[data_idx] == inter_idx
        interaction = inter_particles[data_idx][inter_mask]
        primary_particles = np.argmax(original_node_pred_vtx[inter_mask][:, 3:], axis=1) == primary_label

        # No primary particle in interaction
        if len(clusts[inter_mask][primary_particles]) == 0:
            continue

        is_nu = get_cluster_label(clust_data[data_idx], clusts[inter_mask][primary_particles], column=nu_col)
        #print(is_nu, len(clusts[inter_mask][primary_particles]))
        is_nu = is_nu[is_nu > -1]
        is_nu = np.argmax(np.bincount(is_nu.astype(int))) if len(is_nu) else -1
        #print(is_nu)
        #print(inter_idx, len(clusts[inter_mask]))
        vtx_resolution = -1
        clust_assn_vtx = [-1, -1, -1]
        if len(ppn_candidates):
            # Now evaluate vertex candidates for this interaction
            clust_assn_vtx = node_assn_vtx[positives.astype(bool) & (inter_mask[good_index])] * spatial_size
            clust_assn_vtx = clust_assn_vtx.mean(axis=0)
            #print(vtx_candidate, clust_assn_vtx)
            vtx_resolution += np.linalg.norm(clust_assn_vtx-vtx_candidate)
            #print("resolution = ", vtx_resolution)
            vtx_candidates.append(vtx_candidate)
        row_candidates_names.append(("num_ppn_candidates", "vtx_resolution",
                                    "vtx_candidate_x", "vtx_candidate_y", "vtx_candidate_z",
                                    "vtx_true_x", "vtx_true_y", "vtx_true_z",
                                    "vtx_std_x", "vtx_std_y", "vtx_std_z",
                                    "num_primaries", "is_nu", "num_pix"))
        row_candidates_values.append((len(ppn_candidates), vtx_resolution,
                                    vtx_candidate[0], vtx_candidate[1], vtx_candidate[2],
                                    clust_assn_vtx[0], clust_assn_vtx[1], clust_assn_vtx[2],
                                    vtx_std[0], vtx_std[1], vtx_std[2],
                                    len(c_candidates), is_nu, np.sum([len(c) for c in clusts[inter_mask][primary_particles]])))

    vtx_candidates = np.array(vtx_candidates)
    #print(vtx_candidates)
    node_assn_vtx = np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1)
    row_true_names, row_true_values = [], []
    num_true = len(np.unique(clust_data[data_idx][:, 7]))
    for inter_idx in np.unique(clust_data[data_idx][:, 7]):
        if inter_idx < 0:
            continue
        inter_mask = clust_data[data_idx][:, 7] == inter_idx
        vtx = get_vertex(kinematics, clust_data, data_idx, inter_idx, vtx_col=vtx_col)

        # FIXME why are there sometimes several true vtx for same interaction?
        # using ancestor_vtx
        vtx_candidate = [-1, -1, -1]
        distance = -1
        if len(vtx_candidates):
            d = cdist([vtx], vtx_candidates).reshape((-1,))
            vtx_candidate = vtx_candidates[np.argmin(d)]
            distance = np.linalg.norm(vtx_candidate-vtx)
        is_nu = clust_data[data_idx][inter_mask][:, 8]
        is_nu = is_nu[is_nu > -1]
        is_nu = np.argmax(np.bincount(is_nu.astype(int))) if len(is_nu) else -1
        #print(inter_idx, distance, is_nu)
        row_true_names.append(("inter_idx", "vtx_true_x", "vtx_true_y", "vtx_true_z",
                                "vtx_candidate_x", "vtx_candidate_y", "vtx_candidate_z",
                                "vtx_resolution", "is_nu", "num_pix"))
        row_true_values.append((inter_idx, vtx[0], vtx[1], vtx[2],
                                vtx_candidate[0], vtx_candidate[1], vtx_candidate[2],
                                distance, is_nu, np.count_nonzero(inter_mask)))

    if len(vtx_candidates):
        vtx_resolution /= len(vtx_candidates)

    row_names = ('accuracy_score', 'num_pred_positives', 'num_true_positives',
                'accuracy_true_positives', 'accuracy_pred_positives',
                'accuracy_position', 'n_clusts_vtx', 'n_clusts_vtx_positives',
                'vtx_resolution', 'num_candidates', 'num_true')
    row_values = (accuracy_positives, np.count_nonzero(pred_positives), np.count_nonzero(positives),
                (pred_positives == positives)[positives > 0].sum(), (pred_positives == positives)[pred_positives > 0].sum(),
                accuracy_position, n_clusts_vtx, n_clusts_vtx_positives,
                vtx_resolution, len(vtx_candidates), num_true)

    return [(row_names, row_values), (row_candidates_names, row_candidates_values), (row_true_names, row_true_values)]
