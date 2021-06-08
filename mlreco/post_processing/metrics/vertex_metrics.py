import numpy as np
from scipy.spatial.distance import cdist

from mlreco.post_processing import post_processing
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label_np, get_momenta_label_np
from mlreco.utils.ppn import uresnet_ppn_point_selector, uresnet_ppn_type_point_selector


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)


@post_processing(['vertex-metrics', 'vertex-candidates'],
                ['input_data', 'seg_label', 'clust_data', 'particles', 'kinematics'],
                ['node_pred_vtx', 'clusts', 'inter_particles', 'inter_group_pred'])
def vertex_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                    seg_label=None, clust_data=None, particles=None, kinematics=None,
                    node_pred_vtx=None, clusts=None, inter_particles=None, inter_group_pred=None,
                    data_idx=None, input_data=None, **kwargs):
    spatial_size = module_cfg.get('spatial_size', 768)
    vtx_col = module_cfg.get('vtx_col', 9)
    vtx_positives_col = module_cfg.get('vtx_positives_col', 12)
    primary_label = module_cfg.get('primary_label', 1)
    endpoint_label = module_cfg.get('endpoint_label', 0)
    track_label = module_cfg.get('track_label', 1)
    shower_label = module_cfg.get('shower_label', 0)

    node_pred_vtx = node_pred_vtx[data_idx]
    original_node_pred_vtx = node_pred_vtx
    clusts = clusts[data_idx]

    node_x_vtx = get_cluster_label_np(kinematics[data_idx], clusts, column=vtx_col)
    node_y_vtx = get_cluster_label_np(kinematics[data_idx], clusts, column=vtx_col+1)
    node_z_vtx = get_cluster_label_np(kinematics[data_idx], clusts, column=vtx_col+2)

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
        inter_mask = inter_group_pred[data_idx] == inter_idx
        interaction = inter_particles[data_idx][inter_mask]
        # Identify predicted primary particles within the interaction
        primary_particles = np.argmax(original_node_pred_vtx[inter_mask][:, 3:], axis=1) == primary_label
        ppn_candidates = []
        print(inter_idx, len(clusts[inter_mask]))
        # Look at PPN predictions for each primary particle
        for c in clusts[inter_mask][primary_particles]:
            # Some gymnastics to accomodate PPN post-processing
            # while only running on the voxels of current primary particle
            clust_res = {
                'points': [masking(res['points'][data_idx])[c]],
                'mask_ppn2': [masking(res['mask_ppn2'][data_idx])[c]],
                'classify_endpoints': [masking(res['classify_endpoints'][data_idx])[c]],
                'segmentation': [masking(res['segmentation'][data_idx])[c]]
            }
            clust_input = masking(input_data[data_idx])[c]

            ppn = uresnet_ppn_type_point_selector(clust_input, clust_res, entry=0, score_threshold=0.7, type_threshold=2)
            if ppn.shape[0] == 0:
                continue
            ppn_voxels = ppn[:, :3]
            ppn_score = ppn[:, 5]
            #ppn_occupancy = ppn[:, 6]
            ppn_type = ppn[:, 7:12]
            ppn_endpoints = np.argmax(ppn[:, 12:14], axis=1)
            #no_delta = ppn_type[:, 3] < 0.5

            # Pick the PPN points predicted as start point for tracks
            # Pick any PPN point for showers (primary?)
            ppn_candidates.append(ppn[((ppn_type[:, track_label] > 0.5) & (ppn_endpoints == endpoint_label)) | (ppn_type[:, shower_label] > 0.5)])

        # Take barycenter
        if len(ppn_candidates):
            print("ppn_candidates", ppn_candidates)
            ppn_candidates = np.concatenate(ppn_candidates, axis=0)
            vtx_candidate = np.mean(ppn_candidates[:, :3], axis=0)
            vtx_candidates.append(vtx_candidate)

            # Now evaluate vertex candidates for this interaction
            clust_assn_vtx = node_assn_vtx[positives.astype(bool) & (inter_mask[good_index])] * spatial_size
            print(vtx_candidate, clust_assn_vtx)
            vtx_resolution += np.mean(np.abs(clust_assn_vtx-vtx_candidate))
        row_candidates_names.append(("num_ppn_candidates",))
        row_candidates_values.append((len(ppn_candidates),))

    if len(vtx_candidates):
        vtx_resolution /= len(vtx_candidates)

    row_names = ('accuracy_score', 'num_pred_positives', 'num_true_positives',
                'accuracy_true_positives', 'accuracy_pred_positives',
                'accuracy_position', 'n_clusts_vtx', 'n_clusts_vtx_positives',
                'vtx_resolution')
    row_values = (accuracy_positives, np.count_nonzero(pred_positives), np.count_nonzero(positives),
                (pred_positives == positives)[positives > 0].sum(), (pred_positives == positives)[pred_positives > 0].sum(),
                accuracy_position, n_clusts_vtx, n_clusts_vtx_positives,
                vtx_resolution)

    return [(row_names, row_values), (row_candidates_names, row_candidates_values)]
