import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from mlreco.post_processing import post_processing
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label_np, get_momenta_label_np
from mlreco.utils.ppn import uresnet_ppn_point_selector, uresnet_ppn_type_point_selector

#def find_vtx_candidates(res, data_idx, inter_group_pred, inter_particles):

@post_processing(['vertex-metrics', 'vertex-candidates', 'vertex-true'],
                ['input_data', 'seg_label', 'clust_data', 'particles', 'kinematics'],
                ['node_pred_vtx', 'clusts', 'inter_particles', 'inter_group_pred', 'particles_seg'])
def vertex_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                    seg_label=None, clust_data=None, particles=None, kinematics=None,
                    node_pred_vtx=None, clusts=None, inter_particles=None, inter_group_pred=None,
                    data_idx=None, input_data=None, particles_seg=None, **kwargs):
    spatial_size = module_cfg.get('spatial_size', 768)
    vtx_col = module_cfg.get('vtx_col', 9)
    vtx_positives_col = module_cfg.get('vtx_positives_col', 12)
    primary_label = module_cfg.get('primary_label', 1)
    endpoint_label = module_cfg.get('endpoint_label', 0)
    track_label = module_cfg.get('track_label', 1)
    shower_label = module_cfg.get('shower_label', 0)
    nu_col = module_cfg.get('nu_col', 8)

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
        ppn_candidates, c_candidates = [], []

        # No primary particle in interaction
        if len(clusts[inter_mask][primary_particles]) == 0:
            continue

        is_nu = get_cluster_label_np(clust_data[data_idx], clusts[inter_mask][primary_particles], column=nu_col)
        #print(is_nu, len(clusts[inter_mask][primary_particles]))
        is_nu = is_nu[is_nu > -1]
        is_nu = np.argmax(np.bincount(is_nu.astype(int))) if len(is_nu) else -1
        #print(is_nu)
        #print(inter_idx, len(clusts[inter_mask]))
        # Look at PPN predictions for each primary particle
        for c_idx, c in enumerate(clusts[inter_mask][primary_particles]):
            c_seg = particles_seg[data_idx][inter_mask][primary_particles][c_idx]
            if c_seg == shower_label:
                # TODO select primary fragment
                shower_primaries = np.argmax(res['shower_node_pred'][data_idx], axis=1) == 0
                shower_primary = None
                for p in res['shower_fragments'][data_idx][shower_primaries]:
                    if len(np.intersect1d(c, p)):
                        shower_primary = p
                        break
                if shower_primary is not None:
                    c = shower_primary

            # If it is not a shower or track particle, ignore
            if c_seg not in [track_label, shower_label]:
                continue

            # Some gymnastics to accomodate PPN post-processing
            # while only running on the voxels of current primary particle
            clust_res = {
                'points': [masking(res['points'][data_idx])[c]],
                'mask_ppn2': [masking(res['mask_ppn2'][data_idx])[c]],
                'classify_endpoints': [masking(res['classify_endpoints'][data_idx])[c]],
                'segmentation': [masking(res['segmentation'][data_idx])[c]]
            }
            clust_input = masking(input_data[data_idx])[c]

            ppn = uresnet_ppn_type_point_selector(clust_input, clust_res, entry=0, score_threshold=0.5, type_threshold=2)
            if ppn.shape[0] == 0:
                continue
            ppn_voxels = ppn[:, :3]
            ppn_score = ppn[:, 5]
            #ppn_occupancy = ppn[:, 6]
            ppn_type = ppn[:, 7:12]
            ppn_endpoints = np.argmax(ppn[:, 12:14], axis=1)

            no_delta = ppn_type[:, 3] < 0.5
            ppn = ppn[no_delta]
            ppn_voxels = ppn_voxels[no_delta]
            ppn_score = ppn_score[no_delta]
            ppn_type = ppn_type[no_delta]
            ppn_endpoints = ppn_endpoints[no_delta]

            # Pick the PPN points predicted as start point for tracks
            # Pick any PPN point for showers (primary?)
            ppn_candidates.append(ppn[((ppn_type[:, track_label] > 0.5) & (ppn_endpoints == endpoint_label)) | (ppn_type[:, shower_label] > 0.5)])
            c_candidates.append(c)

        if len(ppn_candidates) > 1:
            ppn_candidates2 = []
            for p_idx, points in enumerate(ppn_candidates):
                #print(p_idx, points[:, :3])
                #print(points.shape)
                #print(masking(input_data[data_idx])[np.hstack(c_candidates[:p_idx] + c_candidates[p_idx+1:])][:, :3].shape)
                #print(p_idx, np.hstack(c_candidates[:p_idx] + c_candidates[p_idx+1:]))

                d = cdist(points[:, :3], masking(input_data[data_idx])[np.hstack([c for idx, c in enumerate(c_candidates) if idx != p_idx])][:, :3])
                #print(p_idx, d.min(axis=1))
                if d.min() < 7:
                    ppn_candidates2.append(points[np.where(d.min(axis=1) < 7)[0]])
            ppn_candidates = ppn_candidates2

        vtx_resolution = -1
        vtx_std, vtx_candidate = [-1, -1, -1], [-1, -1, -1]
        clust_assn_vtx = [-1, -1, -1]
        # Take barycenter
        if len(ppn_candidates):
            ppn_candidates = np.concatenate(ppn_candidates, axis=0)
            #print("ppn_candidates", ppn_candidates[:, :4])
            
            # Refine with dbscan to eliminate ppn candidates that are
            # far away (e.g. middle of a track)
            # ppn_candidates_group = DBSCAN(eps=7, min_samples=1).fit(ppn_candidates[:, :3]).labels_
            # groups, counts = np.unique(ppn_candidates_group, return_counts=True)
            # #print(counts)
            # ppn_candidates = ppn_candidates[ppn_candidates_group == groups[np.argmax(counts)]]

            #print("Selecting %d / %d points after dbscan" % (len(ppn_candidates), len(ppn_candidates_group)))
            # Now take barycenter
            vtx_candidate = np.mean(ppn_candidates[:, :3], axis=0)
            vtx_std = np.std(ppn_candidates[:, :3], axis=0)
            vtx_candidates.append(vtx_candidate)

            # Now evaluate vertex candidates for this interaction
            clust_assn_vtx = node_assn_vtx[positives.astype(bool) & (inter_mask[good_index])] * spatial_size
            clust_assn_vtx = clust_assn_vtx.mean(axis=0)
            #print(vtx_candidate, clust_assn_vtx)
            vtx_resolution += np.linalg.norm(clust_assn_vtx-vtx_candidate)
            #print("resolution = ", vtx_resolution)

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
        #vtx =
        vtx, counts = np.unique(kinematics[data_idx][inter_mask][:, [vtx_col, vtx_col+1, vtx_col+2]], axis=0, return_counts=True)
        vtx = vtx[np.argmax(counts)]
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
