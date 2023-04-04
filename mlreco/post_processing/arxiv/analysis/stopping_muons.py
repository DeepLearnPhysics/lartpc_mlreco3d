import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from mlreco.post_processing import post_processing


@post_processing(['stopping-muons-pred', 'stopping-muons-true'], ['input_data', 'seg_label', 'clust_data', 'particles_asis'], ['segmentation', 'particles', 'particles_seg'])
def stopping_muons(cfg, module_cfg, data_blob, res, logdir, iteration,
                    data_idx=None, seg_label=None, seg_prediction=None,
                    particles_asis=None, clust_data=None, input_data=None,
                    particles=None, particles_seg=None, **kwargs):
    """
    Find stopping muons for calibration purpose (dE/dx).

    Parameters
    ----------
    data_blob: dict
        The input data dictionary from iotools.
    res: dict
        The output of the network, formatted using `analysis_keys`.
    cfg: dict
        Configuration.
    logdir: string
        Path to folder where CSV logs can be stored.
    iteration: int
        Current iteration number.

    Notes
    -----
    N/A.
    """
    spatial_size = module_cfg.get('spatial_size', 768)
    track_label = module_cfg.get('track_label', 1)
    threshold = module_cfg.get('threshold', 10)
    segment_length = module_cfg.get('segment_length', 10)
    min_overlap = module_cfg.get('min_overlap', 10)
    segment_threshold = module_cfg.get('segment_threshold', 3)
    dEdx_ratio = module_cfg.get('dEdx_ratio', 3.)
    coords_col = module_cfg.get('coords_col', (1, 4))

    # Identify true stopping muons first
    # ==================================
    # Loop over true group ids <-> particle idx
    # Use pdg + end position to make sure it is
    # stopping in the volume.
    true_stopping_muons = []
    true_muons = []
    for part_id in np.unique(clust_data[data_idx][:, 6]):
        p = particles_asis[data_idx][int(part_id)]
        if p.pdg_code() in [13, -13]:
            #print("Muon!")
            true_muons.append(part_id)
            end_position = np.array([
                p.end_position().x(),
                p.end_position().y(),
                p.end_position().z()
            ])
            voxels = clust_data[data_idx][clust_data[data_idx][:, 6] == part_id][:, coords_col[0]:coords_col[1]]
            d = cdist(voxels, [end_position])
            end_voxel = voxels[d.argmin(axis=0)]
            if (end_voxel > threshold).all() and (end_voxel < spatial_size - threshold).all():
                #print("\t stopping!", end_position, end_voxel)
                true_stopping_muons.append(part_id)
            #else:
            #    print("Discarding", part_id, end_position, end_voxel)
    print('true stopping muons', true_stopping_muons)
    true_stopping_muons = np.array(true_stopping_muons)

    # Then identify predicted stopping muons.
    # =======================================
    pred_stopping_muons = []
    result = []
    row_names, row_values = [], []
    row_names_pred, row_values_pred = [], []
    true_stopping_muons_matching = np.zeros((len(true_stopping_muons),))
    for p in particles[data_idx][particles_seg[data_idx] == track_label]:
        voxels = input_data[data_idx][p][:, coords_col[0]:coords_col[1]]
        # It must be touching at most once the boundary
        # FIXME what if x position can be shifted?
        # touching = np.any((voxels < threshold) | (voxels > spatial_size - threshold), axis=1)
        # if touching.any():
        #     clusters_touching = DBSCAN(eps=threshold/2., min_samples=1).fit(voxels[touching]).labels_
        #     if len(np.unique(clusters_touching[clusters_touching>-1])) > 1:
        #         continue

        # Now compute segments' de/dx
        pca = PCA(n_components=2)
        pca_voxels = pca.fit_transform(voxels)
        #length = (pca_voxels[:, 0].max() - pca_voxels[:, 0].min())
        #print(len(pca_voxels), N, (pca_voxels[:, 0].max() - pca_voxels[:, 0].min()))
        bins = np.arange(pca_voxels[:, 0].min(), pca_voxels[:, 0].max(), segment_length)
        index = np.digitize(pca_voxels[:, 0], bins)
        partition = [pca_voxels[index == i] for i in range(1, len(bins))]
        edeps = [input_data[data_idx][p][index == i, 4] for i in range(1, len(bins))]
        dEdx = []
        dE, dx, dn = [], [], []
        #print(partition)
        for segment, edep in zip(partition, edeps):
            if len(segment):
                segment_dE = edep.sum()
                segment_dx = np.abs(segment[:, 0].max() - segment[:, 0].min())
                if segment_dx > 0:
                    dEdx.append(segment_dE/segment_dx)
                    dE.append(segment_dE)
                    dx.append(segment_dx)
                    dn.append(len(segment))
        dEdx = np.array(dEdx)
        dE = np.array(dE)
        dx = np.array(dx)
        dn = np.array(dn)

        if len(dEdx) <= 2*segment_threshold:
            continue

        # Check if there is a Bragg peak
        # print(np.argmax(dEdx - dEdx.mean()), len(dEdx))
        # keep = keep and not len(dEdx) <= 2*segment_threshold

        # Use segment_threshold to exclude when dEdx maximum
        # occurs far from edges
        keep1 = (dEdx.argmax() < segment_threshold or dEdx.argmax() >= len(dEdx) - segment_threshold)

        # Compare maximum dEdx to mean far from edges
        keep2 = (dEdx.max() > dEdx_ratio * dEdx[segment_threshold:-segment_threshold].mean())

        keep = keep1 and keep2
        # Check if can be associated to one of the true stopping muons
        # using pixel overlap
        matched = False
        overlap = -1
        true_p = -1
        if len(true_stopping_muons):
            overlaps = np.array([(clust_data[data_idx][p, 6] == true_p).sum() for true_p in true_stopping_muons])
            print('max overlap', overlaps)
            if overlaps.max() > min_overlap:
                matched = True
                true_p = true_stopping_muons[overlaps.argmax()]
                overlap = overlaps.max()
                true_stopping_muons_matching[overlaps.argmax()] += 1
        #print(keep, matched, dEdx.argmax(), len(dEdx), dEdx.max()/dEdx[segment_threshold:-segment_threshold].mean())
        row_names.append(('argmax_dEdx', 'max_dEdx', 'num_segments',
                        'mean_dEdx', 'mean_dEdx_corr', 'std_dEdx', 'std_dEdx_corr', 'min_dEdx', 'voxel_count',
                        'average_dx', 'average_dE', 'average_dn',
                        'matched', 'overlap', 'true_p', 'keep', 'keep1', 'keep2'))
        row_values.append((np.argmax(dEdx), np.max(dEdx), len(dEdx),
                        dEdx.mean(), dEdx[segment_threshold:-segment_threshold].mean(), dEdx.std(), dEdx[segment_threshold:-segment_threshold].std(), dEdx.min(), voxels.shape[0],
                        dx.mean(), dE.mean(), dn.mean(),
                        matched, overlap, true_p, keep, keep1, keep2))

    true_stopping_muons_matching = np.array(true_stopping_muons_matching)

    row_names_true, row_values_true = [], []
    for idx, part_id in enumerate(true_muons):
        p = particles_asis[data_idx][int(part_id)]
        row_names_true.append(("id", "matched", "keep",
                                "end_x", "end_y", "end_z",
                                "pdg"))
        matched = -1
        if part_id in true_stopping_muons:
            matched = true_stopping_muons_matching[np.where(true_stopping_muons == part_id)[0]][0]
        print('matched', matched)
        row_values_true.append((part_id, matched, part_id in true_stopping_muons,
                                p.end_position().x(), p.end_position().y(), p.end_position().z(),
                                p.pdg_code()))

    return [(row_names, row_values), (row_names_true, row_values_true)]
