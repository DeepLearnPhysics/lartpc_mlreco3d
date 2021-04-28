import numpy as np

from mlreco.post_processing import post_processing


@post_processing(['nue-selection-true', 'nue-selection-pred'], ['seg_label', 'clust_data', 'particles_asis'],
                ['segmentation', 'inter_group_pred', 'particles', 'particles_seg', 'node_pred_type', 'node_pred_vtx'])
def nue_selection(cfg, module_cfg, data_blob, res, logdir, iteration,
                data_idx=None, clust_data=None, particles_asis=None,
                inter_group_pred=None, particles=None, particles_seg=None,
                node_pred_type=None, node_pred_vtx=None, **kwargs):
    """
    Find electron neutrinos.

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
    shower_label = module_cfg.get('shower_label', 0)
    track_label = module_cfg.get('track_label', 1)
    electron_label = module_cfg.get('electron_label', 1)
    proton_label = module_cfg.get('proton_label', 4)

    # Loop over true interactions
    row_names_true, row_values_true = [], []
    for inter_id in np.unique(clust_data[data_idx][:, 7]):
        interaction_mask = clust_data[data_idx][:, 7] == inter_id
        for part_id in np.unique(clust_data[data_idx][interaction_mask, 6]):
            particle = particles_asis[data_idx][int(part_id)]
            #print(inter_id, particle.creation_process(), particle.nu_current_type(), particle.nu_interaction_type())

        row_names_true.append(("inter_id", "num_particles", "num_voxels"))
        row_values_true.append((inter_id, len(np.unique(clust_data[data_idx][interaction_mask, 6])), np.count_nonzero(interaction_mask)))

    # Loop over predicted interactions
    row_names_pred, row_values_pred = [], []
    for inter_id in np.unique(inter_group_pred[data_idx]):
        interaction_mask = inter_group_pred[data_idx] == inter_id
        current_interaction = particles[data_idx][interaction_mask]
        current_types = np.argmax(node_pred_type[data_idx][interaction_mask], axis=1)
        current_particles_seg = particles_seg[data_idx][interaction_mask]

        # Require >= 1 electron shower
        keep1 = (shower_label in current_particles_seg) and (electron_label in current_types[current_particles_seg == shower_label])
        # Require >= 1 proton track
        keep2 = (track_label in current_particles_seg) and (proton_label in current_types[current_particles_seg == track_label])
        # Require among predicted primaries exactly 1 electron shower and >= 1 proton track
        primaries = np.argmax(node_pred_vtx[data_idx][interaction_mask, 3:], axis=1)
        keep3 = (primaries[(current_types == electron_label) & (current_particles_seg == shower_label)] == 1).sum() == 1
        keep4 = (primaries[(current_types == proton_label) & (current_particles_seg == track_label)] == 1).sum() >= 1

        row_names_pred.append(("inter_id", "num_particles", "keep",
                                "require_electron_shower", "require_proton_track", "require_primary_electron", "require_primary_proton"))
        row_values_pred.append((inter_id, len(current_interaction), keep1 and keep2 and keep3 and keep4,
                                keep1, keep2, keep3, keep4))

    return [(row_names_true, row_values_true), (row_names_pred, row_values_pred)]
