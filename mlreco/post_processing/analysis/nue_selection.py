import numpy as np

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.cluster import get_cluster_label

@post_processing(['nue-selection-true', 'nue-selection-primaries'], ['seg_label', 'clust_data', 'particles_asis', 'kinematics'],
                ['segmentation', 'inter_group_pred', 'particles', 'particles_seg', 'node_pred_type', 'node_pred_vtx'])
def nue_selection(cfg, module_cfg, data_blob, res, logdir, iteration,
                data_idx=None, clust_data=None, particles_asis=None, kinematics=None,
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

    min_overlap_count = module_cfg.get('min_overlap_count', 10)

    row_names_true, row_values_true = [], []
    row_names_primaries, row_values_primaries = [], []

    # Find predicted primary particles in the event
    pred_primary_count = 0
    pred_primary_particles = []
    #print(np.amax(particles[data_idx]), clust_data[data_idx].shape)
    for pred_idx, pred_part in enumerate(particles[data_idx]):
        is_primary = np.argmax(node_pred_vtx[data_idx][pred_idx, 3:])
        if not is_primary: continue
        pred_primary_count += 1
        pred_primary_particles.append((pred_idx, pred_part))
        #print('Predicted primary', pred_idx, len(pred_part))
    #print(clust_data[data_idx].shape, kinematics[data_idx].shape)
    # Loop over true interactions
    for inter_id in np.unique(clust_data[data_idx][:, 7]):
        if inter_id == -1:
            continue
        interaction_mask = clust_data[data_idx][:, 7] == inter_id

        #print(np.where(interaction_mask)[0])
        #nu_id = get_cluster_label(clust_data[data_idx], np.where(interaction_mask)[0], column=8)
        nu_id, counts = np.unique(clust_data[data_idx][interaction_mask, 8], return_counts=True)
        nu_id = nu_id[np.argmax(counts)]
        # if len(nu_id) > 1:
        #     raise Exception("Interaction has > 1 nu id !")
        # else:
        #     nu_id = nu_id[0]

        # We only want to process MPV/Neutrino true interactions
        if nu_id < 1: continue

        # Identify true primary particles
        primary_count = 0
        primary_particles = []
        #print(np.unique(kinematics[data_idx][:, 12], return_counts=True))
        for part_id in np.unique(clust_data[data_idx][interaction_mask, 6]):
            particle = particles_asis[data_idx][int(part_id)]
            particle_mask = interaction_mask & (clust_data[data_idx][:, 6] == part_id)
            #print(inter_id, particle.creation_process(), particle.nu_current_type(), particle.nu_interaction_type())
            is_primary = get_cluster_label(kinematics[data_idx], [np.where(particle_mask)[0]], column=12)
            #print(part_id, particle.pdg_code(), is_primary)
            pdg = particle.pdg_code()
            if is_primary > 0:
                primary_count += 1
                primary_particles.append((particle, np.where(particle_mask)[0]))
                #print('true primary particle', particle.pdg_code(), len(np.where(particle_mask)[0]))
        #print("interaction has primaries = ", primary_count)

        # Loop over true primary particles and match to predicted primary particle
        matched_primaries_count = 0
        for p, part in primary_particles:
            part_idx = np.unique(clust_data[data_idx][part, 6])[0]
            part_type = np.unique(clust_data[data_idx][part, 9])[0]
            true_seg = np.unique(clust_data[data_idx][part, 10])[0]
            matched_pred_part = None
            matched_pred_idx = -1
            max_intersection = 0
            for pred_idx, pred_part in pred_primary_particles:
                intersection = np.intersect1d(part, pred_part)
                if len(intersection) > max_intersection:
                    max_intersection = len(intersection)
                    matched_pred_part = pred_part
                    matched_pred_idx = pred_idx

            num_pred_voxels = -1
            pred_type = -1
            pred_seg = -1
            sum_pred_voxels = -1
            if max_intersection > min_overlap_count:
                matched_primaries_count += 1
                num_pred_voxels = len(matched_pred_part)
                pred_type = np.argmax(node_pred_type[data_idx][matched_pred_idx])
                pred_seg = particles_seg[data_idx][matched_pred_idx]
                #sum_pred_voxels = clust_data[data_idx][matched_pred_part, 4].sum()
                #print('matching ', matched_pred_idx, part_idx, part_type)
            row_names_primaries.append(("inter_id", "true_id", "num_true_voxels", "num_pred_voxels",
                                        "overlap", "true_type", "pred_type", "true_pdg", "pred_id",
                                        "true_seg", "pred_seg", "sum_true_voxels", #"sum_pred_voxels",
                                        "energy_deposit", "energy_init"))
            row_values_primaries.append((inter_id, part_idx, len(part), num_pred_voxels,
                                        max_intersection, part_type, pred_type, p.pdg_code(), matched_pred_idx,
                                        true_seg, pred_seg, clust_data[data_idx][part, 4].sum(), #sum_pred_voxels,
                                        p.energy_deposit(), p.energy_init()))

        row_names_true.append(("inter_id", "num_particles", "num_voxels", "sum_voxels",
                            "num_primary_particles", "num_matched_primary_particles", "num_pred_particles", "num_pred_primary_particles"))
        row_values_true.append((inter_id, len(np.unique(clust_data[data_idx][interaction_mask, 6])), np.count_nonzero(interaction_mask), np.sum(clust_data[data_idx][interaction_mask, 4]),
                            primary_count, matched_primaries_count, len(particles[data_idx]), pred_primary_count))

    # Loop over predicted interactions
    # row_names_pred, row_values_pred = [], []
    # for inter_id in np.unique(inter_group_pred[data_idx]):
    #     interaction_mask = inter_group_pred[data_idx] == inter_id
    #     current_interaction = particles[data_idx][interaction_mask]
    #     current_types = np.argmax(node_pred_type[data_idx][interaction_mask], axis=1)
    #     current_particles_seg = particles_seg[data_idx][interaction_mask]

        # # Require >= 1 electron shower
        # keep1 = (shower_label in current_particles_seg) and (electron_label in current_types[current_particles_seg == shower_label])
        # # Require >= 1 proton track
        # keep2 = (track_label in current_particles_seg) and (proton_label in current_types[current_particles_seg == track_label])
        # # Require among predicted primaries exactly 1 electron shower and >= 1 proton track
        # primaries = np.argmax(node_pred_vtx[data_idx][interaction_mask, 3:], axis=1)
        # keep3 = (primaries[(current_types == electron_label) & (current_particles_seg == shower_label)] == 1).sum() == 1
        # keep4 = (primaries[(current_types == proton_label) & (current_particles_seg == track_label)] == 1).sum() >= 1
        #
        # row_names_pred.append(("inter_id", "num_particles", "keep",
        #                         "require_electron_shower", "require_proton_track", "require_primary_electron", "require_primary_proton"))
        # row_values_pred.append((inter_id, len(current_interaction), keep1 and keep2 and keep3 and keep4,
        #                         keep1, keep2, keep3, keep4))

    return [(row_names_true, row_values_true), (row_names_primaries, row_values_primaries)]
