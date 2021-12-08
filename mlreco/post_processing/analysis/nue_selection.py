import numpy as np
import scipy

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.vertex import predict_vertex, get_vertex
from mlreco.utils.groups import type_labels

def get_predicted_primary_particles(res, data_idx=0):
    """
    Returns a list of predicted primary particles.

    Each element in the list is a tuple:
    - index of the predicted particles (within all predicted particles)
    - list of voxel indices for this particle
    """
    particles = res['particles']
    node_pred_vtx = res['node_pred_vtx']

    pred_primary_particles = []
    #print(np.amax(particles[data_idx]), clust_data[data_idx].shape)
    for pred_idx, pred_part in enumerate(particles[data_idx]):
        is_primary = np.argmax(node_pred_vtx[data_idx][pred_idx, 3:])
        if not is_primary: continue
        pred_primary_particles.append((pred_idx, pred_part))
    return pred_primary_particles

def get_true_primary_particles(clust_data, particles_asis, inter_id,
                            data_idx=0,
                            min_particle_voxel_count=20,
                            clust_data_noghost=None):
    """
    Returns a list of true primary particles.

    Each element in the list is a tuple:
    - larcv::Particle corresponding to the particle
    - list of voxel indices for this particle
    """
    interaction_mask = clust_data[data_idx][:, 7] == inter_id
    primary_particles = []
    #print(np.unique(kinematics[data_idx][:, 12], return_counts=True))
    for part_id in np.unique(clust_data[data_idx][interaction_mask, 6]):
        # Ignore -1 - double check, I don't think that is possible
        # if part_id < 0:
        #     continue
        particle = particles_asis[data_idx][int(part_id)]
        particle_mask = interaction_mask & (clust_data[data_idx][:, 6] == part_id)

        is_primary = particle.group_id() == particle.parent_id()
        #pdg = particle.pdg_code()
        # List of voxel indices for this particle
        particle_idx = np.where(particle_mask)[0]

        # Same but using true non-ghost labels
        if clust_data_noghost is not None:
            particle_noghost_idx = np.where((clust_data_noghost[data_idx][:, 7] == inter_id) & (clust_data_noghost[data_idx][:, 6] == part_id))[0]
        else:
            particle_noghost_idx = particle_idx

        # Only record if primary + above voxel count threshold
        if is_primary and len(particle_noghost_idx) > min_particle_voxel_count:
            primary_particles.append((particle, particle_idx))
    return primary_particles


def match(primary_particles, pred_primary_particles, min_overlap_count=1):
    """
    Input:
    - array of N true particles (list of voxel indices)
    - array of M predicted particles (list of voxel indices)

    Returns:
    - array of N matched predicted particles, contains index
    of matched particle within [0, M] for each true particle.
    - array of N values, contains overlap for each true particle
    with matched particle.

    Entries where the overlap is < min_overlap_count get assigned -1.
    """
    overlap_matrix = np.zeros((len(primary_particles), len(pred_primary_particles)), dtype=np.int64)

    for i, part in enumerate(primary_particles):
        for j, pred_part in enumerate(pred_primary_particles):
            overlap_matrix[i, j] = len(np.intersect1d(part, pred_part))

    idx = overlap_matrix.argmax(axis=1)
    intersections = overlap_matrix.max(axis=1)

    idx[intersections < min_overlap_count] = -1
    intersections[intersections < min_overlap_count] = -1

    return idx, intersections

def match_interactions(clust_data, inter_id, inter_group_pred, particles,
                    data_idx=0, min_overlap_count=1):
    """
    Use match (overlap) function to match a true interaction (defined by
    inter_id) with a predicted interaction.

    Returns:
    - index of matched predicted interaction (within inter_group_pred)
    - overlap count
    (-1 if no match was found)
    """
    interaction_mask = clust_data[data_idx][:, 7] == inter_id
    inter_group_pred_idx = inter_group_pred[data_idx][inter_group_pred[data_idx]>-1]
    matched_inter_idx, matched_inter_overlap = match([np.where(interaction_mask)[0]],
        [np.hstack(particles[data_idx][inter_group_pred[data_idx] == iid]) for iid in np.unique(inter_group_pred_idx)],
        min_overlap_count=min_overlap_count)

    idx = matched_inter_idx[0]
    intersection = matched_inter_overlap[0]

    if idx > -1:
        idx = inter_group_pred_idx[idx]

    return idx, intersection

@post_processing(['nue-selection-true', 'nue-selection-primaries'],
                ['input_data', 'seg_label', 'clust_data', 'particles_asis', 'kinematics'],
                ['segmentation', 'inter_group_pred', 'particles', 'particles_seg', 'node_pred_type', 'node_pred_vtx'])
def nue_selection(cfg, module_cfg, data_blob, res, logdir, iteration,
                data_idx=None, input_data=None, clust_data=None, particles_asis=None, kinematics=None,
                inter_group_pred=None, particles=None, particles_seg=None,
                node_pred_type=None, node_pred_vtx=None, clust_data_noghost=None, **kwargs):
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
    spatial_size             = module_cfg.get('spatial_size', 768)
    shower_label             = module_cfg.get('shower_label', 0)
    track_label              = module_cfg.get('track_label', 1)
    electron_label           = module_cfg.get('electron_label', 1)
    proton_label             = module_cfg.get('proton_label', 4)
    min_overlap_count        = module_cfg.get('min_overlap_count', 10)
    # Minimum voxel count for a true non-ghost particle to be considered
    min_particle_voxel_count = module_cfg.get('min_particle_voxel_count', 20)
    # We want to count how well we identify interactions with some PDGs
    # as primary particles
    primary_pdgs             = np.unique(module_cfg.get('primary_pdgs', []))
    attaching_threshold      = module_cfg.get('attaching_threshold', 2)
    inter_threshold          = module_cfg.get('inter_threshold', 10)

    # Translate into particle type labels
    primary_types = np.unique([type_labels[pdg] for pdg in primary_pdgs])

    row_names_true, row_values_true = [], []
    row_names_primaries, row_values_primaries = [], []

    #
    # 1. Find predicted primary particles in the event
    #
    pred_primary_particles = get_predicted_primary_particles(res, data_idx=data_idx)
    pred_primary_count = len(pred_primary_particles)

    #
    # 2. Loop over true interactions
    #
    for inter_id in np.unique(clust_data[data_idx][:, 7]):
        if inter_id == -1:
            continue
        interaction_mask = clust_data[data_idx][:, 7] == inter_id
        nu_id = get_cluster_label(clust_data[data_idx], [np.where(interaction_mask)[0]], column=8)[0]

        #
        # IF we only want to process MPV/Neutrino true interactions
        #
        # if nu_id < 1: continue

        #
        # 3. Identify true primary particles
        #
        primary_particles = get_true_primary_particles(clust_data, particles_asis, inter_id, data_idx=data_idx)
        primary_count = len(primary_particles)

        #
        # Select interactions that have
        # - at least 1 lepton among primary particles
        # - and at least 2 total primary particles
        # Note: this cut can be done in later analysis stage.
        #
        # if primary_count < 2:
        #     continue

        #
        # 4. Loop over true primary particles and match to predicted primary particle
        # Also record PID confusion matrix
        #
        matched_primaries_count = 0
        confusion_matrix = np.zeros((5, 5), dtype=np.int64)
        # Record count of certain primary PDGs
        true_primary_pdgs, pred_primary_pdgs = {}, {}
        for pdg in primary_pdgs:
            true_primary_pdgs[pdg] = 0
        for pdg in primary_types:
            pred_primary_pdgs[pdg] = 0

        matched_idx, matched_overlap = match([p for _, p in primary_particles],
                                            [p for _, p in pred_primary_particles],
                                            min_overlap_count=min_overlap_count)
        for i, (p, part) in enumerate(primary_particles):
            if int(p.pdg_code()) in primary_pdgs:
                true_primary_pdgs[int(p.pdg_code())] += 1
            # Record energy if electron/proton is contained
            is_contained = p.position().x() >= 0 and p.position().x() <= spatial_size \
                        and p.position().y() >= 0 and p.position().y() <= spatial_size \
                        and p.position().z() >= 0 and p.position().z() <= spatial_size \
                        and p.end_position().x() >= 0 and p.end_position().x() <= spatial_size \
                        and p.end_position().y() >= 0 and p.end_position().y() <= spatial_size \
                        and p.end_position().z() >= 0 and p.end_position().z() <= spatial_size
            voxels = clust_data[data_idx][part][:, 1:4]
            true_length = scipy.spatial.distance.cdist(voxels, voxels).max()

            part_idx = get_cluster_label(clust_data[data_idx], [part], column=6)[0]
            part_type = get_cluster_label(clust_data[data_idx], [part], column=9)[0]
            true_seg = get_cluster_label(clust_data[data_idx], [part], column=10)[0]

            matched_pred_idx = matched_idx[i]
            max_intersection = matched_overlap[i]

            num_pred_voxels = -1
            pred_type = -1
            pred_seg = -1
            sum_pred_voxels = -1
            pred_length = -1
            matched_pred_part = []
            if matched_pred_idx > -1:
                # We found a match for this particle.
                matched_pred_part = pred_primary_particles[matched_pred_idx][1]

                matched_primaries_count += 1
                num_pred_voxels = len(matched_pred_part)
                pred_type = np.argmax(node_pred_type[data_idx][matched_pred_idx])
                pred_seg = particles_seg[data_idx][matched_pred_idx]

                voxels = clust_data[data_idx][matched_pred_part][:, 1:4]
                pred_length = scipy.spatial.distance.cdist(voxels, voxels).max()

                if pred_type in primary_types:
                    pred_primary_pdgs[int(pred_type)] += 1
                #sum_pred_voxels = clust_data[data_idx][matched_pred_part, 4].sum()
                #print('matching ', matched_pred_idx, part_idx, part_type)
                confusion_matrix[part_type, pred_type] += 1

            row_names_primaries.append(("inter_id", "true_id", "num_true_voxels", "num_pred_voxels",
                                        "overlap", "true_type", "pred_type", "true_pdg", "pred_id",
                                        "true_seg", "pred_seg", "sum_true_voxels", #"sum_pred_voxels",
                                        "energy_deposit", "energy_init", "is_contained",
                                        "true_length", "pred_length"))
            row_values_primaries.append((inter_id, part_idx, len(part), num_pred_voxels,
                                        max_intersection, part_type, pred_type, p.pdg_code(), matched_pred_idx,
                                        true_seg, pred_seg, clust_data[data_idx][part, 4].sum(), #sum_pred_voxels,
                                        p.energy_deposit(), p.energy_init(), is_contained,
                                        true_length, pred_length))

        #
        # 5. Loop over predicted interactions and
        # find a match for current true interaction
        #
        matched_inter_id, max_overlap = match_interactions(clust_data, inter_id, inter_group_pred, particles, data_idx=data_idx)

        matched_inter_num_voxels = -1
        vtx_resolution = -1
        if matched_inter_id > -1:
            matched_inter = np.hstack(particles[data_idx][inter_group_pred[data_idx] == matched_inter_id])
            matched_inter_num_voxels = len(matched_inter)

            #
            # Compute vertex prediction performance
            #
            if nu_id >= 1:
                ppn_candidates, c_candidates, vtx_candidate, vtx_std = predict_vertex(matched_inter_id, data_idx, input_data, res,
                                                                                    attaching_threshold=attaching_threshold,
                                                                                    inter_threshold=inter_threshold)
                vtx = get_vertex(kinematics, clust_data, data_idx, inter_id)
                #print(vtx, vtx_candidate, len(ppn_candidates))
                if len(ppn_candidates):
                    vtx_resolution = np.linalg.norm(vtx_candidate-vtx)
                else:
                    vtx_resolution = np.nan

        #
        # Record PID confusion matrix
        #
        # print(confusion_matrix)

        #
        # Energy reconstruction (simple study)
        #

        # Recording info to CSV file
        row_names_true.append(("inter_id", "nu_id", "num_true_particles", "num_voxels", "sum_voxels",
                            "num_true_primary_particles", "num_matched_primary_particles", "num_pred_particles", "num_pred_primary_particles",) \
                            + tuple(["true_pdg_%d" % pdg for pdg in true_primary_pdgs]) \
                            + tuple(["pred_type_%d" % type for type in pred_primary_pdgs]) \
                            + ("overlap", "matched_inter_num_voxels", "vtx_resolution") \
                            + tuple(["pid_confusion_%d_%d" % (i, j) for i in range(5) for j in range(5)]))
        row_values_true.append((inter_id, nu_id, len(np.unique(clust_data[data_idx][interaction_mask, 6])), np.count_nonzero(interaction_mask), np.sum(clust_data[data_idx][interaction_mask, 4]),
                            primary_count, matched_primaries_count, len(particles[data_idx]), pred_primary_count,) \
                            + tuple([true_primary_pdgs[pdg] for pdg in true_primary_pdgs]) \
                            + tuple([pred_primary_pdgs[type] for type in pred_primary_pdgs]) \
                            + (max_overlap, matched_inter_num_voxels, vtx_resolution) \
                            + tuple([confusion_matrix[i, j] for i in range(5) for j in range(5)]))

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
