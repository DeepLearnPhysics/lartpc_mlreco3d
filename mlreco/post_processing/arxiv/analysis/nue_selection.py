import numpy as np
import scipy

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.vertex import predict_vertex, get_vertex
from mlreco.utils.globals import PDG_TO_PID


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
    primary_types = np.unique([PDG_TO_PID[pdg] for pdg in primary_pdgs])

    row_names_true, row_values_true = [], []
    row_names_primaries, row_values_primaries = [], []

    #
    # 1. Find predicted primary particles in the event
    #
    pred_particles = []
    pred_particles_is_primary = []
    #print(np.amax(particles[data_idx]), clust_data[data_idx].shape)
    for pred_idx, pred_part in enumerate(particles[data_idx]):
        is_primary = np.argmax(node_pred_vtx[data_idx][pred_idx, 3:])
        #if not is_primary: continue
        #if len(pred_part) <= min_particle_voxel_count: continue
        pred_particles.append((pred_idx, pred_part))
        pred_particles_is_primary.append(is_primary)

    pred_particles_is_primary = np.array(pred_particles_is_primary, dtype=np.bool)
    print(pred_particles_is_primary)
    # pred_primary_count = pred_particles_is_primary.sum()

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
        # primary_count, true_particles_count = 0, 0
        true_particles = []
        true_particles_is_primary = []
        #print(np.unique(kinematics[data_idx][:, 12], return_counts=True))
        for part_id in np.unique(clust_data[data_idx][interaction_mask, 6]):
            # Ignore -1 - double check, I don't think that is possible
            # if part_id < 0:
            #     continue
            particle = particles_asis[data_idx][int(part_id)]
            particle_mask = interaction_mask & (clust_data[data_idx][:, 6] == part_id)

            is_primary = particle.group_id() == particle.parent_id()
            pdg = particle.pdg_code()
            # List of voxel indices for this particle
            particle_idx = np.where(particle_mask)[0]
            # Same but using true non-ghost labels
            particle_noghost_idx = np.where((clust_data_noghost[data_idx][:, 7] == inter_id) & (clust_data_noghost[data_idx][:, 6] == part_id))[0]
            print(data_idx, inter_id, len(particle_noghost_idx), len(particle_idx))
            # Only record if primary + above voxel count threshold
            if len(particle_noghost_idx) > min_particle_voxel_count:
                if is_primary:
                    # primary_particles.append((particle, particle_idx))
                    print(data_idx, inter_id, 'True primary particle %d with true noghost voxel count = ' % part_id, len(particle_noghost_idx), ' and predicted deghosted voxel count = ', len(particle_idx))
                else:
                    print(data_idx, inter_id, 'True particle %d with true nonghost voxel count = ' % part_id, len(particle_noghost_idx), ' and predicted deghosted voxel count = ', len(particle_idx))
                true_particles.append((particle, particle_idx))
                true_particles_is_primary.append(is_primary)

        true_particles_is_primary = np.array(true_particles_is_primary, dtype=np.bool)
        primary_count = true_particles_is_primary[[len(x[1]) > min_particle_voxel_count for x in true_particles]].sum()
        true_particles_count = len(true_particles)
        #print('primary count', primary_count)
        #
        # Select interactions that have
        # - at least 1 lepton among primary particles
        # - and at least 2 total primary particles
        # Note: this cut can be done in later analysis stage.
        #
        if true_particles_count == 0:
            print("\nSkipping interaction b/c no true particle", inter_id)
            continue

        matched_primaries_count = 0 # Count of predicted primaries that are matched to a true particle (possibly not primary)
        confusion_matrix = np.zeros((5, 5), dtype=np.int64)
        # Record count of certain primary PDGs
        true_primary_pdgs, pred_primary_pdgs = {}, {}
        for pdg in primary_pdgs:
            true_primary_pdgs[pdg] = 0
        for pdg in primary_types:
            pred_primary_pdgs[pdg] = 0

        #
        # 5. Loop over predicted interactions and
        # find a match for current true interaction
        #

        interaction_mask = np.hstack([t[1] for t in true_particles])
        #print(interaction_mask)
        matched_inter_id = -1
        matched_inter = None
        max_overlap = 0
        for iid in np.unique(inter_group_pred[data_idx]):
            # Only include predicted particles that pass the cut in the predicted interaction mask
            pred_inter_particles = [x for x in particles[data_idx][inter_group_pred[data_idx] == iid] if len(x) > min_particle_voxel_count]
            if len(pred_inter_particles) == 0: continue
            pred_interaction_mask = np.hstack(pred_inter_particles)
            #intersection = np.intersect1d(pred_interaction_mask, np.where(interaction_mask)[0])
            intersection = np.intersect1d(pred_interaction_mask, interaction_mask)
            #print('Predicted interaction ', iid, len(pred_interaction_mask), len(intersection))
            if len(intersection) > max_overlap and len(intersection) > min_overlap_count:
                max_overlap = len(intersection)
                matched_inter_id = iid
                matched_inter = pred_interaction_mask
        matched_inter_num_voxels = -1
        vtx_resolution = -1
        pred_primary_count = 0
        if matched_inter_id > -1:
            #matched_inter = np.hstack(particles[data_idx][inter_group_pred[data_idx] == matched_inter_id])
            matched_inter_num_voxels = len(matched_inter)

            #print(data_idx, inter_id, "Matching true interaction with ", matched_inter_id, len(matched_inter), max_overlap)

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
            # 4. Loop over true primary particles and match to predicted primary particle
            # Also record PID confusion matrix
            #

            matched_pred_idx_list = []
            intersections = []
            for p, part in true_particles:
                matched_pred_part = None
                matched_pred_idx = -1
                max_intersection = 0
                for p_idx in np.arange(len(pred_particles))[inter_group_pred[data_idx] == matched_inter_id]:
                    pred_idx, pred_part = pred_particles[p_idx]
                    if len(pred_part) <= min_particle_voxel_count: continue
                    intersection = np.intersect1d(part, pred_part)
                    if len(intersection) > max_intersection and len(intersection) > min_overlap_count:
                        max_intersection = len(intersection)
                        matched_pred_part = pred_part
                        matched_pred_idx = pred_idx
                intersections.append(max_intersection)
                if max_intersection > min_overlap_count:
                    # We found a match for this particle.
                    matched_pred_idx_list.append((p, part, matched_pred_idx, matched_pred_part))
                    print(data_idx, inter_id, "Matched particle with true voxel count = ", len(part),  " and predicted voxel count/type = ", len(matched_pred_part))
                else:
                    matched_pred_idx_list.append(None)

            # Now loop over true primaries to record them and their match
            for idx in np.arange(len(true_particles))[true_particles_is_primary]:
                p, part = true_particles[idx]
                if int(p.pdg_code()) in primary_pdgs and len(part) > min_particle_voxel_count:
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

                num_pred_voxels = -1
                pred_type = -1
                pred_seg = -1
                sum_pred_voxels = -1
                pred_length = -1
                if matched_pred_idx_list[idx] is not None:
                    _, _, matched_pred_idx, matched_pred_part = matched_pred_idx_list[idx]
                    num_pred_voxels = len(matched_pred_part)
                    pred_type = np.argmax(node_pred_type[data_idx][matched_pred_idx])
                    pred_seg = particles_seg[data_idx][matched_pred_idx]

                    voxels = clust_data[data_idx][matched_pred_part][:, 1:4]
                    pred_length = scipy.spatial.distance.cdist(voxels, voxels).max()
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
            # to avoid duplicate counting, we loop over unique list
            unique_matched_pred_idx_list = np.unique([x[2] for x in matched_pred_idx_list if x is not None])
            for matched_pred_idx in unique_matched_pred_idx_list:
                #pred_idx, pred_part = pred_particles[matched_pred_idx]
                if pred_particles_is_primary[matched_pred_idx]: #and len(pred_part) > min_particle_voxel_count:
                    matched_primaries_count += 1
            # Predicted primary pdg counts should be done regardless of matching status
            for p_idx in np.arange(len(pred_particles))[inter_group_pred[data_idx] == matched_inter_id]:
                pred_idx, pred_part = pred_particles[p_idx]
                print('Predicted particle %d with predicted deghosted voxel count ' % pred_idx, len(pred_part) )
                assert p_idx == pred_idx
                if len(pred_part) <= min_particle_voxel_count: continue
                pred_type = np.argmax(node_pred_type[data_idx][pred_idx])
                print(data_idx, inter_id, matched_inter_id, 'Predicted particle', pred_idx, " voxel count = ", len(pred_part), " type = ", pred_type)
                print('Is primary?', pred_particles_is_primary[pred_idx])
                if pred_particles_is_primary[pred_idx]: #and len(pred_part) > min_particle_voxel_count:
                    pred_primary_count += 1
                    if pred_type in primary_types:
                        pred_primary_pdgs[int(pred_type)] += 1

        # Logging
        pred_lepton_count = pred_primary_pdgs[2] + pred_primary_pdgs[1]
        #if len(particles[data_idx]) >= 2 and  pred_lepton_count == 1:
        true_lepton_count = true_primary_pdgs[13] + true_primary_pdgs[-13] + true_primary_pdgs[11] + true_primary_pdgs[-11]
        print(data_idx, inter_id, nu_id, "True primary count = ", primary_count, "predicted primary count = ", pred_primary_count, " true lepton count = ", true_lepton_count, "pred lepton count = ", pred_lepton_count)

        # Recording info to CSV file
        row_names_true.append(("inter_id", "nu_id", "num_true_particles", "num_voxels", "sum_voxels",
                            "num_true_primary_particles", "num_matched_primary_particles", "num_pred_particles", "num_pred_primary_particles",) \
                            + tuple(["true_pdg_%d" % pdg for pdg in true_primary_pdgs]) \
                            + tuple(["pred_type_%d" % type for type in pred_primary_pdgs]) \
                            + ("overlap", "matched_inter_num_voxels", "vtx_resolution") \
                            + tuple(["pid_confusion_%d_%d" % (i, j) for i in range(5) for j in range(5)]))
        row_values_true.append((inter_id, nu_id, true_particles_count, len(interaction_mask), np.sum(clust_data[data_idx][interaction_mask, 4]),
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
