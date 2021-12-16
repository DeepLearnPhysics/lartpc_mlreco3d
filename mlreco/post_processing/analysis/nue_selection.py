
import numpy as np
import scipy

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.vertex import predict_vertex, get_vertex
from mlreco.utils.groups import type_labels

from mlreco.analysis.ui import FullChainEvaluator
from mlreco.analysis.particle import NullInteraction, NullParticle, match
from pprint import pprint


@post_processing(['nu-selection-pt', 'nu-selection-tp'],
                ['input_data', 'seg_label', 'clust_data', 'particles_asis', 'kinematics'],
                ['segmentation', 'inter_group_pred', 'particles', 'particles_seg', 'node_pred_type', 'node_pred_vtx'])
def nue_selection(cfg, processor_cfg, data_blob, res, logdir, iteration,
                data_idx=None, input_data=None, clust_data=None, particles_asis=None, kinematics=None,
                inter_group_pred=None, particles=None, particles_seg=None,
                node_pred_type=None, node_pred_vtx=None, clust_data_noghost=None, **kwargs):

    spatial_size = processor_cfg.get('spatial_size', 768)
    row_names_pt, row_values_pt = [], []
    predictor = FullChainEvaluator(None, data_blob, res, cfg, processor_cfg)

    # 1. Get Pred -> True matching (for each predicted particle, match one of truth)

    matches, _, _ = predictor.match_interactions(data_idx, mode='pt')

    index = predictor.index[data_idx]
    for pair in matches:
        # print("Particle: ", pair[0])
        pred_int, true_int = pair[0], pair[1]

        pred_inter_names, pred_inter_values = pred_int.get_names_and_values()
        true_inter_names, true_inter_values = true_int.get_names_and_values()

        if isinstance(true_int, NullInteraction):
            # Fill nans if no match
            for pred_p in pred_int.particles:
                true_p = NullParticle()
                pred_particle_names, pred_particle_values = pred_p.get_names_and_values()
                true_particle_names, true_particle_values = true_p.get_names_and_values()
        else:
            parts, true_parts = pair[0].particles, pair[1].particles

            matched_particles, _, _ = match(parts, true_parts, 
                primaries=True, min_overlap_count=10, mode='particles')

            for ppair in matched_particles:

                pred_p, true_p = ppair[0], ppair[1]

                # print("    ", pred_p)
                # print("    ", true_p)

                pred_particle_names, pred_particle_values = pred_p.get_names_and_values()
                true_particle_names, true_particle_values = true_p.get_names_and_values()

        row_names_pt.append(
            tuple(['Index']
            + pred_inter_names + true_inter_names \
            + pred_particle_names + true_particle_names))

        row_values_pt.append(
            tuple([index]
            + pred_inter_values + true_inter_values \
            + pred_particle_values + true_particle_values))


    # 2. Get True -> Pred matching (for each true particle, match one of predictions)

    row_names_tp, row_values_tp = [], []

    matches, _, _ = predictor.match_interactions(data_idx, mode='tp')

    index = predictor.index[data_idx]
    for pair in matches:
        # print("Particle: ", pair[0])
        # The tuple order (pred, truth) remains the same
        pred_int, true_int = pair[0], pair[1]

        pred_inter_names, pred_inter_values = pred_int.get_names_and_values()
        true_inter_names, true_inter_values = true_int.get_names_and_values()

        if isinstance(true_int, NullInteraction):
            # Fill nans if no match
            for pred_p in pred_int.particles:
                true_p = NullParticle()
                pred_particle_names, pred_particle_values = pred_p.get_names_and_values()
                true_particle_names, true_particle_values = true_p.get_names_and_values()
        else:
            parts, true_parts = pair[0].particles, pair[1].particles

            matched_particles, _, _ = match(parts, true_parts, 
                primaries=True, min_overlap_count=10, mode='particles')

            for ppair in matched_particles:

                pred_p, true_p = ppair[0], ppair[1]

                pred_particle_names, pred_particle_values = pred_p.get_names_and_values()
                true_particle_names, true_particle_values = true_p.get_names_and_values()

        row_names_tp.append(
            tuple(['Index']
            + pred_inter_names + true_inter_names \
            + pred_particle_names + true_particle_names))

        row_values_tp.append(
            tuple([index]
            + pred_inter_values + true_inter_values \
            + pred_particle_values + true_particle_values))

    return [(row_names_pt, row_values_pt), (row_names_tp, row_values_tp)]
