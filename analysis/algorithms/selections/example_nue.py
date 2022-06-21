from collections import OrderedDict
from analysis.algorithms.utils import count_primary_particles, get_particle_properties
from analysis.classes.ui import FullChainEvaluator

from analysis.decorator import evaluate
from analysis.classes.particle import match_particles_fn, matrix_iou

from pprint import pprint
import time


@evaluate(['interactions', 'particles'], mode='per_batch')
def debug_pid(data_blob, res, data_idx, analysis_cfg, cfg):

    interactions, particles = [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['match_primaries']

    predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg)
    image_idxs = data_blob['index']
    for i, index in enumerate(image_idxs):

        # Process Interaction Level Information
        matches, counts = predictor.match_interactions(i,
            mode='true_to_pred',
            match_particles=True,
            drop_nonprimary_particles=primaries,
            return_counts=True)

        for i, interaction_pair in enumerate(matches):
            true_int, pred_int = interaction_pair[0], interaction_pair[1]
            true_int_dict = count_primary_particles(true_int, prefix='true')
            pred_int_dict = count_primary_particles(pred_int, prefix='pred')
            if pred_int is None:
                pred_int_dict['true_interaction_matched'] = False
            else:
                pred_int_dict['true_interaction_matched'] = True
            true_int_dict['true_nu_id'] = true_int.nu_id
            pred_int_dict['interaction_match_counts'] = counts[i]
            interactions_dict = OrderedDict({'Index': index})
            interactions_dict.update(true_int_dict)
            interactions_dict.update(pred_int_dict)
            interactions.append(interactions_dict)

            # Process particle level information
            pred_particles, true_particles = [], true_int.particles
            if pred_int is not None:
                pred_particles = pred_int.particles
            matched_particles, _, ious = match_particles_fn(true_particles,
                                                            pred_particles)
            for i, m in enumerate(matched_particles):
                particles_dict = OrderedDict({'Index': index})
                true_p, pred_p = m[0], m[1]
                pred_particle_dict = get_particle_properties(pred_p,
                    vertex=pred_int.vertex,
                    prefix='pred')
                true_particle_dict = get_particle_properties(true_p,
                    vertex=true_int.vertex,
                    prefix='true')
                if pred_p is not None:
                    pred_particle_dict['true_particle_is_matched'] = True
                else:
                    pred_particle_dict['true_particle_is_matched'] = False
                pred_particle_dict['particle_match_counts'] = ious[i]

                particles_dict.update(pred_particle_dict)
                particles_dict.update(true_particle_dict)

                particles.append(particles_dict)

    return [interactions, particles]



@evaluate(['interactions_test', 'node_features', 'edge_features'], mode='per_batch')
def test_selection(data_blob, res, data_idx, analysis_cfg, cfg):

    # Set default fieldnames and values. (Needed for logger to work)
    interactions_tp = []
    node_df = []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['match_primaries']

    predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)
    image_idxs = data_blob['index']

    # Match True interactions to Predicted interactions

    for i, index in enumerate(image_idxs):

        print('-------------------Index: {}---------------------'.format(index))

        matches = predictor.match_interactions(i, mode='tp', match_particles=True, primaries=primaries)

        for interaction_pair in matches:
            true_int, pred_int = interaction_pair[0], interaction_pair[1]
            if pred_int is not None:
                pred_particles, true_particles = pred_int.particles, true_int.particles
            else:
                pred_particles, true_particles = [], true_int.particles

            # Match true particles to predicted particles
            matched_particles, _, _ = match(true_particles, pred_particles)

            if pred_int is None:
                print("No predicted interaction match = ", matched_particles)
                true_count_primary_leptons = true_int.primary_particle_counts[1] \
                                            + true_int.primary_particle_counts[2]
                true_count_primary_particles = sum(true_int.primary_particle_counts.values())
                for p in true_int.particles:
                    update_dict = OrderedDict({
                        'index': index,
                        'pred_interaction_id': -1,
                        'true_interaction_id': true_int.id,
                        'pred_particle_type': -1,
                        'pred_particle_size': -1,
                        'pred_particle_is_primary': False,
                        'pred_particle_is_matched': False,
                        'true_particle_type': p.pid,
                        'true_particle_size': p.size,
                        'true_particle_is_primary': False,
                        'true_particle_is_matched': False,
                        'pred_count_primary_leptons': 0,
                        'pred_count_primary_particles': 0,
                        'true_count_primary_leptons': true_count_primary_leptons,
                        'true_count_primary_particles': true_count_primary_particles,
                        'true_interaction_is_matched': False,
                        'pred_particle_E': -1,
                        'true_particle_E': p.sum_edep})
                    interactions_tp.append(update_dict)

            else:
                true_count_primary_leptons = true_int.primary_particle_counts[1] \
                                           + true_int.primary_particle_counts[2]
                true_count_primary_particles = sum(true_int.primary_particle_counts.values())

                pred_count_primary_leptons = {}
                pred_count_primary_particles = {}

                update_dicts = []

                for p in pred_particles:
                    if p.is_primary:
                        # Count matched primaries
                        pred_count_primary_particles[p.id] = True
                        if (p.pid == 1 or p.pid == 2):
                            # Count matched primary leptons
                            pred_count_primary_leptons[p.id] = True

                for m in matched_particles:
                    update_dict = OrderedDict({
                        'index': index,
                        'pred_interaction_id': -1,
                        'true_interaction_id': true_int.id,
                        'pred_particle_type': -1,
                        'pred_particle_size': -1,
                        'pred_particle_is_primary': False,
                        'pred_particle_is_matched': False,
                        'true_particle_type': -1,
                        'true_particle_size': -1,
                        'true_particle_is_primary': False,
                        'true_particle_is_matched': False,
                        'pred_count_primary_leptons': 0,
                        'pred_count_primary_particles': 0,
                        'true_count_primary_leptons': true_count_primary_leptons,
                        'true_count_primary_particles': true_count_primary_particles,
                        'true_interaction_is_matched': True,
                        'pred_particle_E': -1,
                        'true_particle_E': -1})

                    update_dict['pred_interaction_id'] = pred_int.id
                    update_dict['true_interaction_id'] = true_int.id

                    p1, p2 = m

                    if p2 is not None:
                        p2.is_matched = True
                        update_dict['pred_particle_type'] = p2.pid
                        update_dict['pred_particle_size'] = p2.size
                        update_dict['pred_particle_E'] = p2.sum_edep
                        update_dict['pred_particle_is_primary'] = p2.is_primary
                        update_dict['pred_particle_is_matched'] = True
                        update_dict['true_particle_is_matched'] = True

                    update_dict['true_particle_type'] = p1.pid
                    update_dict['true_particle_size'] = p1.size
                    update_dict['true_particle_is_primary'] = p1.is_primary
                    update_dict['true_particle_E'] = p1.sum_edep

                    update_dicts.append(update_dict)

                    node_dict = OrderedDict({'node_feat_{}'.format(i) : p2.node_features[i] \
                        for i in range(p2.node_features.shape[0])})

                    node_df.append(node_dict)


                for d in update_dicts:
                    d['pred_count_primary_leptons'] = sum(pred_count_primary_leptons.values())
                    d['pred_count_primary_particles'] = sum(pred_count_primary_particles.values())
                    interactions_tp.append(d)

    return [interactions_tp, node_df]
