from collections import OrderedDict
from turtle import update
from analysis.classes.ui import FullChainEvaluator

from analysis.decorator import evaluate
from analysis.classes.particle import match

from pprint import pprint
import time


@evaluate(['interactions_test'], mode='per_batch')
def test_selection(data_blob, res, data_idx, analysis_cfg, cfg):

    # Set default fieldnames and values. (Needed for logger to work)
    particles_pt, particles_tp = [], []
    interactions_pt, interactions_tp = [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['match_primaries']

    predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)
    image_idxs = data_blob['index']

    # Match True interactions to Predicted interactions

    for i, index in enumerate(image_idxs):

        # print('-------------------Index: {}---------------------'.format(index))

        matches = predictor.match_interactions(i, mode='tp', match_particles=True, primaries=primaries)

        for interaction_pair in matches:
            true_int, pred_int = interaction_pair[0], interaction_pair[1]
            if pred_int is not None:
                pred_particles, true_particles = pred_int.particles, true_int.particles
            else:
                pred_particles, true_particles = [], true_int.particles

            # Match true particles to predicted particles
            matched_particles, _, _ = match(true_particles, pred_particles, 
                                            primaries=primaries, min_overlap_count=10)
            

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
                        'interaction_matched': False,
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
                        'num_particles': 0,
                        'num_true_particles': len(true_int.particles)})
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
                        'interaction_matched': False,
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
                        'num_particles': 0,
                        'num_true_particles': 0})
                    
                    update_dict['pred_interaction_id'] = pred_int.id
                    update_dict['true_interaction_id'] = true_int.id
                    
                    p1, p2 = m

                    if p2 is not None:
                        p2.is_matched = True
                        update_dict['pred_particle_type'] = p2.pid
                        update_dict['pred_particle_size'] = p2.size
                        update_dict['pred_particle_is_primary'] = p2.is_primary
                        update_dict['pred_particle_is_matched'] = True
                        update_dict['true_particle_is_matched'] = True

                    update_dict['true_particle_type'] = p1.pid
                    update_dict['true_particle_size'] = p1.size
                    update_dict['true_particle_is_primary'] = p1.is_primary

                    update_dicts.append(update_dict)



                for d in update_dicts:
                    d['pred_count_primary_leptons'] = sum(pred_count_primary_leptons.values())
                    d['pred_count_primary_particles'] = sum(pred_count_primary_particles.values())
                    interactions_tp.append(d)

    return [interactions_tp]