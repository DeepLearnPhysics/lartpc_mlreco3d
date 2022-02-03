from collections import OrderedDict
from analysis.classes.ui import FullChainEvaluator

from analysis.decorator import evaluate
from analysis.classes.particle import match

from pprint import pprint


@evaluate(['nue_selection_pt', 'nue_selection_tp'], mode='per_batch')
def nue_selection(data_blob, res, data_idx, analysis_cfg, module_config):

    # Set default fieldnames and values. (Needed for logger to work)
    fields = OrderedDict(analysis_cfg['analysis']['fields'])
    pred_to_truth, truth_to_pred = [], []
    deghosting = analysis_cfg['analysis']['deghosting']

    predictor = FullChainEvaluator(data_blob, res, module_config, analysis_cfg, deghosting=deghosting)
    image_idxs = data_blob['index']

    # 1. Get Pred -> True matching (for each predicted particle, match one of truth)
    for i, index in enumerate(image_idxs):

        matches = predictor.match_interactions(i, mode='pt', match_particles=False)
        
        for interaction_pair in matches:
            pred_int, true_int = interaction_pair[0], interaction_pair[1]

            if true_int is not None:
                pred_particles, true_particles = pred_int.particles, true_int.particles
            else:
                pred_particles, true_particles = pred_int.particles, []

            matched_particles, _, _ = match(pred_particles, true_particles, 
                                            primaries=True, min_overlap_count=1)

            pred_int_dict = pred_int.get_info()
            true_int_dict = {}
            if true_int is not None:
                true_int_dict = true_int.get_info()

            for ppair in matched_particles:
                # print("    ", ppair[0], ppair[1])

                pred_p, true_p = ppair[0], ppair[1]

                true_particle_dict = {}
                pred_particle_dict = pred_p.get_info()
                pred_particle_is_matched = len(pred_p.match) > 0
                true_particle_is_matched = False

                if true_p is not None:
                    true_particle_dict = true_p.get_info()
                    true_particle_is_matched = len(true_p.match) > 0

                update_dict = OrderedDict(fields)
                update_dict['index'] = index
                update_dict.update(pred_int_dict)
                update_dict.update(true_int_dict)
                update_dict.update(pred_particle_dict)
                update_dict.update(true_particle_dict)

                update_dict['pred_particle_is_matched'] = pred_particle_is_matched
                update_dict['true_particle_is_matched'] = true_particle_is_matched

                update_dict = OrderedDict(sorted(update_dict.items()))

                pred_to_truth.append(update_dict)

    # Match True interactions to Predicted interactions

    for i, index in enumerate(image_idxs):

        matches = predictor.match_interactions(i, mode='tp')
        
        for interaction_pair in matches:
            true_int, pred_int = interaction_pair[0], interaction_pair[1]
            if pred_int is not None:
                pred_particles, true_particles = pred_int.particles, true_int.particles
            else:
                pred_particles, true_particles = [], true_int.particles

            # Match true particles to predicted particles
            matched_particles, _, _ = match(true_particles, pred_particles, 
                                            primaries=True, min_overlap_count=1)

            true_int_dict = true_int.get_info()
            pred_int_dict = {}
            if pred_int is not None:
                pred_int_dict = pred_int.get_info()

            for ppair in matched_particles:

                true_p, pred_p = ppair[0], ppair[1]

                pred_particle_dict = {}
                true_particle_dict = true_p.get_info()
                true_particle_is_matched = len(true_p.match) > 0
                pred_particle_is_matched = False

                if pred_p is not None:
                    pred_particle_dict = pred_p.get_info()
                    pred_particle_is_matched = len(pred_p.match) > 0

                update_dict = OrderedDict(fields)
                update_dict['index'] = index
                update_dict.update(pred_int_dict)
                update_dict.update(true_int_dict)
                update_dict.update(pred_particle_dict)
                update_dict.update(true_particle_dict)

                update_dict['pred_particle_is_matched'] = pred_particle_is_matched
                update_dict['true_particle_is_matched'] = true_particle_is_matched
                update_dict = OrderedDict(sorted(update_dict.items()))
                truth_to_pred.append(update_dict)


    return [pred_to_truth, truth_to_pred]