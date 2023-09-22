import numpy as np
import copy
from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import *
from collections import OrderedDict

from mlreco.utils.globals import *


def reconstruct_images(data, result, mode='true_to_pred', **kwargs):
    
    if mode == 'true_to_pred':
        return reconstruct_images_t2r(data, result, **kwargs)
    elif mode == 'pred_to_true':
        return reconstruct_images_r2t(data, result, **kwargs)
    else:
        raise ValueError("Mode {} is not supported!".format(mode))


@write_to(['reco_output'])
def reconstruct_images_t2r(data, result, **kwargs):
    
    reco_output = []
    
    # matching_mode = kwargs.get('matching_mode', 'true_to_pred')
    
    data_mode             = kwargs.get('data_mode', False)
    proton_KE_threshold   = kwargs.get('proton_KE_threshold', 40)
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})
    units                 = kwargs.get('units', 'px')
    meta                  = data['meta'][0]
    
    
    for i, index in enumerate(data['index']):
        
        # Build particle matching dictionary
        particle_match_dict = {}
        for mparts in result['matched_particles'][i]:
            assert (type(mparts[0]) is TruthParticle)
            true_p, reco_p = mparts[0], mparts[1]
            particle_match_dict[true_p.id] = reco_p
            
        # For saving per image information
        index_dict = OrderedDict({
            'Index': index,
            # 'run': data_blob['run_info'][idx][0],
            # 'subrun': data_blob['run_info'][idx][1],
            # 'event': data_blob['run_info'][idx][2]
        })
            
        matched_interactions, icounts = result['matched_interactions'][i], result['interaction_match_overlap'][i]
            
        interaction_logger = InteractionLogger(int_fieldnames, meta=meta, units=units)
        interaction_logger.prepare()
            
        # Loop over true interactions
        for i, mints in enumerate(matched_interactions):
            
            value = icounts[i]
            
            assert (type(mints[0]) is TruthInteraction)
            true_int, pred_int = mints[0], mints[1]

            true_int_dict = interaction_logger.produce(true_int, mode='true')
            pred_int_dict = interaction_logger.produce(pred_int, mode='reco')
            
            index_dict['interaction_match_value'] = value
            
            index_dict.update(true_int_dict)
            index_dict.update(pred_int_dict)
            
            particle_logger = ParticleLogger(particle_fieldnames, 
                                             meta=meta, units=units)
            particle_logger.prepare()
            
            for true_p in true_int.particles:
                update_dict = copy.deepcopy(index_dict)
                assert type(true_p) is TruthParticle
                if len(true_p.match) > 0:
                    match_id, match_value = true_p.match[0], true_p.match_overlap[0]
                    pred_p = particle_match_dict[true_p.id]
                    assert type(pred_p) is Particle
                elif len(true_p.match) == 0:
                    match_id, match_value, pred_p = -1, -1, None
                else:
                    msg = "Performing true to reco matching but encountered "\
                        "two matches for one true particle. (This should not happen)"
                    raise RuntimeError(msg)
                true_p_dict = particle_logger.produce(true_p, mode='true')
                pred_p_dict = particle_logger.produce(pred_p, mode='reco')
                
                update_dict['particle_match_value'] = match_value
                update_dict.update(true_p_dict)
                update_dict.update(pred_p_dict)
                
                reco_output.append(update_dict)
            
        
    return [reco_output]


@write_to(['reco_output'])
def reconstruct_images_r2t(data, result, **kwargs):
    
    reco_output = []
    
    # matching_mode = kwargs.get('matching_mode', 'true_to_pred')
    
    data_mode             = kwargs.get('data_mode', False)
    proton_KE_threshold   = kwargs.get('proton_KE_threshold', 40)
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})
    units                 = kwargs.get('units', 'px')
    meta                  = data['meta'][0]
    
    
    for i, index in enumerate(data['index']):
        
        # Build particle matching dictionary
        particle_match_dict = {}
        for mparts in result['matched_particles'][i]:
            assert (type(mparts[0]) is Particle)
            pred_p, true_p = mparts[0], mparts[1]
            particle_match_dict[pred_p.id] = true_p
            
        # For saving per image information
        index_dict = OrderedDict({
            'Index': index,
            # 'run': data_blob['run_info'][idx][0],
            # 'subrun': data_blob['run_info'][idx][1],
            # 'event': data_blob['run_info'][idx][2]
        })
            
        matched_interactions, icounts = result['matched_interactions'][i], result['interaction_match_overlap'][i]
            
        interaction_logger = InteractionLogger(int_fieldnames, meta=meta, units=units)
        interaction_logger.prepare()
            
        # Loop over true interactions
        for i, mints in enumerate(matched_interactions):
            
            value = icounts[i]
            
            assert (type(mints[0]) is Interaction)
            pred_int, true_int = mints[0], mints[1]

            true_int_dict = interaction_logger.produce(true_int, mode='true')
            pred_int_dict = interaction_logger.produce(pred_int, mode='reco')
            
            index_dict['interaction_match_value'] = value
            
            index_dict.update(true_int_dict)
            index_dict.update(pred_int_dict)
            
            particle_logger = ParticleLogger(particle_fieldnames, 
                                             meta=meta, units=units)
            particle_logger.prepare()
            
            for pred_p in pred_int.particles:
                update_dict = copy.deepcopy(index_dict)
                assert type(pred_p) is Particle
                if len(pred_p.match) > 0:
                    match_id, match_value = pred_p.match[0], pred_p.match_overlap[0]
                    true_p = particle_match_dict[pred_p.id]
                    assert type(true_p) is TruthParticle
                elif len(pred_p.match) == 0:
                    match_id, match_value, true_p = -1, -1, None
                else:
                    msg = "Performing true to reco matching but encountered "\
                        "two matches for one true particle. (This should not happen)"
                    raise RuntimeError(msg)
                true_p_dict = particle_logger.produce(true_p, mode='true')
                pred_p_dict = particle_logger.produce(pred_p, mode='reco')
                
                update_dict['particle_match_value'] = match_value
                update_dict.update(true_p_dict)
                update_dict.update(pred_p_dict)
                
                reco_output.append(update_dict)
            
        
    return [reco_output]
