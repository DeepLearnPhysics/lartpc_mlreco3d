import numpy as np
from collections import OrderedDict

from analysis.post_processing import post_processing
from mlreco.utils.globals import *
from analysis.classes.matching import (match_particles_fn,
                                       match_interactions_fn,
                                       weighted_matrix_iou,
                                       generate_match_pairs)
from analysis.classes.data import *

@post_processing(data_capture=['index'], 
                 result_capture=['particles',
                                 'truth_particles'])
def match_particles(data_dict,
                    result_dict,
                    matching_mode='true_to_pred',
                    min_overlap=0,
                    overlap_mode='iou',
                    weight=False,
                    list_principal_matches=True):
    pred_particles = result_dict['particles']
    
    out = {}
    
    if overlap_mode == 'chamfer':
        true_particles = [ia for ia in result_dict['truth_particles'] if ia.truth_size > 0]
    else:
        true_particles = [ia for ia in result_dict['truth_particles'] if ia.size > 0]
    
    # Only consider interactions with nonzero predicted nonghost
    matched_particles = []
    
    if matching_mode == 'pred_to_true':
        overlap_matrix, value_matrix = weighted_matrix_iou(pred_particles, true_particles, weight=weight)
        matched_particles, counts = match_particles_fn(
            pred_particles, 
            true_particles, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        matches = generate_match_pairs(true_particles, pred_particles, 
                                       prefix='matched_particles',
                                       only_principal=list_principal_matches)
        out['matched_particles_r2t'] = matches['matched_particles_r2t']
        out['particle_match_overlap_r2t'] = matches['matched_particles_r2t_values']
    elif matching_mode == 'true_to_pred':
        overlap_matrix, value_matrix = weighted_matrix_iou(true_particles, pred_particles, weight=weight)
        matched_particles, counts = match_particles_fn(
            true_particles, 
            pred_particles, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        matches = generate_match_pairs(true_particles, pred_particles, 
                                       prefix='matched_particles',
                                       only_principal=list_principal_matches)
        out['matched_particles_t2r'] = matches['matched_particles_t2r']
        out['particle_match_overlap_t2r'] = matches['matched_particles_t2r_values']
    elif matching_mode == 'both':
        overlap_matrix, value_matrix = weighted_matrix_iou(true_particles, pred_particles, weight=weight)
        matched_particles, counts = match_particles_fn(
            true_particles, 
            pred_particles, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        t2r_matches = generate_match_pairs(true_particles, pred_particles, 
                                           prefix='matched_particles',
                                           only_principal=list_principal_matches)
        out['matched_particles_t2r'] = t2r_matches['matched_particles_t2r']
        out['particle_match_overlap_t2r'] = t2r_matches['matched_particles_t2r_values']
        overlap_matrix, value_matrix = weighted_matrix_iou(pred_particles, true_particles, weight=weight)
        matched_particles, counts = match_particles_fn(
            pred_particles, 
            true_particles, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        r2t_matches = generate_match_pairs(true_particles, pred_particles, 
                                           prefix='matched_particles',
                                           only_principal=list_principal_matches)
        out['matched_particles_r2t'] = r2t_matches['matched_particles_r2t']
        out['particle_match_overlap_r2t'] = r2t_matches['matched_particles_r2t_values']
    else:
        raise ValueError("matching_mode must be one of 'true_to_pred' or 'pred_to_true' or 'both'.")
    return out
    


@post_processing(data_capture=['index'], 
                 result_capture=['interactions',
                                 'truth_interactions'])
def match_interactions(data_dict,
                       result_dict,
                       matching_mode='true_to_pred',
                       min_overlap=0,
                       overlap_mode='iou',
                       weight=False,
                       list_principal_matches=True):

    pred_interactions = result_dict['interactions']
    
    out = {}
    
    if overlap_mode == 'chamfer':
        true_interactions = [ia for ia in result_dict['truth_interactions'] if ia.truth_size > 0]
    else:
        true_interactions = [ia for ia in result_dict['truth_interactions'] if ia.size > 0]
    
    # Only consider interactions with nonzero predicted nonghost
    
    if matching_mode == 'pred_to_true':
        overlap_matrix, value_matrix = weighted_matrix_iou(pred_interactions, true_interactions, weight=weight)
        matched_interactions, counts = match_interactions_fn(
            pred_interactions, 
            true_interactions, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        matches = generate_match_pairs(true_interactions, pred_interactions, 
                                       prefix='matched_interactions',
                                       only_principal=list_principal_matches)
        out['matched_interactions_r2t'] = matches['matched_interactions_r2t']
        out['interaction_match_overlap_r2t'] = matches['matched_interactions_r2t_values']
    elif matching_mode == 'true_to_pred':
        overlap_matrix, value_matrix = weighted_matrix_iou(true_interactions, pred_interactions, weight=weight)
        matched_interactions, counts = match_interactions_fn(
            true_interactions, 
            pred_interactions, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        matches = generate_match_pairs(true_interactions, pred_interactions, 
                                       prefix='matched_interactions',
                                       only_principal=list_principal_matches)
        out['matched_interactions_t2r'] = matches['matched_interactions_t2r']
        out['interaction_match_overlap_t2r'] = matches['matched_interactions_t2r_values']
    elif matching_mode == 'both':
        overlap_matrix, value_matrix = weighted_matrix_iou(true_interactions, pred_interactions, weight=weight)
        matched_interactions, counts = match_interactions_fn(
            true_interactions, 
            pred_interactions, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        matches = generate_match_pairs(true_interactions, pred_interactions, 
                                       prefix='matched_interactions',
                                       only_principal=list_principal_matches)
        out['matched_interactions_t2r'] = matches['matched_interactions_t2r']
        out['interaction_match_overlap_t2r'] = matches['matched_interactions_t2r_values']
        
        overlap_matrix, value_matrix = weighted_matrix_iou(pred_interactions, true_interactions, weight=weight)
        matched_interactions, counts = match_interactions_fn(
            pred_interactions, 
            true_interactions, 
            value_matrix,
            overlap_matrix,
            min_overlap=min_overlap)
        matches = generate_match_pairs(true_interactions, pred_interactions, 
                                       prefix='matched_interactions',
                                       only_principal=list_principal_matches)
        out['matched_interactions_r2t'] = matches['matched_interactions_r2t']
        out['interaction_match_overlap_r2t'] = matches['matched_interactions_r2t_values']
    else:
        raise ValueError("matching_mode must be one of 'recursive', 'true_to_pred' or 'pred_to_true'.")
    
    return out


# ----------------------------- Helper functions -----------------------------

def match_parts_within_ints(int_matches):
    '''
    Given list of matches Tuple[(Truth)Interaction, (Truth)Interaction], 
    return list of particle matches Tuple[TruthParticle, Particle]. 

    This means rather than matching all predicted particles againts
    all true particles, it has an additional constraint that only
    particles within a matched interaction pair can be considered
    for matching. 
    '''

    matched_particles, match_overlap = [], []

    for m in int_matches:
        ia1, ia2 = m[0], m[1]
        num_parts_1, num_parts_2 = -1, -1
        if m[0] is not None:
            num_parts_1 = len(m[0].particles)
        if m[1] is not None:
            num_parts_2 = len(m[1].particles)
        if num_parts_1 <= num_parts_2:
            ia1, ia2 = m[0], m[1]
        else:
            ia1, ia2 = m[1], m[0]
            
        for p in ia2.particles:
            if len(p.match) == 0:
                if type(p) is Particle:
                    matched_particles.append((None, p))
                    match_overlap.append(-1)
                else:
                    matched_particles.append((p, None))
                    match_overlap.append(-1)
            for match_id in p.match:
                if type(p) is Particle:
                    matched_particles.append((ia1[match_id], p))
                else:
                    matched_particles.append((p, ia1[match_id]))
                match_overlap.append(p._match_overlap[match_id])
    return matched_particles, np.array(match_overlap)


def check_particle_matches(loaded_particles, clear=False):
    match_dict = OrderedDict({})
    for p in loaded_particles:
        for i, m in enumerate(p.match):
            match_dict[int(m)] = p.match_overlap[i]
        if clear:
            p._match = []
            p._match_overlap = OrderedDict()

    match_overlap = np.array(list(match_dict.values()))
    match = np.array(list(match_dict.keys())).astype(int)
    perm = np.argsort(match_overlap)[::-1]
    match_overlap = match_overlap[perm]
    match = match[perm]

    return match, match_overlap
