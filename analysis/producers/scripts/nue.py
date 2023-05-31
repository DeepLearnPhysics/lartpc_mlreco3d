from analysis.producers.decorator import write_to
from collections import OrderedDict
import copy, sys
from pprint import pprint

from mlreco.utils.globals import *

@write_to(['interactions', 'particles'])
def run_nue_reco(data_blob, result, **kwargs):
    
    interactions, particles = [], []
    
    image_idxs = data_blob['index']
    
    int_dict = OrderedDict({
        'Index': -1,
        'true_interaction_id': -1,
        'reco_interaction_id': -1,
        'match_score': -1,
        'comment': 'N/A',
        'true_nu_energy_init': -1,
        'true_nu_interaction_type': -1,
        'true_nu_current_type': -1,
        'true_nu_interaction_mode': -1,
        'true_nu_range_based_energy': -1,
        'true_nu_ccqe_energy': -1,
        'reco_nu_range_based_energy': -1,
        'reco_nu_ccqe_energy': -1,
        'true_interaction_is_contained': False,
        'reco_interaction_is_contained': False,
        'true_vertex_x': -sys.maxsize,
        'true_vertex_y': -sys.maxsize,
        'true_vertex_z': -sys.maxsize,
        'reco_vertex_x': -sys.maxsize,
        'reco_vertex_y': -sys.maxsize,
        'reco_vertex_z': -sys.maxsize,
        'true_nu_id': -1,
        'reco_flash_time': -sys.maxsize,
        'true_topology': 'N/A',
        'reco_topology': 'N/A'
    })
    
    particle_dict = OrderedDict({
        'Index': -1,
        'true_particle_id': -1,
        'reco_particle_id': -1,
        'match_score': -1,
        'comment': -1,
        'true_particle_is_contained': False,
        'reco_particle_is_contained': False,
        'true_energy_init': -1,
        'true_range_based_energy': -1,
        'true_calorimetric_energy': -1,
        'reco_range_based_energy': -1,
        'reco_calorimetric_energy': -1,
        'true_start_point_x': -sys.maxsize,
        'true_start_point_y': -sys.maxsize,
        'true_start_point_z': -sys.maxsize,
        'reco_start_point_x': -sys.maxsize,
        'reco_start_point_y': -sys.maxsize,
        'reco_start_point_z': -sys.maxsize,
    })
    
    for idx, index in enumerate(image_idxs):
        ia_dict = copy.deepcopy(int_dict)
        ia_dict['Index'] = index
        particle_dict['Index'] = index
        
        # 1. Loop over interactions:
        for i, match in enumerate(result['matched_interactions'][idx]):
            true_ia, reco_ia = match
            if true_ia is None or reco_ia is None:
                continue
            # Only count principal matches (largest overlap)
            if not true_ia.is_principal_match or not reco_ia.is_principal_match:
                continue
            int_dict['true_interaction_id'] = true_ia.id
            int_dict['reco_interaction_id'] = reco_ia.id
            int_dict['reco_flash_time'] = reco_ia.flash_time
            int_dict['match_score'] = result['interaction_match_counts'][idx][i]
            int_dict['true_nu_id'] = true_ia.nu_id
            
            int_dict['true_vertex_x'] = true_ia.vertex[0]
            int_dict['true_vertex_y'] = true_ia.vertex[1]
            int_dict['true_vertex_z'] = true_ia.vertex[2]
            
            int_dict['reco_vertex_x'] = reco_ia.vertex[0]
            int_dict['reco_vertex_y'] = reco_ia.vertex[1]
            int_dict['reco_vertex_z'] = reco_ia.vertex[2]
            
            int_dict['true_topology'] = true_ia.topology
            int_dict['reco_topology'] = reco_ia.topology
            
            if true_ia.nu_id == 1:
                
                int_dict['true_nu_energy_init'] = true_ia.nu_energy_init * 1000
                int_dict['true_nu_current_type'] = true_ia.nu_current_type
                int_dict['true_nu_interaction_mode'] = true_ia.nu_interaction_mode
                int_dict['true_nu_interaction_type'] = true_ia.nu_interaction_type
                
                int_dict['true_interaction_is_contained'] = true_ia.is_contained
                int_dict['reco_interaction_is_contained'] = reco_ia.is_contained
                
            interactions.append(int_dict)
                
        for i, match in enumerate(result['matched_particles'][idx]):
            true_p, reco_p = match
            if true_ia is None or reco_ia is None:
                continue
            # Only count principal matches (largest overlap)
            if not true_ia.is_principal_match or not reco_ia.is_principal_match:
                continue
            particle_dict['true_particle_id'] = true_p.id
            particle_dict['reco_particle_id'] = reco_p.id
            particle_dict['match_score'] = result['particle_match_counts'][idx][i]
            particle_dict['true_particle_is_contained'] = true_p.is_contained
            particle_dict['reco_particle_is_contained'] = reco_p.is_contained
            
            particle_dict['true_energy_init'] = true_p.energy_init
            particle_dict['true_range_based_energy'] = true_p.csda_kinetic_energy
            particle_dict['reco_range_based_energy'] = reco_p.csda_kinetic_energy
            
            particle_dict['true_calorimetric_energy'] = true_p.depositions_sum
            particle_dict['reco_calorimetric_energy'] = reco_p.depositions_sum
            
            particle_dict['true_start_point_x'] = true_p.start_point[0]
            particle_dict['true_start_point_y'] = true_p.start_point[1]
            particle_dict['true_start_point_z'] = true_p.start_point[2]

            particle_dict['reco_start_point_x'] = reco_p.start_point[0]
            particle_dict['reco_start_point_y'] = reco_p.start_point[1]
            particle_dict['reco_start_point_z'] = reco_p.start_point[2]
            
            particles.append(particle_dict)
                
    
    return [interactions, particles]