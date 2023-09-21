from collections import OrderedDict
from functools import partial

import numpy as np
import sys

from analysis.classes import TruthInteraction, TruthParticle, Interaction

def tag(tag_name):
    '''
    Tags a function with a str indicator for truth inputs only,
    reco inputs only, or both.
    '''
    def tags_decorator(func):
        func._tag = tag_name
        return func

    return tags_decorator

def attach_prefix(update_dict, prefix):
    '''
    Simple function that adds a prefix to all keys in update_dict
    '''
    if prefix is None:
        return update_dict
    out = OrderedDict({})

    for key, val in update_dict.items():
        new_key = "{}_".format(prefix) + str(key)
        out[new_key] = val

    return out

class AnalysisLogger:
    '''
    Base class for analysis tools logger interface.
    '''

    def __init__(self, fieldnames: dict):
        self.fieldnames = fieldnames
        self._data_producers = []

    def prepare(self):
        for name in self.fieldnames:
            f = getattr(self, name)
            self._data_producers.append(f)

    # def prepare(self):
    #     for fname, args_dict in self.fieldnames.items():
    #         if args_dict is None:
    #             f = getattr(self, fname)
    #         else:
    #             assert 'args' in args_dict
    #             kwargs = args_dict['args']
    #             f = partial(getattr(self, fname), **kwargs)
    #         self._data_producers.append(f)

    def produce(self, particle, mode=None):

        out = OrderedDict()
        if mode not in ['reco', 'true', None]:
            raise ValueError('Logger.produce mode argument must be either \
                             "true" or "reco", or None.')

        for f in self._data_producers:
            if hasattr(f, '_tag'):
                if f._tag is not None and f._tag != mode:
                    continue
            update_dict = f(particle)
            # print(f, update_dict)
            out.update(update_dict)

        out = attach_prefix(out, mode)

        return out
    

class ParticleLogger(AnalysisLogger):

    def __init__(self, fieldnames: dict, meta=None):
        super(ParticleLogger, self).__init__(fieldnames)
        self.meta = meta

    @staticmethod
    def id(particle):
        out = {'particle_id': -1}
        if hasattr(particle, 'id'):
            out['particle_id'] = particle.id
        return out
    
    @staticmethod
    def interaction_id(particle):
        out = {'particle_interaction_id': -1}
        if hasattr(particle, 'interaction_id')  :
            out['particle_interaction_id'] = particle.interaction_id
        return out
    
    @staticmethod
    def nu_id(particle):
        out = {'particle_nu_id': -1}
        if hasattr(particle, 'nu_id'):
            out['particle_nu_id'] = particle.nu_id
        return out

    @staticmethod
    def pid(particle):
        out = {'particle_type': -1}
        if hasattr(particle, 'pid'):
            out['particle_type'] = particle.pid
        return out
    
    @staticmethod
    @tag('reco')
    def primary_scores(particle):
        out = {'primary_score_0': -1,
               'primary_score_1': -1}
        if particle is not None:
            out['primary_score_0'] = particle.primary_scores[0]
            out['primary_score_1'] = particle.primary_scores[1]
        return out
    
    @staticmethod
    @tag('reco')
    def pid_scores(particle):
        out = {'pid_score_0': -1,
               'pid_score_1': -1,
               'pid_score_2': -1,
               'pid_score_3': -1,
               'pid_score_4': -1}
        if particle is not None:
            out['pid_score_0'] = particle.pid_scores[0]
            out['pid_score_1'] = particle.pid_scores[1]
            out['pid_score_2'] = particle.pid_scores[2]
            out['pid_score_3'] = particle.pid_scores[3]
            out['pid_score_4'] = particle.pid_scores[4]
        return out

    @staticmethod
    def semantic_type(particle):
        out = {'particle_semantic_type': -1}
        if hasattr(particle, 'semantic_type'):
            out['particle_semantic_type'] = particle.semantic_type
        return out
    
    @staticmethod
    def size(particle):
        out = {'particle_size': -1}
        if hasattr(particle, 'size'):
            out['particle_size'] = particle.size
        return out

    @staticmethod
    def is_primary(particle):
        out = {'particle_is_primary': False}
        if hasattr(particle, 'is_primary'):
            out['particle_is_primary'] = bool(particle.is_primary)
        return out
    
    @staticmethod
    def start_point(particle):
        out = {
            # 'particle_has_startpoint': False,
            'particle_start_point_x': -1,
            'particle_start_point_y': -1,
            'particle_start_point_z': -1
        }
        if (particle is not None) and (particle.start_point is not None):
            # out['particle_has_startpoint'] = True
            out['particle_start_point_x'] = particle.start_point[0]
            out['particle_start_point_y'] = particle.start_point[1]
            out['particle_start_point_z'] = particle.start_point[2]
        return out
    
    @staticmethod
    def end_point(particle):
        out = {
            # 'particle_has_endpoint': False,
            'particle_end_point_x': -1,
            'particle_end_point_y': -1,
            'particle_end_point_z': -1
        }
        if (particle is not None) and (particle.end_point is not None) \
            and (not (particle.end_point == -1).all()):
            # out['particle_has_endpoint'] = True
            out['particle_end_point_x'] = particle.end_point[0]
            out['particle_end_point_y'] = particle.end_point[1]
            out['particle_end_point_z'] = particle.end_point[2]
        return out
    
    @staticmethod
    def start_point_is_touching(particle, threshold=5.0):
        out = {'particle_start_point_is_touching': True}
        if type(particle) is TruthParticle:
            if particle.size > 0:
                diff = particle.points - particle.start_point.reshape(1, -1)
                dists = np.linalg.norm(diff, axis=1)
                min_dist = np.min(dists)
                if min_dist > threshold:
                    out['particle_startpoint_is_touching'] = False
        return out
    
    @staticmethod
    @tag('true')
    def creation_process(particle):
        out = {'particle_creation_process': 'N/A'}
        if type(particle) is TruthParticle:
            out['particle_creation_process'] = particle.creation_process
        return out
    
    @staticmethod
    #@tag('true')
    def momentum(particle):
        min_int = -sys.maxsize - 1
        out = {
            'p'          : min_int,
            'particle_px': min_int,
            'particle_py': min_int,
            'particle_pz': min_int,
        }
        if particle is not None:
            out['particle_px'] = particle.momentum[0]
            out['particle_py'] = particle.momentum[1]
            out['particle_pz'] = particle.momentum[2]
            out['p']           = np.linalg.norm(particle.momentum)
        return out
    
    @staticmethod
    @tag('true')
    def truth_start_dir(particle):
        min_int = -sys.maxsize - 1
        out = {
            'truth_start_dir_x': min_int,
            'truth_start_dir_y': min_int,
            'truth_start_dir_z': min_int,
        }
        if type(particle) is TruthParticle:
            out['truth_start_dir_x'] = particle.truth_start_dir[0]
            out['truth_start_dir_y'] = particle.truth_start_dir[1]
            out['truth_start_dir_z'] = particle.truth_start_dir[2]
        return out
    
    @staticmethod
    @tag('true')
    def energy_init(particle):
        out = {
            'energy_init': -1,
        }
        if type(particle) is TruthParticle:
            out['energy_init'] = particle.energy_init
        return out
    
    @staticmethod
    @tag('true')
    def energy_deposit(particle):
        out = {
            'energy_deposit': -1,
        }
        if type(particle) is TruthParticle:
            out['energy_deposit'] = particle.energy_deposit
        return out
    
    @staticmethod
    @tag('true')
    def num_children(particle):
        out = {
            'num_children': -1,
        }
        if type(particle) is TruthParticle:
            out['num_children'] = len(particle.children_id)
        return out
    
    @staticmethod
    @tag('true')
    def children_counts(particle):
        out = {'children_counts_0': -1,
               'children_counts_1': -1,
               'children_counts_2': -1,
               'children_counts_3': -1,
               'children_counts_4': -1}
        if type(particle) is TruthParticle:
            out['children_counts_0'] = particle.children_counts[0]
            out['children_counts_1'] = particle.children_counts[1]
            out['children_counts_2'] = particle.children_counts[2]
            out['children_counts_3'] = particle.children_counts[3]
            out['children_counts_4'] = particle.children_counts[4]
        return out
    
    @staticmethod
    def reco_start_dir(particle):
        out = {
            'particle_start_dir_x': 0,
            'particle_start_dir_y': 0,
            'particle_start_dir_z': 0
        }
        if particle is not None and hasattr(particle, 'start_dir'):
            v = particle.start_dir
            out['particle_start_dir_x'] = v[0]
            out['particle_start_dir_y'] = v[1]
            out['particle_start_dir_z'] = v[2]
        return out
    
    @staticmethod
    def reco_end_dir(particle):
        out = {
            'particle_end_dir_x': 0,
            'particle_end_dir_y': 0,
            'particle_end_dir_z': 0
        }
        if particle is not None and hasattr(particle, 'end_dir'):
            v = particle.start_dir
            out['particle_end_dir_x'] = v[0]
            out['particle_end_dir_y'] = v[1]
            out['particle_end_dir_z'] = v[2]
        return out
    
    @staticmethod
    def reco_length(particle):
        out = {'particle_length': -1}
        if particle is not None:
            out['particle_length'] = particle.length
        return out

    @staticmethod
    # @tag('reco')
    def calo_ke(particle):
        out = {'calo_ke': -1}
        if particle is not None:
            out['calo_ke'] = particle.calo_ke
        return out
    
    @staticmethod
    # @tag('reco')
    def csda_ke(particle):
        out = {'csda_ke': -1}
        if particle is not None:
            out['csda_ke'] = particle.csda_ke
        return out
    
    @staticmethod
    def is_contained(particle):
        out = {'particle_is_contained': False}
        if particle is not None:
            out['particle_is_contained'] = particle.is_contained
        return out

    @staticmethod
    def depositions_sum(particle):
        out = {'particle_depositions_sum': -1}
        if particle is not None:
            out['particle_depositions_sum'] = particle.depositions_sum
        return out
    
    @staticmethod
    @tag('true')
    def truth_depositions_sum(particle):
        out = {'particle_truth_depositions_sum': -1}
        if particle is not None and type(particle) is TruthParticle:
            out['particle_truth_depositions_sum'] = particle.truth_depositions_sum
        return out
    
    @staticmethod
    def matched(particle):
        out = {'matched': False}
        if particle is not None:
            out['matched'] = particle.matched
        return out
    
    @staticmethod
    def is_principal_match(particle):
        out = {'is_principal_match': False}
        if particle is not None:
            out['is_principal_match'] = particle.is_principal_match
        return out
    

class InteractionLogger(AnalysisLogger):

    def __init__(self, fieldnames: dict, meta=None):
        super(InteractionLogger, self).__init__(fieldnames)
        self.meta = meta

    @staticmethod
    def id(ia):
        out = {'interaction_id': -1}
        if ia is not None:
            out['interaction_id'] = ia.id
        return out
    
    @staticmethod
    def size(ia):
        out = {'interaction_size': -1}
        if ia is not None:
            out['interaction_size'] = ia.size
        return out
    
    @staticmethod
    def nu_id(ia):
        out = {'nu_id': -1}
        if ia is not None:
            out['nu_id'] = ia.nu_id
        return out
    
    @staticmethod
    def volume_id(ia):
        out = {'volume_id': -1}
        if ia is not None:
            out['volume_id'] = ia.volume_id
        return out
    
    @staticmethod
    def count_primary_particles(ia, ptypes=None):
        
        mapping = {
            0: 'num_primary_photons',
            1: 'num_primary_electrons',
            2: 'num_primary_muons',
            3: 'num_primary_pions',
            4: 'num_primary_protons'
        }
        
        if ptypes is not None:
            out = {mapping[pid] : -1 for pid in ptypes}
        else:
            ptypes = list(mapping.keys())
            out = {mapping[pid] : -1 for pid in ptypes}

        if ia is not None:
            if type(ia) is Interaction:
                for pid in ptypes:
                    out[mapping[pid]] = ia.primary_counts[pid]
            if type(ia) is TruthInteraction:
                for pid in ptypes:
                    out[mapping[pid]] = ia.truth_primary_counts[pid]
            
        return out
    
    @staticmethod
    def count_particles(ia, ptypes=None):
        
        mapping = {
            0: 'num_photons',
            1: 'num_electrons',
            2: 'num_muons',
            3: 'num_pions',
            4: 'num_protons'
        }
        
        if ptypes is not None:
            out = {mapping[pid] : -1 for pid in ptypes}
        else:
            ptypes = list(mapping.keys())
            out = {mapping[pid] : -1 for pid in ptypes}

        if ia is not None:
            if type(ia) is Interaction:
                for pid in ptypes:
                    out[mapping[pid]] = ia.primary_counts[pid]
            if type(ia) is TruthInteraction:
                for pid in ptypes:
                    out[mapping[pid]] = ia.truth_primary_counts[pid]
            
        return out
    
    @staticmethod
    def topology(ia):
        out = {'topology': 'N/A'}
        if ia is not None:
            out['topology'] = ia.topology
        return out
    
    @staticmethod
    @tag('true')
    def truth_topology(ia):
        out = {'truth_topology': 'N/A'}
        assert (ia is None) or (type(ia) is TruthInteraction)
        if ia is not None:
            out['truth_topology'] = ia.truth_topology
        return out
    
    @staticmethod
    def is_contained(ia):
        out = {'interaction_is_contained': False}
        if ia is not None:
            out['interaction_is_contained'] = ia.is_contained
        return out
    
    @staticmethod
    def vertex(ia):
        out = {
            # 'has_vertex': False,
            'vertex_x': -sys.maxsize,
            'vertex_y': -sys.maxsize,
            'vertex_z': -sys.maxsize,
            'vertex_mode': 'N/A'
            # 'vertex_info': None
        }
        if ia is not None and hasattr(ia, 'vertex'):
            out['vertex_x'] = ia.vertex[0]
            out['vertex_y'] = ia.vertex[1]
            out['vertex_z'] = ia.vertex[2]
            out['vertex_mode'] = ia.vertex_mode
        return out
    
    @staticmethod
    @tag('true')
    def truth_vertex(ia):
        out = {
            # 'has_vertex': False,
            'truth_vertex_x': -sys.maxsize,
            'truth_vertex_y': -sys.maxsize,
            'truth_vertex_z': -sys.maxsize,
            # 'vertex_info': None
        }
        if ia is not None and hasattr(ia, 'truth_vertex'):
            out['truth_vertex_x'] = ia.truth_vertex[0]
            out['truth_vertex_y'] = ia.truth_vertex[1]
            out['truth_vertex_z'] = ia.truth_vertex[2]
        return out
    
    @staticmethod
    @tag('true')
    def nu_info(ia):
        assert (ia is None) or (type(ia) is TruthInteraction)
        out = {
            'nu_interaction_type': 'N/A',
            'nu_interaction_mode': 'N/A',
            'nu_current_type': 'N/A',
            'nu_energy_init': 'N/A'
        }
        if ia is not None:
            if ia.nu_id == 1:
                out['nu_interaction_type'] = ia.nu_interaction_type
                out['nu_interaction_mode'] = ia.nu_interaction_mode
                out['nu_current_type']     = ia.nu_current_type
                out['nu_energy_init']      = ia.nu_energy_init
        return out
    
    @staticmethod
    @tag('reco')
    def flash_match_info(ia):
        assert (ia is None) or (type(ia) is Interaction)
        out = {
            'fmatched': False,
            'flash_time': -sys.maxsize,
            'flash_total_pE': -sys.maxsize,
            'flash_id': -sys.maxsize,
            'flash_hypothesis': -sys.maxsize
        }
        if ia is not None:
            if hasattr(ia, 'fmatched'):
                out['fmatched'] = ia.fmatched
                out['flash_time'] = ia.flash_time
                out['flash_total_pE'] = ia.flash_total_pE
                out['flash_id'] = ia.flash_id
                out['flash_hypothesis'] = ia.flash_hypothesis
        return out
    
    @staticmethod
    @tag('reco')
    def crt_match_info(ia):
        out = {
            'crthit_matched': False,
            'crthit_matched_particle_id': -1,
            'crthit_id': -1
        }
        assert (ia) is None or (type(ia) is Interaction)
        if ia is not None:
            out['crthit_id'] = ia.crthit_id
            out['crthit_matched'] = ia.crthit_matched
            out['crthit_matched_particle_id'] = ia.crthit_matched_particle_id
        return out
    
    @staticmethod
    def matched(ia):
        out = {'matched': False}
        if ia is not None:
            out['matched'] = ia.matched
        return out
    
    @staticmethod
    def is_principal_match(ia):
        out = {'is_principal_match': False}
        if ia is not None:
            out['is_principal_match'] = ia.is_principal_match
        return out
