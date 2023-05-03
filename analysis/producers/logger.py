from collections import OrderedDict
from functools import partial

import numpy as np
import sys

from mlreco.utils.globals import PID_LABELS
from analysis.classes import TruthInteraction, TruthParticle, Interaction

def tag(tag_name):
    """Tags a function with a str indicator for truth inputs only,
    reco inputs only, or both.
    """
    def tags_decorator(func):
        func._tag = tag_name
        return func
    return tags_decorator

def attach_prefix(update_dict, prefix):
    """Simple function that adds a prefix to all keys in update_dict"""
    if prefix is None:
        return update_dict
    out = OrderedDict({})

    for key, val in update_dict.items():
        new_key = "{}_".format(prefix) + str(key)
        out[new_key] = val

    return out

class AnalysisLogger:
    """
    Base class for analysis tools logger interface.
    """

    def __init__(self, fieldnames: dict):
        self.fieldnames = fieldnames
        self._data_producers = []

    def prepare(self):
        for fname, args_dict in self.fieldnames.items():
            if args_dict is None:
                f = getattr(self, fname)
            else:
                assert 'args' in args_dict
                kwargs = args_dict['args']
                f = partial(getattr(self, fname), **kwargs)
            self._data_producers.append(f)

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
            out.update(update_dict)

        out = attach_prefix(out, mode)

        return out
    

class ParticleLogger(AnalysisLogger):

    def __init__(self, fieldnames: dict):
        super(ParticleLogger, self).__init__(fieldnames)

    @staticmethod
    def id(particle):
        out = {'particle_id': -1}
        if hasattr(particle, 'id'):
            out['particle_id'] = particle.id
        return out
    
    @staticmethod
    def interaction_id(particle):
        out = {'particle_interaction_id': -1}
        if hasattr(particle, 'interaction_id'):
            out['particle_interaction_id'] = particle.interaction_id
        return out

    @staticmethod
    def pdg_type(particle):
        out = {'particle_type': -1}
        if hasattr(particle, 'pid'):
            out['particle_type'] = particle.pid
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
        out = {'particle_is_primary': -1}
        if hasattr(particle, 'is_primary'):
            out['particle_is_primary'] = particle.is_primary
        return out
    
    @staticmethod
    def start_point(particle):
        out = {
            'particle_has_startpoint': False,
            'particle_startpoint_x': -1,
            'particle_startpoint_y': -1,
            'particle_startpoint_z': -1
        }
        if (particle is not None) and (particle.start_point is not None) \
            and (not (particle.start_point == -1).all()):
            out['particle_has_startpoint'] = True
            out['particle_startpoint_x'] = particle.start_point[0]
            out['particle_startpoint_y'] = particle.start_point[1]
            out['particle_startpoint_z'] = particle.start_point[2]
        return out
    
    @staticmethod
    def end_point(particle):
        out = {
            'particle_has_endpoint': False,
            'particle_endpoint_x': -1,
            'particle_endpoint_y': -1,
            'particle_endpoint_z': -1
        }
        if (particle is not None) and (particle.end_point is not None) \
            and (not (particle.end_point == -1).all()):
            out['particle_has_endpoint'] = True
            out['particle_endpoint_x'] = particle.end_point[0]
            out['particle_endpoint_y'] = particle.end_point[1]
            out['particle_endpoint_z'] = particle.end_point[2]
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
            out['particle_creation_process'] = particle.asis.creation_process()
        return out
    
    @staticmethod
    @tag('true')
    def momentum(particle):
        min_int = -sys.maxsize - 1
        out = {
            'particle_px': min_int,
            'particle_py': min_int,
            'particle_pz': min_int,
        }
        if type(particle) is TruthParticle:
            out['particle_px'] = particle.asis.px()
            out['particle_py'] = particle.asis.py()
            out['particle_pz'] = particle.asis.pz()
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
        if particle is not None and hasattr(particle, 'length'):
            out['particle_length'] = particle.length
        return out
    
    @staticmethod
    def csda_kinetic_energy(particle):
        out = {'csda_kinetic_energy': -1}
        if particle is not None:
            out['csda_kinetic_energy'] = particle.csda_kinetic_energy
    
    @staticmethod
    def is_contained(particle, vb, threshold=30):

        out = {'particle_is_contained': False}
        if particle is not None and len(particle.points) > 0:
            if not isinstance(threshold, np.ndarray):
                threshold = threshold * np.ones((3,))
            else:
                assert len(threshold) == 3
                assert len(threshold.shape) == 1

            vb = np.array(vb)

            x = (vb[0, 0] + threshold[0] <= particle.points[:, 0]) \
                        & (particle.points[:, 0] <= vb[0, 1] - threshold[0])
            y = (vb[1, 0] + threshold[1] <= particle.points[:, 1]) \
                        & (particle.points[:, 1] <= vb[1, 1] - threshold[1])
            z = (vb[2, 0] + threshold[2] <= particle.points[:, 2]) \
                        & (particle.points[:, 2] <= vb[2, 1] - threshold[2])

            out['particle_is_contained'] =  (x & y & z).all()
        return out

    @staticmethod
    def depositions_sum(particle):
        out = {'particle_depositions_sum': -1}
        if particle is not None:
            out['particle_depositions_sum'] = particle.depositions_sum
        return out
    

class InteractionLogger(AnalysisLogger):

    def __init__(self, fieldnames: dict):
        super(InteractionLogger, self).__init__(fieldnames)

    @staticmethod
    def id(ia):
        out = {'interaction_id': -1}
        if hasattr(ia, 'id'):
            out['interaction_id'] = ia.id
        return out
    
    @staticmethod
    def size(ia):
        out = {'interaction_size': -1}
        if ia is not None:
            out['interaction_size'] = ia.size
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
            for pid in ptypes:
                out[mapping[pid]] = ia.primary_counts[pid]
            
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
            for pid in ptypes:
                out[mapping[pid]] = ia.primary_counts[pid]
            
        return out
    
    @staticmethod
    def topology(ia):
        out = 'N/A'
        if ia is not None:
            out['topology'] = ia.topology
    
    @staticmethod
    def is_contained(ia, vb, threshold=30):

        out = {'interaction_is_contained': False}
        if ia is not None and len(ia.points) > 0:
            if not isinstance(threshold, np.ndarray):
                threshold = threshold * np.ones((3,))
            else:
                assert len(threshold) == 3
                assert len(threshold.shape) == 1

            vb = np.array(vb)

            x = (vb[0, 0] + threshold[0] <= ia.points[:, 0]) \
                        & (ia.points[:, 0] <= vb[0, 1] - threshold[0])
            y = (vb[1, 0] + threshold[1] <= ia.points[:, 1]) \
                        & (ia.points[:, 1] <= vb[1, 1] - threshold[1])
            z = (vb[2, 0] + threshold[2] <= ia.points[:, 2]) \
                        & (ia.points[:, 2] <= vb[2, 1] - threshold[2])

            out['interaction_is_contained'] =  (x & y & z).all()
        return out
    
    @staticmethod
    def vertex(ia):
        out = {
            # 'has_vertex': False,
            'vertex_x': -sys.maxsize,
            'vertex_y': -sys.maxsize,
            'vertex_z': -sys.maxsize,
            # 'vertex_info': None
        }
        if ia is not None and hasattr(ia, 'vertex'):
            out['vertex_x'] = ia.vertex[0]
            out['vertex_y'] = ia.vertex[1]
            out['vertex_z'] = ia.vertex[2]
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
            if ia.nu_id == 1 and isinstance(ia.nu_info, dict):
                out.update(ia.nu_info)
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
                out['flash_time'] = ia.fmatch_time
                out['flash_total_pE'] = ia.fmatch_total_pE
                out['flash_id'] = ia.fmatch_id
                out['flash_hypothesis'] = ia.flash_hypothesis
        return out
