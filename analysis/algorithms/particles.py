from collections import OrderedDict
from functools import partial, partialmethod
import numpy as np
import sys
from analysis.algorithms.logger import AnalysisLogger

from analysis.classes import Particle, TruthParticle
from analysis.algorithms.utils import attach_prefix
from analysis.algorithms.calorimetry import get_particle_direction, compute_track_length


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
    def startpoint(particle):
        out = {
            'particle_has_startpoint': False,
            'particle_startpoint_x': -1,
            'particle_startpoint_y': -1,
            'particle_startpoint_z': -1
        }
        if hasattr(particle, 'startpoint') \
            and not (particle.startpoint == -1).all():
            out['particle_has_startpoint'] = True
            out['particle_startpoint_x'] = particle.startpoint[0]
            out['particle_startpoint_y'] = particle.startpoint[1]
            out['particle_startpoint_z'] = particle.startpoint[2]
        return out
    
    @staticmethod
    def endpoint(particle):
        out = {
            'particle_has_endpoint': False,
            'particle_endpoint_x': -1,
            'particle_endpoint_y': -1,
            'particle_endpoint_z': -1
        }
        if hasattr(particle, 'endpoint') \
            and not (particle.endpoint == -1).all():
            out['particle_has_endpoint'] = True
            out['particle_endpoint_x'] = particle.endpoint[0]
            out['particle_endpoint_y'] = particle.endpoint[1]
            out['particle_endpoint_z'] = particle.endpoint[2]
        return out
    
    @staticmethod
    def startpoint_is_touching(particle, threshold=5.0):
        out = {'particle_startpoint_is_touching': True}
        if type(particle) is TruthParticle:
            if particle.size > 0:
                diff = particle.points - particle.startpoint.reshape(1, -1)
                dists = np.linalg.norm(diff, axis=1)
                min_dist = np.min(dists)
                if min_dist > threshold:
                    out['particle_startpoint_is_touching'] = False
        return out
    
    @staticmethod
    def creation_process(particle):
        out = {'particle_creation_process': 'N/A'}
        if type(particle) is TruthParticle:
            out['particle_creation_process'] = particle.asis.creation_process()
        return out
    
    @staticmethod
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
    def reco_direction(particle, **kwargs):
        out = {
            'particle_dir_x': 0,
            'particle_dir_y': 0,
            'particle_dir_z': 0
        }
        if particle is not None:
            v = get_particle_direction(particle, **kwargs)
            out['particle_dir_x'] = v[0]
            out['particle_dir_y'] = v[1]
            out['particle_dir_z'] = v[2]
        return out
    
    @staticmethod
    def reco_length(particle):
        out = {'particle_length': -1}
        if particle is not None \
            and particle.semantic_type == 1 \
            and len(particle.points) > 0:
            out['particle_length'] = compute_track_length(particle.points)
        return out
    
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
    def sum_edep(particle):
        out = {'particle_sum_edep': -1}
        if particle is not None:
            out['particle_sum_edep'] = particle.sum_edep
        return out