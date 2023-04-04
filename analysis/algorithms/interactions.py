import numpy as np

from analysis.classes import Interaction
from collections import OrderedDict, Counter
from analysis.algorithms.utils import attach_prefix
from analysis.algorithms.logger import AnalysisLogger
from mlreco.utils.globals import PID_LABEL_TO_PARTICLE, PARTICLE_TO_PID_LABEL
from analysis.classes import TruthInteraction

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
        if hasattr(ia, 'size'):
            out['interaction_size'] = ia.size
        return out
    
    @staticmethod
    def count_primary_particles(ia, ptypes=None):
        all_types = sorted(list(PID_LABEL_TO_PARTICLE.keys()))
        if ptypes is None:
            ptypes = all_types
        elif set(ptypes).issubset(set(all_types)):
            pass
        elif len(ptypes) == 0:
            return {}
        else:
            raise ValueError('"ptypes under count_primary_particles must \
                             either be None or a list of particle type ids \
                             to be counted.')
        
        out = OrderedDict({'count_primary_'+name.lower() : 0 \
                           for name in PARTICLE_TO_PID_LABEL.keys() \
                            if PARTICLE_TO_PID_LABEL[name] in ptypes})

        if ia is not None and hasattr(ia, 'primary_particle_counts'):
            out.update({'count_primary_'+key.lower() : val \
                        for key, val in ia.primary_particle_counts.items() \
                        if key.upper() != 'OTHER' \
                            and PARTICLE_TO_PID_LABEL[key.upper()] in ptypes})
        return out
            
    
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
            'vertex_x': -1,
            'vertex_y': -1,
            'vertex_z': -1,
            # 'vertex_info': None
        }
        if ia is not None and hasattr(ia, 'vertex'):
            out['vertex_x'] = ia.vertex[0]
            out['vertex_y'] = ia.vertex[1]
            out['vertex_z'] = ia.vertex[2]
        return out
    
    @staticmethod
    def nu_info(ia):
        assert type(ia) is TruthInteraction
        out = {
            'nu_interaction_type': 'N/A',
            'nu_interaction_mode': 'N/A',
            'nu_current_type': 'N/A',
            'nu_energy_init': 'N/A'
        }
        if ia.nu_id == 1 and hasattr(ia, 'nu_info'):
            out.update(ia.nu_info)
        return out