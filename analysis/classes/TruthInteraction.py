import numpy as np

from typing import List
from collections import OrderedDict, defaultdict

from . import Interaction, TruthParticle
from .Interaction import _process_interaction_attributes


class TruthInteraction(Interaction):
    """
    Data structure mirroring <Interaction>, reserved for true interactions
    derived from true labels / true MC information.

    See <Interaction> documentation for shared attributes.
    Below are attributes exclusive to TruthInteraction

    Attributes
    ----------
    depositions_MeV : np.ndarray, default np.array([])
        Similar as `depositions`, i.e. using adapted true labels.
        Using true MeV energy deposits instead of rescaled ADC units.
    """

    def __init__(self,
                 interaction_id: int = -1, 
                 particles: List[TruthParticle] = None,
                 depositions_MeV : np.ndarray = np.empty(0, dtype=np.float32),
                 truth_index: np.ndarray = np.empty(0, dtype=np.int64),
                 truth_points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 truth_depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 truth_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 **kwargs):
        
        # Initialize private attributes to be set by setter only
        self._particles  = None
        # Invoke particles setter
        self.particles   = particles
        
        if self._particles is None:
            self._depositions_MeV        = depositions_MeV
            self._truth_depositions      = truth_depositions
            self._truth_depositions_MeV  = truth_depositions_MeV
            self.truth_points = truth_points
            self.truth_index = truth_index
            
        super(TruthInteraction, self).__init__(interaction_id, particles, **kwargs)

        # Neutrino-specific information to be filled elsewhere
        self.nu_interaction_type = -1
        self.nu_interaction_mode = -1
        self.nu_current_type = -1
        self.nu_energy_init = -1.
        
    @property
    def particles(self):
        return self._particles.values()
    
    @particles.setter
    def particles(self, value):
        '''
        <Particle> list getter/setter. The setter also sets
        the general interaction properties
        '''
        
        if self._particles is not None:
            msg = f"Interaction {self.id} already has a populated list of "\
                "particles. You cannot change the list of particles in a "\
                "given Interaction once it has been set."
            raise AttributeError(msg)

        if value is not None:
            self._particles = {p.id : p for p in value}
            id_list, index_list, points_list, depositions_list = [], [], [], []
            true_index_list, true_points_list = [], []
            true_depositions_list, true_depositions_MeV_list = [], []
            depositions_MeV_list = []
            for p in value:
                self.check_particle_input(p)
                id_list.append(p.id)
                index_list.append(p.index)
                points_list.append(p.points)
                depositions_list.append(p.depositions)
                depositions_MeV_list.append(p.depositions_MeV)
                true_index_list.append(p.truth_index)
                true_points_list.append(p.truth_points)
                true_depositions_list.append(p.truth_depositions)
                true_depositions_MeV_list.append(p.truth_depositions_MeV)

            self._particle_ids          = np.array(id_list, dtype=np.int64)
            self._num_particles         = len(value)
            self._num_primaries         = len([1 for p in value if p.is_primary])
            self.index                  = np.atleast_1d(np.concatenate(index_list))
            self.points                 = np.atleast_1d(np.vstack(points_list))
            self.depositions            = np.atleast_1d(np.concatenate(depositions_list))
            self.truth_points           = np.atleast_1d(np.concatenate(true_points_list))
            self.truth_index            = np.atleast_1d(np.concatenate(true_index_list))
            self._depositions_MeV       = np.atleast_1d(np.concatenate(depositions_MeV_list))
            self._truth_depositions     = np.atleast_1d(np.concatenate(true_depositions_list))
            self._truth_depositions_MeV = np.atleast_1d(np.concatenate(true_depositions_MeV_list))
        
    @classmethod
    def from_particles(cls, particles, verbose=False, **kwargs):
        
        assert len(particles) > 0
        init_args = defaultdict(list)
        reserved_attributes = [
            'interaction_id', 'nu_id', 'volume_id', 
            'image_id', 'points', 'index', 'depositions', 'depositions_MeV',
            'truth_depositions_MeV', 'truth_depositions', 'truth_index'
        ]
        
        processed_args = {'particles': []}
        for key, val in kwargs.items():
            processed_args[key] = val
        for p in particles:
            assert type(p) is TruthParticle
            for key in reserved_attributes:
                if key not in kwargs:
                    init_args[key].append(getattr(p, key))
            processed_args['particles'].append(p)
        
        _process_interaction_attributes(init_args, processed_args, **kwargs)
        
        for i, t in enumerate(init_args['truth_depositions_MeV']):
            if len(t.shape) == 0:
                print(t, t.shape)
                print(init_args['truth_index'][i])
        
        # Handle depositions_MeV for TruthParticles
        processed_args['depositions_MeV']       = np.concatenate(init_args['depositions_MeV'])
        processed_args['truth_depositions']     = np.concatenate(init_args['truth_depositions'])
        processed_args['truth_depositions_MeV'] = np.concatenate(init_args['truth_depositions_MeV'])
        
        truth_interaction = cls(**processed_args)
        
        return truth_interaction

    @property
    def depositions_MeV(self):
        return self._depositions_MeV
    
    @property
    def truth_depositions(self):
        return self._truth_depositions

    @property
    def truth_depositions_MeV(self):
        return self._truth_depositions_MeV
    
#    @property
#    def particles(self):
#        return list(self._particles.values())
#
#    @particles.setter
#    def particles(self, value):
#        assert isinstance(value, OrderedDict)
#        parts = {}
#        for p in value.values():
#            self.check_particle_input(p)
#            # Clear match information since Interaction is rebuilt
#            p.match = []
#            p._match_counts = {}
#            parts[p.id] = p
#        self._particles = OrderedDict(sorted(parts.items(), key=lambda t: t[0]))
#        self.update_info()

    @staticmethod
    def check_particle_input(x):
        assert isinstance(x, TruthParticle)

    def __repr__(self):
        msg = super(TruthInteraction, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthInteraction, self).__str__()
        return 'Truth'+msg

