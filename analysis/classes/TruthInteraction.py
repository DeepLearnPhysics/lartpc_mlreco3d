import numpy as np

from typing import List
from collections import OrderedDict, defaultdict
from functools import cached_property

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
                 nu_interaction_type: int = -1,
                 nu_interaction_mode: int = -1,
                 nu_current_type: int = -1,
                 nu_energy_init: float = -1.,
                 **kwargs):
        
        # Initialize private attributes to be set by setter only
        self._particles  = None
        self._particle_counts = np.zeros(6, dtype=np.int64)
        self._primary_counts  = np.zeros(6, dtype=np.int64)
        self._truth_particle_counts = np.zeros(6, dtype=np.int64)
        self._truth_primary_counts  = np.zeros(6, dtype=np.int64)
        # self.particles   = particles
        
        if self._particles is None:
            self._depositions_MeV        = depositions_MeV
            self._truth_depositions      = truth_depositions
            self._truth_depositions_MeV  = truth_depositions_MeV
            self.truth_points = truth_points
            self.truth_index = truth_index
            
        # Invoke particles setter
        super(TruthInteraction, self).__init__(interaction_id, particles, **kwargs)

        # Neutrino-specific information to be filled elsewhere
        self.nu_interaction_type = nu_interaction_type
        self.nu_interaction_mode = nu_interaction_mode
        self.nu_current_type     = nu_current_type
        self.nu_energy_init      = nu_energy_init
        
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
            msg = f"TruthInteraction {self.id} already has a populated list of "\
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

                if p.pid >= 0:
                    self._truth_particle_counts[p.pid] += 1
                    self._truth_primary_counts[p.pid] += int(p.is_primary)
                    if len(p.index) > 0:
                        self._particle_counts[p.pid] += 1
                        self._primary_counts[p.pid] += int(p.is_primary)       
                        
                else:
                    self._truth_particle_counts[-1] += 1
                    self._truth_primary_counts[-1] += int(p.is_primary)
                    if len(p.index) > 0:
                        self._particle_counts[-1] += 1
                        self._primary_counts[-1] += int(p.is_primary)       

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
    
    @property
    def particle_counts(self):
        return self._particle_counts
        
    @property
    def primary_counts(self):
        return self._primary_counts
    
    @property
    def truth_particle_counts(self):
        return self._truth_particle_counts
        
    @property
    def truth_primary_counts(self):
        return self._truth_primary_counts
    
    @cached_property
    def truth_topology(self):
        msg = ""
        encode = {0: 'g', 1: 'e', 2: 'mu', 3: 'pi', 4: 'p', 5: '?'}
        for i, count in enumerate(self._truth_primary_counts):
            if count > 0:
                msg += f"{count}{encode[i]}"
        return msg

    @staticmethod
    def check_particle_input(x):
        assert isinstance(x, TruthParticle)

    def __repr__(self):
        msg = super(TruthInteraction, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthInteraction, self).__str__()
        return 'Truth'+msg

