import numpy as np

from typing import Counter, List, Union
from . import Particle
from mlreco.utils.globals import PDG_TO_PID
from functools import cached_property

class TruthParticle(Particle):
    '''
    Data structure mirroring <Particle>, reserved for true particles
    derived from true labels / true MC information.

    See <Particle> documentation for shared attributes.
    Below are attributes exclusive to TruthParticle.

    Attributes
    ----------
    depositions_MeV : np.ndarray
        (N) Array of energy deposition values for each voxel in MeV
    true_index : np.ndarray, default np.array([])
        (N) IDs of voxels that correspond to the particle within the label tensor
    true_points : np.dnarray, default np.array([], shape=(0,3))
        (N,3) Set of voxel coordinates that make up this particle in the label tensor
    true_depositions : np.ndarray
        (N) Array of charge deposition values for each true voxel
    true_depositions_MeV : np.ndarray
        (N) Array of energy deposition values for each true voxel in MeV
    start_position : np.ndarray
        True start position of the particle
    end_position : np.ndarray
        True end position of the particle
    momentum : float, default np.array([-1,-1,-1])
        True 3-momentum of the particle
    asis : larcv.Particle, optional
        Original larcv.Paticle instance which contains all the truth information
    '''
    def __init__(self, 
                 *args, 
                 depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 pid: int = -1,
                 is_primary: int = -1,
                 truth_index: np.ndarray = np.empty(0, dtype=np.int64), 
                 truth_points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 truth_depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 truth_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 particle_asis: object = None, 
                 **kwargs):

        super(TruthParticle, self).__init__(*args, **kwargs)

        self._pid = pid
        self._is_primary = is_primary

        # Initialize attributes
        self.depositions_MeV        = np.atleast_1d(depositions_MeV)
        self.truth_index            = truth_index
        self.truth_points           = truth_points
        self._truth_size            = truth_points.shape[0]
        self._truth_depositions     = np.atleast_1d(truth_depositions)   # Must be ADC
        self._truth_depositions_MeV = np.atleast_1d(truth_depositions_MeV)   # Must be MeV

        self.asis = particle_asis
        assert PDG_TO_PID[int(self.asis.pdg_code())] == self.pid
            
        # Quantities to be set during post-processing
        # tng stands for true nonghost
        self.length_tng = -1.
        self.csda_kinetic_energy_tng = -1.
        
        # Set start_point and end_point to first and last step in case
        # it wasn't set during initialization
        if self._start_point is None:
            self._start_point = self.first_step
        if self._end_point is None:
            self._end_point   = self.last_step


    @property
    def pid(self):
        return int(self._pid)
    
    @property
    def is_primary(self):
        return self._is_primary

    def __repr__(self):
        msg = super(TruthParticle, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthParticle, self).__str__()
        return 'Truth'+msg
    
    @property
    def truth_size(self):
        return self._truth_size
    
    @property
    def truth_depositions(self):
        return self._truth_depositions
    
    @truth_depositions.setter
    def truth_depositions(self, value):
        assert value.shape[0] == self._truth_size
        self._truth_depositions = np.atleast_1d(value)
    
    @property
    def truth_depositions_MeV(self):
        return self._truth_depositions_MeV
    
    @truth_depositions_MeV.setter
    def truth_depositions_MeV(self, value):
        assert value.shape[0] == self._truth_size
        self._truth_depositions_MeV = np.atleast_1d(value)
        
    @cached_property
    def momentum(self):
        self._momentum = np.array([getattr(self.asis, a)() \
            for a in ['px', 'py', 'pz']], dtype=np.float32)
        return self._momentum
        
    @cached_property
    def truth_start_dir(self):
        if np.linalg.norm(self.momentum) > 0.:
            self._truth_start_dir = self.momentum/np.linalg.norm(self.momentum)
        return self._truth_start_dir
    
    @cached_property
    def start_position(self):
        if self.asis is not None:
            self._start_position = np.array([getattr(self.asis.position(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._start_position = -np.ones(3)
        return self._start_position
    
    @cached_property
    def end_position(self):
        if self.asis is not None:
            self._end_position = np.array([getattr(self.asis.end_position(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._end_position = -np.ones(3)
        return self._end_position
    
    @cached_property
    def first_step(self):
        if self.asis is not None:
            self._first_step = np.array([getattr(self.asis.first_step(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._first_step = -np.ones(3)
        return self._first_step
    
    @cached_property
    def last_step(self):
        if self.semantic_type == 1:
            self._last_step = np.array([getattr(self.asis.last_step(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._last_step = -np.ones(3)
        return self._last_step
    
    @cached_property
    def energy_init(self):
        return float(self.asis.energy_init())
    
    @cached_property
    def energy_deposit(self):
        return float(self.asis.energy_deposit())