import numpy as np

from typing import Counter, List, Union
from . import Particle
from mlreco.utils.globals import PDG_TO_PID, SHAPE_LABELS
from mlreco.utils.utils import pixel_to_cm
from functools import cached_property
import sys

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
    truth_index : np.ndarray, default np.array([])
        (N) IDs of voxels that correspond to the particle within the label tensor
    truth_points : np.dnarray, default np.array([], shape=(0,3))
        (N,3) Set of voxel coordinates that make up this particle in the label tensor
    truth_depositions : np.ndarray
        (N) Array of charge deposition values for each true voxel
    truth_depositions_MeV : np.ndarray
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
                 is_primary: bool = False,
                 truth_index: np.ndarray = np.empty(0, dtype=np.int64), 
                 truth_points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 sed_index: np.ndarray = np.empty(0, dtype=np.int64), 
                 sed_points: np.ndarray = np.empty((0, 3), dtype=np.float32),
                 truth_depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 truth_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 sed_depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 particle_asis: object = None, 
                 length_tng: float = -1.,
                 csda_kinetic_energy_tng: float = -1.,
                 **kwargs):

        super(TruthParticle, self).__init__(*args, **kwargs)

        self._pid = pid
        self._is_primary = is_primary

        # Initialize attributes
        self.depositions_MeV        = np.atleast_1d(depositions_MeV)
        self.truth_index            = truth_index
        self.truth_points           = truth_points
        self.sed_index              = sed_index
        self.sed_points             = sed_points
        self.sed_depositions        = np.atleast_1d(sed_depositions)
        self._sed_size              = sed_points.shape[0]
        self._truth_size            = truth_points.shape[0]
        self._truth_depositions     = np.atleast_1d(truth_depositions)   # Must be ADC
        self._truth_depositions_MeV = np.atleast_1d(truth_depositions_MeV)   # Must be MeV
        self._truth_depositions_sum = self._truth_depositions.sum()
        
        self._children_counts = np.zeros(len(SHAPE_LABELS), dtype=np.int64)

        self.asis = particle_asis
        # print(self.pid, PDG_TO_PID[int(self.asis.pdg_code())])
        assert (PDG_TO_PID[int(self.asis.pdg_code())] == self.pid) or (self.pid == -1)
            
        # Quantities to be set during post-processing
        # tng stands for true nonghost
        self.length_tng = length_tng
        self.csda_kinetic_energy_tng = csda_kinetic_energy_tng
        
        if self.asis is not None:
            self._start_position = np.array([getattr(self.asis.position(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._start_position = -np.ones(3) * sys.maxsize
            
        if self.asis is not None:
            self._end_position = np.array([getattr(self.asis.end_position(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._end_position = -np.ones(3) * sys.maxsize
            
        if self.asis is not None:
            self._first_step = np.array([getattr(self.asis.first_step(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._first_step = -np.ones(3) * sys.maxsize
            
        if self.semantic_type == 1:
            self._last_step = np.array([getattr(self.asis.last_step(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)
        else:
            self._last_step = -np.ones(3) * sys.maxsize
        


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
    def children_counts(self):
        return self._children_counts
    
    @property
    def truth_size(self):
        return self._truth_size
    
    @property
    def sed_size(self):
        return self._sed_size
    
    @property
    def truth_depositions(self):
        return self._truth_depositions
    
    @truth_depositions.setter
    def truth_depositions(self, value):
        assert value.shape[0] == self._truth_size
        self._truth_depositions = np.atleast_1d(value)
        self._truth_depositions_sum = np.sum(self._truth_depositions)
        
    @property
    def truth_depositions_sum(self):
        return self._truth_depositions_sum
    
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
    
    @property
    def start_position(self):
        return self._start_position
    
    @property
    def end_position(self):
        return self._end_position
    
    @property
    def first_step(self):
        return self._first_step
    
    @property
    def last_step(self):
        return self._last_step
    
    @cached_property
    def energy_init(self):
        return float(self.asis.energy_init())
    
    @cached_property
    def energy_deposit(self):
        return float(self.asis.energy_deposit())
    
    def convert_to_cm(self, meta):
        
        assert self._units == 'px'
        
        if len(self.points) > 0:
            self.points = pixel_to_cm(self.points, meta)
        self.start_point = pixel_to_cm(self.start_point, meta)
        self.end_point = pixel_to_cm(self.end_point, meta)
        
        self.truth_points = pixel_to_cm(self.truth_points, meta)
        if len(self.sed_points) > 0:
            self.sed_points = pixel_to_cm(self.sed_points, meta)
        if self.asis is not None:
            self._first_step = pixel_to_cm(self._first_step, meta)
            self._start_position = pixel_to_cm(self._start_position, meta)
            self._end_position = pixel_to_cm(self._end_position, meta)
            if self.semantic_type == 1:
                self._last_step = pixel_to_cm(self._last_step, meta)

        self._units = 'cm'