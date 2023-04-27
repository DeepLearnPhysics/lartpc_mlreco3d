import numpy as np

from typing import Counter, List, Union
from . import Particle
from mlreco.utils.globals import PDG_TO_PID

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
                 momentum: np.ndarray = -np.ones(3, dtype=np.float32),
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
        if particle_asis is not None:
            self.start_position    = particle_asis.position()
            self.end_position      = particle_asis.end_position()

        self.asis = particle_asis
        assert PDG_TO_PID[int(self.asis.pdg_code())] == self.pid

        self.start_point = np.array([getattr(particle_asis.first_step(), a)() \
            for a in ['x', 'y', 'z']], dtype=np.float32)
        if self.semantic_type == 1:
            self.end_point = np.array([getattr(particle_asis.last_step(), a)() \
                for a in ['x', 'y', 'z']], dtype=np.float32)

        self.momentum    = np.array([getattr(particle_asis, a)() \
            for a in ['x', 'y', 'z']], dtype=np.float32)
        if np.linalg.norm(self.momentum) > 0.:
            self.start_dir = self.momentum/np.linalg.norm(self.momentum)


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
    
    def is_contained(self, spatial_size):

        check_contained = self.start_position.x() >= 0 and self.start_position.x() <= spatial_size \
                      and self.start_position.y() >= 0 and self.start_position.y() <= spatial_size \
                      and self.start_position.z() >= 0 and self.start_position.z() <= spatial_size \
                      and self.end_position.x()   >= 0 and self.end_position.x()   <= spatial_size \
                      and self.end_position.y()   >= 0 and self.end_position.y()   <= spatial_size \
                      and self.end_position.z()   >= 0 and self.end_position.z()   <= spatial_size
        return check_contained

    def purity_efficiency(self, other_particle):
        overlap = len(np.intersect1d(self.index, other_particle.index))
        return {
            "purity": overlap / len(other_particle.index),
            "efficiency": overlap / len(self.index)
        }

