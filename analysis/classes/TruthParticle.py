import numpy as np
from larcv import larcv
from typing import Counter, List, Union

from . import Particle

from mlreco.utils import pixel_to_cm
from mlreco.utils.globals import PDG_TO_PID, SHAPE_LABELS
from mlreco.utils.decorators import inherit_docstring

@inherit_docstring(Particle)
class TruthParticle(Particle):
    '''
    Data structure mirroring <Particle>, reserved for true particles
    derived from true labels / true MC information.

    It inherits from all the attributes of the corresponding `larcv.Particle`.
    Please see `larcv.Particle` documentation for more information.

    Attributes
    ----------
    depositions_MeV : np.ndarray
        (N) Array of energy deposition values for each reconstructed voxel in MeV
    truth_index : np.ndarray, default np.array([])
        (N_t) IDs of voxels that correspond to the particle within the label tensor
    truth_points : np.dnarray, default np.array([], shape=(0,3))
        (N_t, 3) Set of voxel coordinates that make up this particle in the label tensor
    truth_depositions : np.ndarray
        (N_t) Array of charge deposition values for each true voxel
    truth_depositions_MeV : np.ndarray
        (N_t) Array of energy deposition values for each true voxel in MeV
    sed_index : np.ndarray, default np.array([])
        (N_s) IDs of voxels that correspond to the particle with the SED tensor
    sed_points : np.dnarray, default np.array([], shape=(0,3))
        (N_s, 3) Set of voxel coordinates that make up this particle in the SED tensor
    sed_depositions_MeV : np.ndarray, default np.array([])
        (N_s) Array of energy deposition values for each SED voxel in MeV
    direction : np.ndarray
        (3) Unit vector corresponding to the true particle direction (normalized momentum)
    '''
    def __init__(self,
                 *args,
                 depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 truth_index: np.ndarray = np.empty(0, dtype=np.int64),
                 truth_points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 truth_depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 truth_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 sed_index: np.ndarray = np.empty(0, dtype=np.int64),
                 sed_points: np.ndarray = np.empty((0, 3), dtype=np.float32),
                 sed_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 particle_asis: object = larcv.Particle(),
                 **kwargs):

        # Set the attributes of the parent Particle class
        super(TruthParticle, self).__init__(*args, **kwargs)

        # Initialize private attributes to be assigned through setters only
        self._depositions_MeV       = None
        self._truth_index           = None
        self._sed_index             = None

        # Set attributes
        self._depositions_MeV       = np.atleast_1d(depositions_MeV)

        self.truth_index            = truth_index
        self._truth_points          = truth_points
        self._truth_depositions     = np.atleast_1d(truth_depositions)     # Must be ADC
        self._truth_depositions_MeV = np.atleast_1d(truth_depositions_MeV) # Must be MeV

        self.sed_index              = sed_index
        self._sed_points            = sed_points
        self._sed_depositions_MeV   = np.atleast_1d(sed_depositions_MeV)
        
        self._children_counts = np.zeros(len(SHAPE_LABELS), dtype=np.int64)

        # Load truth information from the true particle object
        self.register_larcv_particle(particle_asis)

        # Quantity to be set with the children counting post-processor
        self.children_counts = np.zeros(len(SHAPE_LABELS), dtype=np.int64)

        # Quantities to be set with the direction estimator
        self.truth_start_dir = np.zeros(3)
        self.truth_end_dir   = np.zeros(3)

        # Quantities to be set with track range reconstruction post-processor
        self.length_tng = -1.
        self.csda_kinetic_energy_tng = -1.

    def register_larcv_particle(self, particle):
        '''
        Extracts all the relevant attributes from the a
        `larcv.Particle` and makes it its own.

        Parameters
        ----------
        particle : larcv.Particle
            True MC particle object
        '''
        # Load up all of the scalar information
        shared_keys  = ['track_id', 'creation_process', 'pdg_code', 't']
        scalar_keys  = [pre + k for pre in ['', 'parent_', 'ancestor_'] for k in shared_keys]
        scalar_keys += ['distance_travel', 'energy_deposit', 'energy_init',\
                'id', 'parent_id', 'group_id', 'interaction_id',\
                'mcst_index', 'mct_index', 'num_voxels', 'p', 'shape']
        for k in scalar_keys:
            val = getattr(particle, k)()
            setattr(self, k, val)

        # Load up the children list
        self.children_id = np.array(particle.children_id())

        # Load up all the 3-vector information
        vec_keys = ['position', 'end_position', 'parent_position',\
                'ancestor_position', 'first_step', 'last_step']
        for k in vec_keys:
            larcv_vector = getattr(particle, k)()
            vector = np.array([getattr(larcv_vector, a)() for a in ['x', 'y', 'z']])
            setattr(self, k, vector)

        # Load up the 3-momentum (stored in a peculiar way) and the direction
        self.momentum  = np.array([getattr(particle, f'p{a}')() for a in ['x', 'y', 'z']])
        self.direction = np.zeros(3)
        if np.linalg.norm(self.momentum) > 0.:
            self.direction = self.momentum/np.linalg.norm(self.momentum)

        # Set parent attributes based on the above
        self.semantic_type = self.shape
        self.pid           = PDG_TO_PID[int(self.pdg_code)]
        self.start_point   = self.first_step
        self.end_point     = self.last_step

    def __repr__(self):
        msg = super(TruthParticle, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthParticle, self).__str__()
        return 'Truth'+msg

    @property
    def depositions_MeV(self):
        return self._depositions_MeV

    @depositions_MeV.setter
    def depositions_MeV(self, value):
        assert len(value) == self.size
        self._depositions_MeV = value

    @property
    def truth_index(self):
        return self._truth_index

    @truth_index.setter
    def truth_index(self, value):
        self._truth_index = np.atleast_1d(value)
        self.truth_size = len(self._truth_index)

    @property
    def truth_points(self):
        return self._truth_points

    @truth_points.setter
    def truth_points(self, value):
        assert len(value) == self.truth_size
        self._truth_points = value

    @property
    def truth_depositions(self):
        return self._truth_depositions

    @truth_depositions.setter
    def truth_depositions(self, value):
        assert len(value) == self.truth_size
        self._truth_depositions = value

    @property
    def truth_depositions_MeV(self):
        return self._truth_depositions_MeV

    @truth_depositions_MeV.setter
    def truth_depositions_MeV(self, value):
        assert len(value) == self.truth_size
        self._truth_depositions_MeV = value

    @property
    def sed_index(self):
        return self._sed_index

    @sed_index.setter
    def sed_index(self, value):
        self._sed_index = np.atleast_1d(value)
        self.sed_size = len(self._sed_index)

    @property
    def sed_points(self):
        return self._sed_points

    @sed_points.setter
    def sed_points(self, value):
        assert len(value) == self.sed_size
        self._sed_points = value

    @property
    def sed_depositions_MeV(self):
        return self._sed_depositions_MeV

    @sed_depositions_MeV.setter
    def sed_depositions_MeV(self, value):
        assert len(value) == self.sed_size
        self._sed_depositions_MeV = value
    
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
