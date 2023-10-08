import numpy as np
from larcv import larcv
from typing import Counter, List, Union

from . import Particle

from mlreco.utils import pixel_to_cm
from mlreco.utils.globals import PDG_TO_PID, TRACK_SHP, SHAPE_LABELS, PID_LABELS
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
    truth_id : int
        Index of the particle in the input list of larcv.Particle
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

    # Attributes that specify coordinates
    _COORD_ATTRS = Particle._COORD_ATTRS +\
	['truth_points', 'sed_points', 'position', 'end_position',\
	'parent_position', 'ancestor_position', 'first_step', 'last_step']

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
        if particle_asis is not None:
            self.register_larcv_particle(particle_asis)

        # Quantity to be set with the children counting post-processor
        self.children_counts = np.zeros(len(SHAPE_LABELS), dtype=np.int64)

        # Quantities to be set with the direction estimator
        self.truth_start_dir = np.zeros(3)
        self.truth_end_dir   = np.zeros(3)

        # Quantities to be set with track range reconstruction post-processor
        self.length_tng  = -1.
        self.csda_ke_tng = -1.
        self.calo_ke_tng = -1.
        self.mcs_ke_tng  = -1.

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
                'parent_id', 'group_id', 'interaction_id',\
                'mcst_index', 'mct_index', 'num_voxels', 'p', 'shape']
        for k in scalar_keys:
            val = getattr(particle, k)()
            setattr(self, k, val)
            
        # Exception for particle_id
        self.truth_id = particle.id()

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
        self.start_point   = self.first_step.astype(np.float32)
        self.end_point     = self.last_step.astype(np.float32)

        # Patch to deal with bad LArCV input, TODO: fix it upstream
        if self.start_point[0] == -np.inf and self.end_point[0] == -np.inf:
            self.start_point = self.end_point = np.zeros(3, dtype=self.start_point.dtype)
        if self.end_point[0] == -np.inf:
            self.end_point = self.start_point
            
    def load_larcv_attributes(self, particle_dict):
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
                'parent_id', 'group_id', 'interaction_id',\
                'mcst_index', 'mct_index', 'num_voxels', 'p', 'shape',\
                'pid', 'semantic_type']

        # Load up all the 3-vector information
        vec_keys = ['position', 'end_position', 'parent_position',\
                'ancestor_position', 'first_step', 'last_step',\
                'start_point', 'end_point']
        
        attribute_keys = scalar_keys + vec_keys
        for attr_name in attribute_keys:
            attr = particle_dict[attr_name]
            if type(attr) is bytes:
                attr = attr.decode()
            setattr(self, attr_name, attr)

    def merge(self, particle):
        '''
        Merge another particle object into this one
        '''
        super(TruthParticle, self).merge(particle)

        # Stack the two particle array attributes together
        for attr in ['truth_index', 'truth_depositions', \
                'sed_index', 'sed_depositions']:
            val = np.concatenate([getattr(self, attr), getattr(particle, attr)])
            setattr(self, attr, val)
        for attr in ['truth_points', 'sed_points']:
            val = np.vstack([getattr(self, attr), getattr(particle, attr)])
            setattr(self, attr, val)

        # Stack the two particle array attributes together
        self.index = np.concatenate([self.index, particle.index])
        self.points = np.vstack([self.points, particle.points])
        self.sources = np.vstack([self.sources, particle.sources])
        self.depositions = np.concatenate([self.depositions, particle.depositions])

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

    @property
    def truth_depositions_sum(self):
        return self.truth_depositions.sum()
    
    @property
    def truth_depositions_MeV_sum(self):
        return self.truth_depositions_MeV.sum()
    
    @property
    def sed_depositions_MeV_sum(self):
        return self.sed_depositions_MeV.sum()
    
    @property
    def primary_scores(self):
        '''
        Primary ID scores getter/setter. The setter converts the
        scores to a primary prediction through argmax.
        '''
        raise AttributeError("primary_scores cannot be referenced or assigned for TruthParticles")

    @primary_scores.setter
    def primary_scores(self, primary_scores):
        # If no primary scores are given, the primary status is unknown
        self._primary_scores = primary_scores
    
    @property
    def pid_scores(self):
        '''
        Primary ID scores getter/setter. The setter converts the
        scores to a primary prediction through argmax.
        '''
        raise AttributeError("pid_scores cannot be referenced or assigned for TruthParticles")

    @pid_scores.setter
    def pid_scores(self, pid_scores):
        # If no primary scores are given, the primary status is unknown
        self._pid_scores = pid_scores
