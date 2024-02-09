import numpy as np
from larcv import larcv
from typing import Counter, List, Union

from mlreco.utils import pixel_to_cm
from mlreco.utils.globals import PDG_TO_PID, TRACK_SHP, SHAPE_LABELS, PID_LABELS
from mlreco.utils.decorators import inherit_docstring

from . import ParticleFragment


class TruthParticleFragment(ParticleFragment):
    """
    Data structure mirroring <ParticleFragment>, reserved for true fragments
    derived from true labels / true MC information.

    See <ParticleFragment> documentation for shared attributes.
    Below are attributes exclusive to TruthInteraction

    Attributes
    ----------
    depositions_MeV : np.ndarray, default np.array([])
        Similar as `depositions`, i.e. using adapted true labels.
        Using true MeV energy deposits instead of rescaled ADC units.
    """

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
                 truth_momentum: np.ndarray = np.full(3, -np.inf, dtype=np.float32),
                 truth_start_dir: np.ndarray = np.full(3, -np.inf, dtype=np.float32),
                 particle_asis: object = larcv.Particle(),
                 children_counts: np.ndarray = np.zeros(len(SHAPE_LABELS), dtype=np.int64),
                 **kwargs):
        super(TruthParticleFragment, self).__init__(*args, **kwargs)

        # Set attributes
        self._depositions_MeV       = np.atleast_1d(depositions_MeV)

        self.truth_index            = truth_index
        self._truth_points          = truth_points
        self._truth_depositions     = np.atleast_1d(truth_depositions)     # Must be ADC
        self._truth_depositions_MeV = np.atleast_1d(truth_depositions_MeV) # Must be MeV

        self.sed_index              = sed_index
        self._sed_points            = sed_points
        self._sed_depositions_MeV   = np.atleast_1d(sed_depositions_MeV)
        
        # Load truth information from the true particle object
        self.truth_momentum = truth_momentum
        self.truth_start_dir = np.copy(truth_start_dir)
        if particle_asis is not None:
            self.register_larcv_particle(particle_asis)
            
        # Quantity to be set with the children counting post-processor
        self.children_counts = np.copy(children_counts)
        
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
                'parent_id', 'mcst_index', 'mct_index', 'num_voxels', 'p', 'shape']
        for k in scalar_keys:
            val = getattr(particle, k)()
            setattr(self, k, val)

        # TODO: Move this to main list once this is in every LArCV file
        if hasattr(particle, 'gen_id'):
            setattr(self, 'gen_id', particle.gen_id())
            
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
        self.truth_momentum = np.array([getattr(particle, f'p{a}')() for a in ['x', 'y', 'z']])
        self.truth_start_dir = np.full(3, -np.inf, dtype=np.float32)
        if np.linalg.norm(self.truth_momentum):
            self.truth_start_dir = \
                    self.truth_momentum/np.linalg.norm(self.truth_momentum)

        # Set parent attributes based on the above
        # self.semantic_type = self.shape
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

        # TODO: Move this to main list once this is in every LArCV file
        if 'gen_id' in particle_dict:
            setattr(self, 'gen_id', particle_dict['gen_id'])
            
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

    def __repr__(self):
        msg = super(TruthParticleFragment, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthParticleFragment, self).__str__()
        return 'Truth'+msg
