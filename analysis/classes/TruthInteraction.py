import numpy as np
from larcv import larcv
import sys

from typing import List
from collections import OrderedDict, defaultdict
from functools import cached_property

from . import Interaction, TruthParticle
from .Interaction import _process_interaction_attributes

from mlreco.utils import pixel_to_cm
from mlreco.utils.decorators import inherit_docstring

@inherit_docstring(Interaction)
class TruthInteraction(Interaction):
    """
    Data structure mirroring <Interaction>, reserved for true interactions
    derived from true labels / true MC information.

    Attributes
    ----------
    truth_id : int
        Index of the interaction as stored in the larcv.Particle information
    depositions_MeV : np.ndarray
        (N) Array of energy deposition values for each voxel in MeV
    truth_index : np.ndarray, default np.array([])
        (N_t) IDs of voxels that correspond to the interaction within the label tensor
    truth_points : np.dnarray, default np.array([], shape=(0,3))
        (N_t,3) Set of voxel coordinates that make up this interaction in the label tensor
    truth_depositions : np.ndarray
        (N_t) Array of energy deposition values for each true voxel in MeV
    truth_depositions_MeV : np.ndarray
        (N_t) Array of energy deposition values for each true voxel in MeV
    sed_index : np.ndarray, default np.array([])
        (N_s) IDs of voxels that correspond to the particle with the SED tensor
    sed_points : np.dnarray, default np.array([], shape=(0,3))
        (N_s, 3) Set of voxel coordinates that make up this particle in the SED tensor
    sed_depositions_MeV : np.ndarray, default np.array([])
        (N_s) Array of energy deposition values for each SED voxel in MeV
    truth_vertex : np.ndarray, optional
        (3) 3D coordinates of the true interaction vertex
    """

    # Attributes that specify coordinates
    _COORD_ATTRS = Interaction._COORD_ATTRS +\
	['truth_points', 'sed_points', 'truth_vertex']
 
    # Define placeholder values (-np.inf for float, -sys.maxsize for int)
    _SCALAR_KEYS = {'bjorken_x': -np.inf, 
                    'creation_process': 'N/A',
                    'current_type': -1, 
                    'distance_travel': -np.inf, 
                    'energy_deposit': -np.inf, 
                    'energy_init': -np.inf, 
                    'hadronic_invariant_mass': -np.inf,
                    'id': -1,
                    'inelasticity': -np.inf,
                    'interaction_mode': -sys.maxsize,
                    'interaction_type': -sys.maxsize,
                    'lepton_track_id': -sys.maxsize,
                    'mcst_index': -sys.maxsize,
                    'mct_index': -sys.maxsize,
                    'momentum_transfer': -np.inf,
                    # 'nu_track_id', # Exception
                    'nucleon': -sys.maxsize,
                    'num_voxels': -sys.maxsize,
                    'p': -np.inf,
                    'pdg_code': -sys.maxsize,
                    'quark': -sys.maxsize,
                    't': -np.inf,
                    'target': -sys.maxsize,
                    'theta': -np.inf}
    _VECTOR_KEYS = {'position': np.full(3, -np.inf, dtype=np.float32)}

    def __init__(self,
                 interaction_id: int = -1,
                 truth_id: int = -1,
                 particles: List[TruthParticle] = None,
                 depositions_MeV : np.ndarray = np.empty(0, dtype=np.float32),
                 truth_index: np.ndarray = np.empty(0, dtype=np.int64),
                 truth_points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 sed_index: np.ndarray = np.empty(0, dtype=np.int64),
                 sed_points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 truth_depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 truth_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 sed_depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 nu_pdg_code: int = -1,
                 nu_interaction_type: int = -1,
                 nu_interaction_mode: int = -1,
                 nu_current_type: int = -1,
                 nu_energy_init: float = -1.,
                 truth_vertex: np.ndarray = np.full(3, -np.inf),
                 neutrino: object = None,
                 **kwargs):

        # Store the truth ID of the interaction
        self.truth_id = truth_id

        # Initialize private attributes to be set by setter only
        self._particles  = None
        self._particle_counts = np.zeros(6, dtype=np.int64)
        self._primary_counts  = np.zeros(6, dtype=np.int64)
        self._truth_particle_counts = np.zeros(6, dtype=np.int64)
        self._truth_primary_counts  = np.zeros(6, dtype=np.int64)

        if self._particles is None:
            self._depositions_MeV        = depositions_MeV
            self._truth_depositions      = truth_depositions
            self._truth_depositions_MeV  = truth_depositions_MeV
            self.truth_points = truth_points
            self.truth_index = truth_index

            self.sed_index              = sed_index
            self.sed_points             = sed_points
            self.sed_depositions_MeV    = np.atleast_1d(sed_depositions_MeV)
            self._sed_size              = sed_points.shape[0]

        # Invoke particles setter
        super(TruthInteraction, self).__init__(interaction_id, particles, **kwargs)

        # Neutrino-specific information to be filled elsewhere
        self.nu_pdg_code         = nu_pdg_code
        self.nu_interaction_type = nu_interaction_type
        self.nu_interaction_mode = nu_interaction_mode
        self.nu_current_type     = nu_current_type
        self.nu_energy_init      = nu_energy_init

        # TODO: Must fill this attribute with truth information
        self.truth_vertex = truth_vertex
        self.register_larcv_neutrino()

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
            id_list, index_list, points_list, sources_list, depositions_list = [], [], [], [], []

            true_index_list, true_points_list = [], []
            true_depositions_list, true_depositions_MeV_list = [], []
            depositions_MeV_list = []

            sed_index_list, sed_points_list = [], []
            sed_depositions_MeV_list = []
            for p in value:
                self.check_particle_input(p)
                id_list.append(p.id)

                # Predicted Nonghost
                index_list.append(p.index)
                points_list.append(p.points)
                sources_list.append(p.sources)
                depositions_list.append(p.depositions)
                depositions_MeV_list.append(p.depositions_MeV)

                # True Nonghost
                true_index_list.append(p.truth_index)
                true_points_list.append(p.truth_points)
                true_depositions_list.append(p.truth_depositions)
                true_depositions_MeV_list.append(p.truth_depositions_MeV)

                # SED
                sed_index_list.append(p.sed_index)
                sed_points_list.append(p.sed_points)
                sed_depositions_MeV_list.append(p.sed_depositions_MeV)

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
            self.sources                = np.atleast_1d(np.vstack(sources_list))
            self.depositions            = np.atleast_1d(np.concatenate(depositions_list))
            self.truth_points           = np.atleast_1d(np.concatenate(true_points_list))
            self.truth_index            = np.atleast_1d(np.concatenate(true_index_list))
            self._depositions_MeV       = np.atleast_1d(np.concatenate(depositions_MeV_list))
            self._truth_depositions     = np.atleast_1d(np.concatenate(true_depositions_list))
            self._truth_depositions_MeV = np.atleast_1d(np.concatenate(true_depositions_MeV_list))
            self.sed_index              = np.atleast_1d(np.concatenate(sed_index_list))
            self.sed_points             = np.atleast_1d(np.concatenate(sed_points_list))
            self.sed_depositions_MeV    = np.atleast_1d(np.concatenate(sed_depositions_MeV_list))

    @classmethod
    def from_particles(cls, particles, verbose=False, **kwargs):

        assert len(particles) > 0
        init_args = defaultdict(list)
        # Particle-level attributes that needs to be processed
        reserved_attributes = ['interaction_id',
                               'nu_id',
                               'volume_id',
                               'image_id',
                               'index', 'points', 'sources', 'depositions',
                               'truth_index', 'truth_points',
                               'truth_depositions','truth_depositions_MeV',
                               'sed_index', 'sed_points', 'sed_depositions_MeV']

        processed_args = {'particles': []}
        for key, val in kwargs.items():
            processed_args[key] = val
        for p in particles:
            assert type(p) is TruthParticle
            for key in reserved_attributes:
                if key not in kwargs:
                    init_args[key].append(getattr(p, key))
            processed_args['particles'].append(p)

        _process_truth_interaction_attributes(init_args, processed_args, **kwargs)
        truth_interaction = cls(**processed_args)
        return truth_interaction
    
    def register_larcv_neutrino(self, neutrino=None):
        '''
        Extracts all the relevant attributes from the a
        `larcv::Neutrino` and makes it its own.

        Parameters
        ----------
        neutrino : larcv::Neutrino
            True MC Neutrino Object
        '''
        
        if neutrino is None:
            self.nu_track_id = -1
            for name in self._SCALAR_KEYS:
                if name != 'id':
                    setattr(self, f'nu_{name}', self._SCALAR_KEYS[name])
                else:
                    setattr(self, f'nu_truth_id', self._SCALAR_KEYS[name])
            for name in self._VECTOR_KEYS:
                setattr(self, f'nu_{name}', self._VECTOR_KEYS[name])   
        else:
            self.nu_track_id = neutrino.nu_track_id()
            
            for name in self._SCALAR_KEYS:
                val = getattr(neutrino, name)()
                if name != id:
                    setattr(self, f'nu_{name}', val)
                else:
                    setattr(self, f'nu_truth_id', val)
                
            for name in self._VECTOR_KEYS:
                larcv_vector = getattr(neutrino, name)()
                vector = np.array(
                    [getattr(larcv_vector, a)() for a in ['x', 'y', 'z']])
                setattr(self, f'nu_{name}', vector)
            
        
    def load_larcv_neutrino(self, neutrino_dict):
        '''
        Read saved neutrino information from h5 and restore attributes. 

        Parameters
        ----------
        neutrino_dict : python dict containing larcv::Neutrino information.
        '''
        attribute_keys = list(self._SCALAR_KEYS.keys()) \
                       + list(self._VECTOR_KEYS.keys())
        attribute_keys += ['nu_track_id']
        attribute_keys = [f'nu_{name}' for name in attribute_keys]
        for name in attribute_keys:
            if name in neutrino_dict:
                attr = neutrino_dict[name]
                if type(attr) is bytes:
                    attr = attr.decode()
                setattr(self, name, attr)
        

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


# ------------------------------Helper Functions---------------------------

def _process_truth_interaction_attributes(init_args, processed_args, **kwargs):
    # Call the general function (sets attributes shared with Interaction
    _process_interaction_attributes(init_args, processed_args, **kwargs)

    # Treat the TruthInteraction-specific attributes
    if len(init_args['truth_index']) > 0:
        processed_args['truth_index']           = np.concatenate(init_args['truth_index'])
        processed_args['truth_points']          = np.vstack(init_args['truth_points'])
        processed_args['truth_depositions']     = np.concatenate(init_args['truth_depositions'])
        processed_args['truth_depositions_MeV'] = np.concatenate(init_args['truth_depositions_MeV'])

    if len(init_args['sed_index']):
        processed_args['sed_index']             = np.concatenate(init_args['sed_index'])
        processed_args['sed_points']            = np.vstack(init_args['sed_points'])
        processed_args['sed_depositions_MeV']   = np.concatenate(init_args['sed_depositions_MeV'])
