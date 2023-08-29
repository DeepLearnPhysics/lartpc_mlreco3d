import numpy as np

from typing import Counter, List, Union
from collections import OrderedDict

from mlreco.utils.globals import SHAPE_LABELS, PID_LABELS, PID_TO_PDG
from mlreco.utils.utils import pixel_to_cm

class Particle:
    '''
    Data structure for managing particle-level
    full chain output information.

    Attributes
    ----------
    id : int, default -1
        Unique ID of the particle
    fragment_ids : np.ndarray, default np.array([])
        List of ParticleFragment IDs that make up this particle
    num_fragments: int
        Total number of fragments in this particle
    interaction_id : int, default -1
        ID of the particle's parent interaction
    nu_id : int, default -1
        ID of the particle's parent neutrino
    volume_id : int, default -1
        ID of the detector volume the particle lives in
    image_id : int, default -1
        ID of the image the particle lives in
    size : int
        Total number of voxels that belong to this particle
    index : np.ndarray, default np.array([])
        (N) IDs of voxels that correspond to the particle within the input tensor
    points : np.dnarray, default np.array([], shape=(0,3))
        (N,3) Set of voxel coordinates that make up this particle in the input tensor
    depositions : np.ndarray, defaul np.array([])
        (N) Array of charge deposition values for each voxel
    depositions_sum : float
        Sum of energy depositions
    semantic_type : int, default -1
        Semantic type (shower (0), track (1),
        michel (2), delta (3), low energy (4)) of this particle.
    pid : int
        PDG Type (Photon (0), Electron (1), Muon (2),
        Charged Pion (3), Proton (4)) of this particle
    pid_scores : np.ndarray
        (P) Array of softmax scores associated with each of particle class
    pdg_code : int
        PDG code corresponding to the PID number
    is_primary : bool
        Indicator whether this particle is a primary from an interaction
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    start_point : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle start point
    end_point : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle end point
    start_dir : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle direction estimate w.r.t. the start point
    end_dir : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle direction estimate w.r.t. the end point
    energy_sum : float, default -1
        Energy reconstructed from the particle deposition sum
    csda_kinetic_energy : float, default -1
        Kinetic energy reconstructed from the particle range
    momentum_mcs : float, default -1
        Momentum reconstructed using the MCS method
    match : List[int]
        List of TruthParticle IDs for which this particle is matched to
    units : str, default 'px'
        Units in which coordinates are expressed
    gap_length : float, default -1 
        inter cluster gap length usually divided by track length

    '''

    # Attributes that specify coordinates
    _COORD_ATTRS = ['points', 'start_point', 'end_point']

    def __init__(self,
                 group_id: int = -1,
                 fragment_ids: np.ndarray = np.empty(0, dtype=np.int64),
                 interaction_id: int = -1,
                 nu_id: int = -1,
                 volume_id: int = -1,
                 image_id: int = -1,
                 semantic_type: int = -1,
                 index: np.ndarray = np.empty(0, dtype=np.int64),
                 points: np.ndarray = np.empty(0, dtype=np.float32),
                 depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 pid_scores: np.ndarray = -np.ones(len(PID_LABELS), dtype=np.float32),
                 primary_scores: np.ndarray = -np.ones(2, dtype=np.float32),
                 start_point: np.ndarray = np.full(3, -np.inf),
                 end_point: np.ndarray = np.full(3, -np.inf),
                 start_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 length: float = -1.,
                 csda_kinetic_energy: float = -1.,
                 momentum_mcs: float = -1.,
                 matched: bool = False,
                 is_contained: bool = False,
                 is_primary: bool = False,
                 units: str = 'px', **kwargs):

        # Initialize private attributes to be assigned through setters only
        self._num_fragments   = None
        self._index           = None
        self._depositions     = None
        self._depositions_sum = -1
        self._pid             = -1
        self._size            = -1
        self._is_primary      = is_primary
        self._units           = units

        # Initialize attributes
        self.id             = int(group_id)
        self.fragment_ids   = fragment_ids
        self.interaction_id = int(interaction_id)
        self.nu_id          = int(nu_id)
        self.image_id       = int(image_id)
        self.volume_id      = int(volume_id)
        self.semantic_type  = int(semantic_type)

        self.index          = index
        self.points         = points
        self.depositions    = depositions

        self.pdg_code       = -1
        
        self.pid_scores     = pid_scores
        self.primary_scores = primary_scores
        
        # Quantities to be set during post_processing
        self._start_point         = start_point
        self._end_point           = end_point
        self._start_dir           = start_dir
        self._end_dir             = end_dir
        self.length               = length
        self.csda_kinetic_energy  = csda_kinetic_energy
        self.momentum_mcs         = momentum_mcs
        self.is_contained         = is_contained

        # Quantities to be set by the particle matcher
        self.matched             = matched
        self._is_principal_match = False
        self._match              = list(kwargs.get('match', []))
        self._match_overlap       = kwargs.get('match_overlap', OrderedDict())
        if not isinstance(self._match_overlap, dict):
            raise ValueError(f"{type(self._match_overlap)}")

    @property
    def is_principal_match(self):
        return self._is_principal_match

    @property
    def start_point(self):
        return self._start_point

    @start_point.setter
    def start_point(self, value):
        assert value.shape == (3,)
        if (np.abs(value) < 1e10).all():
            # Only set start_point if not bogus value
            self._start_point = value

    @property
    def end_point(self):
        return self._end_point

    @end_point.setter
    def end_point(self, value):
        assert value.shape == (3,)
        if (np.abs(value) < 1e10).all():
            # Only set start_point if not bogus value
            self._end_point = value

    @property
    def start_dir(self):
        return self._start_dir

    @start_dir.setter
    def start_dir(self, value):
        assert value.shape == (3,)
        self._start_dir = value

    @property
    def end_dir(self):
        return self._end_dir

    @end_dir.setter
    def end_dir(self, value):
        assert value.shape == (3,)
        self._end_dir = value

    @property
    def is_primary(self):
        return bool(self._is_primary)

    @is_primary.setter
    def is_primary(self, value):
        self._is_primary = value

    @property
    def match(self):
        self._match = list(self._match_overlap.keys())
        return np.array(self._match, dtype=np.int64)

    @property
    def match_overlap(self):
        return np.array(list(self._match_overlap.values()), dtype=np.float32)

    @match_overlap.setter
    def match_overlap(self, value):
        assert type(value) is OrderedDict
        self._match_overlap = value

    def clear_match_info(self):
        self._match = []
        self._match_overlap = OrderedDict()
        self.matched = False

    def __repr__(self):
        msg = "Particle(image_id={}, id={}, pid={}, size={})".format(self.image_id, self.id, self._pid, self.size)
        return msg

    def __str__(self):
        fmt = "Particle( Image ID={:<3} | Particle ID={:<3} | Semantic_type: {:<11}"\
                " | PID: {:<8} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2} | Match: {:<3})"
        msg = fmt.format(self.image_id, self.id,
                         SHAPE_LABELS[self.semantic_type] if self.semantic_type in SHAPE_LABELS else "None",
                         PID_LABELS[self.pid] if self.pid in PID_LABELS else "None",
                         int(self.is_primary),
                         self.interaction_id,
                         self.size,
                         self.volume_id,
                         str(self.match))
        return msg

    @property
    def num_fragments(self):
        '''
        Number of particle fragments getter. This attribute has no setter,
        as it can only be set by providing a list of fragment ids.
        '''
        return self._num_fragments

    @property
    def fragment_ids(self):
        '''
        ParticleFragment indices getter/setter. The setter also sets
        the number of fragments.
        '''
        return self._fragment_ids

    @fragment_ids.setter
    def fragment_ids(self, fragment_ids):
        # Count the number of fragments
        self._fragment_ids = fragment_ids
        self._num_fragments = len(fragment_ids)

    @property
    def size(self):
        '''
        Particle size (i.e. voxel count) getter. This attribute has no setter,
        as it can only be set by providing a set of voxel indices.
        '''
        return int(self._size)

    @property
    def index(self):
        '''
        Particle voxel indices getter/setter. The setter also sets
        the particle size, i.e. the voxel count.
        '''
        return self._index

    @index.setter
    def index(self, index):
        # Count the number of voxels
        self._index = np.array(index, dtype=np.int64)
        self._size = len(index)

    @property
    def depositions_sum(self):
        '''
        Total amount of charge/energy deposited. This attribute has no setter,
        as it can only be set by providing a set of depositions.
        '''
        return float(self._depositions_sum)

    @property
    def depositions(self):
        '''
        Particle depositions getter/setter. The setter also sets
        the particle depositions sum.
        '''
        return self._depositions

    @depositions.setter
    def depositions(self, depositions):
        # Sum all the depositions
        self._depositions = depositions
        self._depositions_sum = np.sum(depositions)

    @property
    def pid_scores(self):
        '''
        Particle ID scores getter/setter. The setter converts the
        scores to an particle ID prediction through argmax.
        '''
        return self._pid_scores

    @pid_scores.setter
    def pid_scores(self, pid_scores):
        self._pid_scores = pid_scores
        # If no PID scores are providen, the PID is unknown
        if pid_scores[0] < 0.:
            self._pid = -1
            self._pdg_code = -1
        else:
        # Store the PID scores
            self._pid = int(np.argmax(pid_scores))
            self.pdg_code = int(PID_TO_PDG[self._pid])

    @property
    def pid(self):
        return int(self._pid)

    @pid.setter
    def pid(self, value):
        if value not in PID_LABELS:
            print("WARNING: PID {} not in PID_LABELS".format(value))
            self._pid = -1
        else:
            self._pid = value

    @property
    def primary_scores(self):
        '''
        Primary ID scores getter/setter. The setter converts the
        scores to a primary prediction through argmax.
        '''
        return self._primary_scores

    @primary_scores.setter
    def primary_scores(self, primary_scores):
        # If no primary scores are given, the primary status is unknown
        if primary_scores[0] < 0.:
            self._primary_scores = primary_scores
            self._is_primary = False

        # Store the PID scores and give a best guess
        self._primary_scores = primary_scores
        self._is_primary = bool(np.argmax(primary_scores))

    def convert_to_cm(self, meta):
        '''
        Converts the units of all coordinate attributes to cm.
        '''
        assert self._units == 'px'
        for attr in self._COORD_ATTRS:
            setattr(self, attr, pixel_to_cm(getattr(self, attr), meta))

        self._units = 'cm'

    @property
    def units(self):
        return self._units
