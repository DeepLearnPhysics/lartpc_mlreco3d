import numpy as np

from typing import Counter, List, Union
from collections import OrderedDict

from mlreco.utils.globals import SHAPE_LABELS, TRACK_SHP, \
        PID_LABELS, PID_MASSES, PID_TO_PDG
from mlreco.utils.utils import pixel_to_cm
from mlreco.utils.numba_local import cdist

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
    sources : np.ndarray, default np.array([], shape=(0,2))
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs
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
    start_point : np.ndarray, default np.array([-inf, -inf, -inf])
        (3) Particle start point
    end_point : np.ndarray, default np.array([-inf, -inf, -inf])
        (3) Particle end point
    start_dir : np.ndarray, default np.array([-inf, -inf, -inf])
        (3) Particle direction estimate w.r.t. the start point
    end_dir : np.ndarray, default np.array([-inf, -inf, -inf])
        (3) Particle direction estimate w.r.t. the end point
    length : float, default -1
        Length of the particle (only assigned to track objects)
    is_contained : bool
        Indicator whether this particle is contained or not
    is_valid : bool, default True
        Indicator whether this particle counts towards an interaction topology
    calo_ke : float, default -1
        Kinetic energy reconstructed from the energy depositions alone
    csda_ke : float, default -1
        Kinetic energy reconstructed from the particle range
    mcs_ke  : float, default -1
        Kinetic energy reconstructed using the MCS method
    match : List[int]
        List of TruthParticle IDs for which this particle is matched to
    match_overlap : List[float]
        List of match overlaps (in terms of IoU) between the particle and its matches
    units : str, default 'px'
        Units in which coordinates are expressed
    '''

    # Attributes that specify coordinates
    _COORD_ATTRS = ['points', 'start_point', 'end_point']

    def __init__(self,
                 group_id: int = -1,
                 fragment_ids: np.ndarray = np.empty(0, dtype=np.int64),
                 interaction_id: int = -1,
                 nu_id: int = -1,
                 pid: int = -1,
                 volume_id: int = -1,
                 image_id: int = -1,
                 semantic_type: int = -1,
                 index: np.ndarray = np.empty(0, dtype=np.int64),
                 points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 sources: np.ndarray = np.empty((0,2), dtype=np.float32),
                 depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 pid_scores: np.ndarray = -np.ones(len(PID_LABELS), dtype=np.float32),
                 primary_scores: np.ndarray = -np.ones(2, dtype=np.float32),
                 start_point: np.ndarray = np.full(3, -np.inf, dtype=np.float32),
                 end_point: np.ndarray = np.full(3, -np.inf, dtype=np.float32),
                 start_dir: np.ndarray = np.full(3, -np.inf, dtype=np.float32),
                 end_dir: np.ndarray = np.full(3, -np.inf, dtype=np.float32),
                 length: float = -1.,
                 calo_ke: float = -1.,
                 csda_ke: float = -1.,
                 mcs_ke: float = -1.,
                 matched: bool = False,
                 is_primary: bool = False,
                 is_contained: bool = False,
                 is_valid: bool = True,
                 is_ccrosser: bool = False,
                 coffset: float = -np.inf,
                 units: str = 'px', **kwargs):

        # Initialize private attributes to be assigned through setters only
        self._num_fragments   = None
        self._index           = None
        self._depositions     = None
        self._depositions_sum = -1
        self._pid             = pid
        self._size            = -1
        self._is_primary      = is_primary
        self._units           = units
        if type(units) is bytes:
            self._units = units.decode()

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
        self.sources        = sources
        self.depositions    = depositions

        self.pdg_code       = -1

        self.pid_scores     = np.copy(pid_scores)
        self.primary_scores = np.copy(primary_scores)

        # Quantities to be set during post_processing
        self._start_point = np.copy(start_point)
        if self.semantic_type == TRACK_SHP:
            self._end_point = np.copy(end_point)
        else:
            self._end_point = np.full(3, -np.inf, dtype=np.float32)

        self._start_dir   = np.copy(start_dir)
        self._end_dir     = np.copy(end_dir)
        self.length       = length
        self.calo_ke      = calo_ke
        self.csda_ke      = csda_ke
        self.mcs_ke       = mcs_ke
        self.is_contained = is_contained
        self.is_valid     = is_valid
        self.is_ccrosser  = is_ccrosser
        self.coffset      = coffset

        # Quantities to be set by the particle matcher
        self.matched             = matched
        self._is_principal_match = False
        self._match              = list(kwargs.get('match', []))
        self._match_overlap       = kwargs.get('match_overlap', OrderedDict())
        if not isinstance(self._match_overlap, dict):
            raise ValueError(f"{type(self._match_overlap)}")

    def merge(self, particle):
        '''
        Merge another particle object into this one.

        Info
        ----
        This scrip can only merge two track objects with well
        defined start and end points.
        '''
        # Check that both particles being merged are tracks
        assert self.semantic_type == TRACK_SHP \
                and particle.semantic_type == TRACK_SHP, \
                'Can only merge two track particles'

        # Stack the two particle array attributes together
        for attr in ['index', 'depositions']:
            val = np.concatenate([getattr(self, attr), getattr(particle, attr)])
            setattr(self, attr, val)
        for attr in ['points', 'sources']:
            val = np.vstack([getattr(self, attr), getattr(particle, attr)])
            setattr(self, attr, val)

        # Select end points and end directions appropriately
        points_i = np.vstack([self.start_point, self.end_point])
        points_j = np.vstack([particle.start_point, particle.end_point])
        dirs_i = np.vstack([self.start_dir, self.end_dir])
        dirs_j = np.vstack([particle.start_dir, particle.end_dir])

        dists = cdist(points_i, points_j)
        max_i, max_j = np.unravel_index(np.argmax(dists), dists.shape)

        self.start_point = points_i[max_i]
        self.end_point = points_j[max_j]
        self.start_dir = dirs_i[max_i]
        self.end_dir = dirs_j[max_j]

        # If one of the two particles is a primary, the new one is
        if particle.primary_scores[-1] > self.primary_scores[-1]:
            self.primary_scores = particle.primary_scores

        # For PID, pick the most confident prediction (could be better...)
        if np.max(particle.pid_scores) > np.max(self.pid_scores):
            self.pid_scores = particle.pid_scores

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
            self._start_point = value.astype(np.float32)

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
        msg = "Particle(image_id={}, id={}, pid={}, size={})".format(
            self.image_id, self.id, self._pid, self.size)
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

        Warning: If <pid_scores> are provided by either the constructor or
        the pid_scores.setter, it will override the current pid.
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
            if value != -1:
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

    @property
    def ke(self):
        '''
        Best-guess kinetic energy in MeV
        '''
        if self.semantic_type != TRACK_SHP:
            # If a particle is not a track, can only use calorimetry
            return self.calo_ke
        else:
            # If a particle is a track, pick CSDA for contained tracks and
            # pick MCS for uncontained tracks, unless specified otherwise
            if self.is_contained and self.csda_ke > 0.:
                return self.csda_ke
            elif not self.is_contained and self.mcs_ke > 0.:
                return self.mcs_ke
            else:
                return self.calo_ke

    @property
    def momentum(self):
        '''
        Best-guess momentum in MeV/c
        '''
        ke = self.ke
        if ke > 0. and self.start_dir[0] != -np.inf:
            mass = PID_MASSES[self.pid]
            mom = np.sqrt(ke**2 + 2*ke*mass)
            return mom * self.start_dir
        else:
            return np.full(3, -np.inf, dtype=np.float32)

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
