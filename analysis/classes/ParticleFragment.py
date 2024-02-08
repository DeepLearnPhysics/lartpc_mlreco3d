import numpy as np

from typing import Counter, List, Union
from collections import OrderedDict

from mlreco.utils.globals import SHAPE_LABELS, TRACK_SHP, \
        PID_LABELS, PID_MASSES, PID_TO_PDG
from mlreco.utils.utils import pixel_to_cm
from mlreco.utils.numba_local import cdist


class ParticleFragment:
    '''
    Data structure for managing fragment-level
    full chain output information

    Attributes
    ----------
    id: int
        Unique ID of the particle fragment (different from particle id)
    group_id: int
        Group ID (alias for Particle ID) for which this fragment belongs to.
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
        (N) IDs of voxels that correspondn to the fragment within the image coordinate tensor that
    points : np.dnarray, default np.array([], shape=(0,3))
        (N,3) Set of voxel coordinates that make up this fragment in the input tensor
    depositions : np.ndarray, defaul np.array([])
        (N) Array of energy deposition values for each voxel (rescaled, ADC units)
    is_primary: bool
        If True, then this particle fragment corresponds to
        a primary ionization trajectory within the group of fragments that
        compose a particle.
    '''
    def __init__(self,
                 fragment_id: int = -1,
                 group_id: int = -1,
                 interaction_id: int = -1,
                 image_id: int = -1,
                 volume_id: int = -1,
                 nu_id: int = -1,
                 semantic_type: int = -1,
                 index: np.ndarray = np.empty(0, dtype=np.int64),
                 points: np.ndarray = np.empty((0,3), dtype=np.float32),
                 depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 is_primary: int = -1,
                 start_point: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_point: np.ndarray = -np.ones(3, dtype=np.float32),
                 start_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 length: float = -1.,
                 matched: bool = False,
                 **kwargs):

        # Initialize private attributes to be assigned through setters only
        self._size = None
        self._index = None
        self._depositions = None

        # Initialize attributes
        self.id             = int(fragment_id)
        self.group_id       = group_id
        self.interaction_id = interaction_id
        self.image_id       = image_id
        self.volume_id      = volume_id
        self.semantic_type  = int(semantic_type)
        self.nu_id          = int(nu_id)

        self.index          = index
        self.points         = points
        self.depositions    = depositions

        self.is_primary     = is_primary

        self._start_point    = np.copy(start_point)
        self._end_point      = np.copy(end_point)
        self._start_dir      = np.copy(start_dir)
        self._end_dir        = np.copy(end_dir)
        self.length         = length
        
        # Quantities to be set by the particle matcher
        self.matched             = matched
        self._is_principal_match = False
        self._match              = list(kwargs.get('match', []))
        self._match_overlap       = kwargs.get('match_overlap', OrderedDict())
        if not isinstance(self._match_overlap, dict):
            raise ValueError(f"{type(self._match_overlap)}")

    def __str__(self):
        fmt = "ParticleFragment( Image ID={:<3} | Fragment ID={:<3} | Semantic_type: {:<15}"\
                " | Group ID: {:<3} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2})"
        msg = fmt.format(self.image_id, self.id,
                         SHAPE_LABELS[self.semantic_type] if self.semantic_type in SHAPE_LABELS else "None",
                         self.group_id,
                         self.is_primary,
                         self.interaction_id,
                         self.size,
                         self.volume_id)
        return msg

    def __repr__(self):
        msg = "ParticleFragment(image_id={}, id={}, group_id={}, size={}, shape_id={})".format(
            self.image_id, self.id, self.group_id, self.size, self.semantic_type)
        return msg

    @property
    def size(self):
        '''
        Fragment size (i.e. voxel count) getter. This attribute has no setter,
        as it can only be set by providing a set of voxel indices.
        '''
        return self._size

    @property
    def index(self):
        '''
        Fragment voxel indices getter/setter. The setter also sets
        the fragment size, i.e. the voxel count.
        '''
        return self._index

    @index.setter
    def index(self, index):
        # Count the number of voxels
        self._index = index
        self._size = len(index)

    @property
    def depositions_sum(self):
        '''
        Total amount of charge/energy deposited. This attribute has no setter,
        as it can only be set by providing a set of depositions.
        '''
        return self._depositions_sum

    @property
    def depositions(self):
        '''
        Fragment depositions getter/setter. The setter also sets
        the fragment depositions sum.
        '''
        return self._depositions

    @depositions.setter
    def depositions(self, depositions):
        # Sum all the depositions
        self._depositions = depositions
        self._depositions_sum = np.sum(depositions)
        
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
        
    @property
    def is_principal_match(self):
        return self._is_principal_match
