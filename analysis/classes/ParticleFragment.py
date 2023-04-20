import numpy as np

from mlreco.utils.globals import SHAPE_LABELS

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
                 semantic_type: int = -1,
                 index: np.ndarray = np.empty(0, dtype=np.int64),
                 depositions: np.ndarray = np.empty(0, dtype=np.float32),
                 is_primary: int = -1,
                 start_point: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_point: np.ndarray = -np.ones(3, dtype=np.float32),
                 start_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_dir: np.ndarray = -np.ones(3, dtype=np.float32)):

        # Initialize private attributes to be assigned through setters only
        self._size = None
        self._index = None
        self._depositions = None

        # Initialize attributes
        self.id             = fragment_id
        self.group_id       = group_id
        self.interaction_id = interaction_id
        self.image_id       = image_id
        self.volume_id      = volume_id
        self.semantic_type  = semantic_type

        self.index          = index
        self.depositions    = depositions

        self.is_primary     = is_primary

        self.start_point    = start_point
        self.end_point      = end_point
        self.start_dir      = start_dir
        self.end_dir        = end_dir

    def __repr__(self):
        fmt = "ParticleFragment( Image ID={:<3} | Fragment ID={:<3} | Semantic_type: {:<15}"\
                " | Group ID: {:<3} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2})"
        msg = fmt.format(self.image_id, self.id,
                         SHAPE_LABELS[self.semantic_type] if self.semantic_type in list(range(len(SHAPE_LABELS))) else "None",
                         self.group_id,
                         self.is_primary,
                         self.interaction_id,
                         self.size,
                         self.volume_id)
        return msg

    def __str__(self):
        return self.__repr__()

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
        return self._size

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
