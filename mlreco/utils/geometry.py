import os
import pathlib
import numpy as np


class Geometry:
    '''
    Class which handles very basic geometry queries based on a file
    which contains a list of TPC boundaries.

    Attributes
    ----------
    '''

    def __init__(self, detector=None, boundaries=None,
            sources=None, opdets=None):
        '''
        Initializes a detector geometry object.

        The boundary file is a (N_m, N_t, D, 2) np.ndarray where:
        - N_m is the number of modules (or cryostat) in the detector
        - N_t is the number of TPCs per module (or cryostat)
        - D is the number of dimension (always 3)
        - 2 corresponds to the lower/upper boundaries along that axis

        The sources file is a (N_m, N_t, N_s, 2) np.ndarray where:
        - N_s is the number of contributing logical TPCs to a geometry TPC
        - 2 corresponds to the [module ID, tpc ID] of a contributing pair

        The opdets file is a (N_m[, N_t], N_p, 3) np.ndarray where:
        - N_p is the number of optical detectors per module or TPC
        - 3 corresponds to the [x, y, z] optical detector coordinates

        Parameters
        ----------
        detector : str, optional
            Name of a recognized detector to the geometry from
        boundaries : str, optional
            Path to a `.npy` boundary file to load the boundaries from
        sources : str, optional
            Path to a `.npy` source file to load the sources from
        opdets : str, optional
            Path to a `.npy` opdet file to load the opdet coordinates from
        '''
        # If the boundary file is not provided, fetch a default boundary file
        assert detector is not None or boundaries is not None, \
                'Must minimally provide a detector boundary file source'
        if boundaries is None:
            path = pathlib.Path(__file__).parent
            boundaries = os.path.join(path, 'geo',
                    f'{detector.lower()}_boundaries.npy')

        # If the source file is not a file, fetch the default source file
        if sources is None and detector is not None:
            path = pathlib.Path(__file__).parent
            file_path = os.path.join(path, 'geo',
                    f'{detector.lower()}_sources.npy')
            if os.path.isfile(file_path):
                sources = file_path

        # If the opdets file is not a file, fetch the default opdets file
        if opdets is None and detector is not None:
            path = pathlib.Path(__file__).parent
            file_path = os.path.join(path, 'geo',
                    f'{detector.lower()}_opdets.npy')
            if os.path.isfile(file_path):
                opdets = file_path

        # Check that the boundary file exists, load it
        if not os.path.isfile(boundaries):
            raise FileNotFoundError('Could not find boundary ' \
                    f'file: {boundaries}')
        self.boundaries = np.load(boundaries)

        # Check that the sources file exists, load it
        self.sources = None
        if sources is not None:
            if not os.path.isfile(sources):
                raise FileNotFoundError('Could not find sources ' \
                        f'file: {sources}')
            self.sources = np.load(sources)
            assert self.sources.shape[:2] == self.boundaries.shape[:2], \
                    'There should be one list of sources per TPC'

        # Check that the optical detector file exists, load it
        self.opdets = None
        if opdets is not None:
            if not os.path.isfile(opdets):
                raise FileNotFoundError('Could not find opdets ' \
                        f'file: {opdets}')
            self.opdets = np.load(opdets)
            assert self.opdets.shape[:2] == self.boundaries.shape[:2] \
                    or (self.opdets.shape[0] == self.boundaries.shape[0] \
                    and len(self.opdets.shape) == 3), \
                    'There should be one list of opdets per module or TPC'

        # Store the ranges of each TPC in each axis
        self.ranges = np.abs(self.boundaries[...,1]-self.boundaries[...,0])

        # Build a list of TPCs
        self.build_tpcs()

        # Build a list of modules
        self.build_modules()

        # Build an all-encompassing detector object
        self.build_detector()

        # Build cathodes/anodes if the modules share a central cathode
        if self.boundaries.shape[1] == 2:
            self.build_planes()

        # Containment volumes to be defined by the user
        self.cont_volumes = None
        self.cont_use_source = False

    def build_tpcs(self):
        '''
        Flatten out the geometry array to a simple list of TPCs. Also store
        the total number of TPCs.
        '''
        self.tpcs = self.boundaries.reshape(-1, 3, 2)
        self.num_tpcs = len(self.tpcs)

    def build_modules(self):
        '''
        Convert the list of boundaries of TPCs that make up the modules into
        a list of boundaries that encompass each module. Also store the center
        of each module and the total number of moudules.
        '''
        self.modules = np.empty((len(self.boundaries), 3, 2))
        self.centers = np.empty((len(self.boundaries), 3))
        for m, module in enumerate(self.boundaries):
            self.modules[m] = self.merge_volumes(module)
            self.centers[m] = np.mean(self.modules[m], axis=1)

        self.num_modules = len(self.modules)

    def build_detector(self):
        '''
        Convert the list of boundaries of TPCs that make up the detector
        into a single set of overall detector boundaries.
        '''
        self.detector = self.merge_volumes(self.tpcs)

    def build_planes(self):
        '''
        Convert the list of boundaries of TPCs that make up the modules and
        tpcs into a list of cathode plane positions for each module and anode
        plane positions for each TPC. The cathode/anode positions are expressed
        as a simple number pair [axis, position] with axis the drift axis and
        position the cathode position along that axis.

        Also stores a [axis, side] pair for each TPC which tells which of the
        walls of the TPCs is the cathode wall
        '''
        tpc_shape = self.boundaries.shape[:2]
        self.anodes = np.empty(tpc_shape, dtype = object)
        self.cathodes = np.empty(tpc_shape[0], dtype = object)
        self.drift_dirs = np.empty((*tpc_shape, 3))
        self.cathode_wall_ids = np.empty((*tpc_shape, 2), dtype = np.int32)
        self.anode_wall_ids = np.empty((*tpc_shape, 2), dtype = np.int32)
        for m, module in enumerate(self.boundaries):
            # Check that the module is central-cathode style
            assert len(module) == 2, \
                    'A module with < 2 TPCs has no central cathode'

            # Identify the drift axis
            centers = np.mean(module, axis=-1)
            drift_dir = (centers[1] - centers[0])
            drift_dir /= np.linalg.norm(drift_dir)
            axis = np.where(drift_dir)[0]
            assert len(axis) == 1, \
                    'The drift direction is not aligned with an axis, abort'
            axis = axis[0]

            # Store the cathode position
            midpoint = np.sum(centers, axis=0)/2
            self.cathodes[m] = [axis, midpoint[axis]]

            # Store the wall ID of each TPC that makes up the module
            for t, tpc in enumerate(module):
                # Store which side the anode/cathode are on
                side = int(centers[t][axis] - midpoint[axis] < 0.)
                self.cathode_wall_ids[m, t] = [axis, side]
                self.anode_wall_ids[m, t] = [axis, 1-side]

                # Store the position of the anode for each TPC
                anode_pos = self.boundaries[m, t, axis, 1-side]
                self.anodes[m, t] = [axis, anode_pos]

                # Store the drift direction for each TPC
                self.drift_dirs[m, t] = (-1)**side * drift_dir

    def get_contributors(self, sources):
        '''
        Get the list of [module ID, tpc ID] pairs that contributed to a
        particle or interaction object, as defined in this geometry.

        Parameters
        ----------
        sources : np.ndarray
            (S, 2) : List of [module ID, tpc ID] pairs that created
            the point cloud (as defined upstream)

        Returns
        -------
        List[np.ndarray]
            (2, N_t) Pair of arrays: the first contains the list of
            contributing modules, the second of contributing tpcs.
        '''
        sources = np.unique(sources, axis=0)
        contributor_mask = np.zeros(self.boundaries.shape[:2], dtype=bool)
        for m, module_source in enumerate(self.sources):
            for t, tpc_source in enumerate(module_source):
                for source in sources:
                    if (tpc_source == source).all(axis=-1).any(axis=-1):
                        contributor_mask[m, t] = True
                        break

        return np.where(contributor_mask)

    def get_tpc_index(self, sources, module_id, tpc_id):
        '''
        Get the list of indices of points that belong to a specify
        [module ID, tpc ID] pair.

        Parameters
        ----------
        sources : np.ndarray
            (S, 2) : List of [module ID, tpc ID] pairs that created
            the point cloud (as defined upstream)
        module_id : int
            ID of the module
        tpc_id : int
            ID of the TPC within the module

        Returns
        -------
        np.ndarray
            (N) Index of points that belong to that TPC
        '''
        mask = np.zeros(len(sources), dtype=bool)
        for source in self.sources[module_id, tpc_id]:
            mask |= (sources == source).all(axis=-1)

        return np.where(mask)[0]

    def get_closest_tpc_indexes(self, points):
        '''
        For each TPC, get the list of points that live closer to it
        than any other TPC in the detector.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates

        Returns
        -------
        List[np.ndarray]
            List of index of points that belong to each TPC
        '''
        # Compute the distance from the points to each TPC
        distances = np.empty((self.num_tpcs, len(points)))
        for t in range(self.num_tpcs):
            module_id, tpc_id = t // self.num_modules, t % self.num_modules
            offsets = self.get_tpc_offsets(points, module_id, tpc_id)
            distances[t] = np.linalg.norm(offsets, axis=1)

        # For each TPC, append the list of point indices associated with it
        tpc_indexes = []
        argmins = np.argmin(distances, axis=0)
        for t in range(self.num_tpcs):
            tpc_indexes.append(np.where(argmins == t)[0])

        return tpc_indexes

    def get_tpc_offsets(self, points, module_id, tpc_id):
        '''
        Compute how far each point is from a TPC volume.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) : Point coordinates
        module_id : int
            ID of the module
        tpc_id : int
            ID of the TPC within the module

        Returns
        -------
        np.ndarray
            (N, 3) Offsets w.r.t. to the TPC location
        '''
        # Compute the axis-wise distances of each point to each boundary
        tpc = self.boundaries[module_id, tpc_id]
        ranges = self.ranges[module_id, tpc_id]
        dists = points[..., None] - tpc

        # If a point is between two boundaries, the distance is 0. If it is
        # outside, the distance is that of the closest boundary
        signs = (np.sign(dists[..., 0]) + np.sign(dists[..., 1]))/2
        offsets = signs * np.min(np.abs(dists), axis=-1)

        return offsets

    def get_min_tpc_offset(self, points, module_id, tpc_id):
        '''
        Get the minimum offset to apply to a point cloud to bring it
        within the boundaries of a TPC.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) : Point coordinates
        module_id : int
            ID of the module
        tpc_id : int
            ID of the TPC within the module

        Returns
        -------
        np.ndarray
            (3) Offsets w.r.t. to the TPC location
        '''
        # Compute the distance for each point, get the maximum necessary offset
        offsets = self.get_tpc_offsets(points, module_id, tpc_id)
        offsets = offsets[np.argmax(np.abs(offsets), axis=0), np.arange(3)]

        return offsets

    def translate(self, points, source_id, target_id):
        '''
        Moves a point cloud from one module to another one

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        source_id: int
            Module ID from which to move the point cloud
        target_id : int
            Module ID to which to move the point cloud

        Returns
        -------
        np.ndarray
            (N, 3) Set of translated point coordinates
        '''
        # If the source and target are the same, nothing to do here
        if target_id == source_id:
            return points

        # Fetch the inter-module shift
        offset = self.centers[target_id] - self.centers[source_id]

        # Translate
        return np.copy(points) + offset

    def check_containment(self, points, sources = None,
            allow_multi_module = False):
        '''
        Check whether a point cloud comes within some distance of the
        boundaries of a certain subset of detector volumes, depending on
        the mode.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        sources : np.ndarray, optional
            (S, 2) : List of [module ID, tpc ID] pairs that created the
            point cloud
        allow_multi_module : bool, default False
            Whether to allow particles/interactions to span multiple modules

        Returns
        -------
        bool
            `True` if the particle is contained, `False` if not
        '''
        # If the containment volumes are not defined, throw
        if self.cont_volumes is None:
            raise ValueError('Must call `define_containment_volumes` first')

        # If sources are provided, only consider source volumes
        if self.cont_use_source:
            # Get the contributing TPCs
            assert len(points) == len(sources), \
                    'Need to provide sources to make a source-based check'
            contributors = self.get_contributors(sources)
            if not allow_multi_module and len(np.unique(contributors[0])) > 1:
                return False

            # Define the smallest box containing all contributing TPCs
            index = contributors[0] * self.boundaries.shape[1] + contributors[1]
            volume = self.merge_volumes(self.cont_volumes[index])
            volumes = [volume]
        else:
            volumes = self.cont_volumes

        # Loop over volumes, make sure the cloud is contained in at least one
        contained = False
        for v in volumes:
            if (points > v[:,0]).all() and (points < v[:,1]).all():
                contained = True
                break

        return contained

    def define_containment_volumes(self, margin, \
            cathode_margin = None, mode = 'module'):
        '''
        This function defines a list of volumes to check containment against.
        If the containment is checked against a constant volume, it is more
        efficient to call this function once and call `check_containment`
        reapitedly after.

        Parameters
        ----------
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of walls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is
              specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'tpc', makes sure it is contained within a single TPC
            - If 'module', makes sure it is contained within a single module
            - If 'detector', makes sure it is contained within in the detector
            - If 'source', use the origin of voxels to determine which TPC(s)
              contributed to them, and define volumes accordingly
        '''
        # Translate the margin parameter to a (3,2) matrix
        if np.isscalar(margin):
            margin = np.full((3,2), margin)
        elif len(np.array(margin).shape) == 1:
            assert len(margin) == 3, \
                    'Must provide one value per axis'
            margin = np.repeat([margin], 2, axis=0).T
        else:
            assert np.array(margin).shape == (3,2), \
                    'Must provide two values per axis'
            margin = np.copy(margin)

        # Establish the volumes to check against
        self.cont_volumes = []
        if mode in ['tpc', 'source']:
            for m, module in enumerate(self.boundaries):
                for t, tpc in enumerate(module):
                    vol = self.adapt_volume(tpc, margin, \
                            cathode_margin, m, t)
                    self.cont_volumes.append(vol)
            self.cont_use_source = mode == 'source'
        elif mode == 'module':
            for m in self.modules:
                vol = self.adapt_volume(m, margin)
                self.cont_volumes.append(vol)
            self.cont_use_source = False
        elif mode == 'detector':
            vol = self.adapt_volume(self.detector, margin)
            self.cont_volumes.append(vol)
            self.cont_use_source = False
        else:
            raise ValueError(f'Containement check mode not recognized: {mode}')

        self.cont_volumes = np.array(self.cont_volumes)

    def adapt_volume(self, ref_volume, margin, cathode_margin = None,
            module_id = None, tpc_id = None):
        '''
        Apply margins from a given volume. Takes care of subtleties
        associated with the cathode, if requested.

        Parameters
        ----------
        ref_volume : np.ndarray
            (3, 2) Array of volume boundaries
        margin : np.ndarray
            Minimum distance from a detector wall to be considered contained as
            [[x_low,x_up], [y_low,y_up], [z_low,z_up]], i.e. distance is
            specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        module_id : int, optional
            ID of the module
        tpc_id : int, optional
            ID of the TPC within the module

        Returns
        -------
        np.ndarray
            (3, 2) Updated array of volume boundaries
        '''
        # Reduce the volume according to the margin
        volume = np.copy(ref_volume)
        volume[:,0] += margin[:,0]
        volume[:,1] -= margin[:,1]

        # If a cathode margin is provided, adapt the cathode wall differently
        if cathode_margin is not None:
            axis, side = self.cathode_wall_ids[module_id, tpc_id]
            flip = (-1) ** side
            volume[axis, side] += flip * (cathode_margin - margin[axis, side])

        return volume

    @staticmethod
    def merge_volumes(volumes):
        '''
        Given a list of volumes and their boundaries, find the smallest box
        that encompass all volumes combined.

        Parameters
        ----------
        volumes : np.ndarray
            (N, 3, 2) List of volume boundaries

        Returns
        -------
        np.ndarray
            (3, 2) Boundaries of the combined volume
        '''
        volume = np.empty((3, 2))
        volume[:,0] = np.min(volumes, axis=0)[:,0]
        volume[:,1] = np.max(volumes, axis=0)[:,1]

        return volume
