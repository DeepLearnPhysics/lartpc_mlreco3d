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

    def __init__(self, boundaries, sources=None):
        '''
        Convert a detector boundary file to useful detector attributes.

        The boundary file is a (N_m, N_t, D, 2) np.ndarray where:
        - N_m is the number of modules (or cryostat) in the detector
        - N_t is the number of TPCs per module (or cryostat)
        - D is the number of dimension (always 3)
        - 2 corresponds to the lower/upper boundaries along that axis

        Parameters
        ----------
        boundaries : str
            Name of a recognized detector to get the geometry from or path
            to a `.npy` boundary file to load the boundaries from.
        sources : str, optional
            Name of a recognized detector to get the sources from or path
            to a `.npy` source file to load the sources from.
        '''
        # If the boundaries are not a file, fetch a default boundary file
        if not os.path.isfile(boundaries):
            path = pathlib.Path(__file__).parent
            boundaries = os.path.join(path, 'geo', f'{boundaries.lower()}_boundaries.npy')

        # If the source file is not a file, fetch the default source file
        if sources is not None and not os.path.isfile(sources):
            path = pathlib.Path(__file__).parent
            sources = os.path.join(path, 'geo', f'{sources.lower()}_sources.npy')

        # Check that the boundary file exists, load it
        if not os.path.isfile(boundaries):
            raise FileNotFoundError(f'Could not find boundary file: {boundaries}')
        self.boundaries = np.load(boundaries)

        # Check that the sources file exists, load it
        self.sources = None
        if sources is not None:
            if not os.path.isfile(sources):
                raise FileNotFoundError(f'Could not find sources file: {sources}')
            self.sources = np.load(sources)
            assert self.sources.shape[:2] == self.boundaries.shape[:2], \
                    'There should be one list of sources per TPC'

        # Build TPCs
        self.build_tpcs()

        # Build modules
        self.build_modules()

        # Build detector
        self.build_detector()

        # Build cathodes if the modules share a central cathode
        if self.boundaries.shape[1] == 2:
            self.build_cathodes()

    def build_tpcs(self):
        '''
        Flatten out the geometry array to a simple list of TPCs.
        '''
        self.tpcs = self.boundaries.reshape(-1, 3, 2)

    def build_modules(self):
        '''
        Convert the list of boundaries of TPCs that make up the modules into
        a list of boundaries that encompass each module.
        '''
        self.modules = np.empty((len(self.boundaries), 3, 2))
        for m, module in enumerate(self.boundaries):
            self.modules[m] = self.merge_volumes(module)
        self.ranges = np.abs(self.boundaries[...,1]-self.boundaries[...,0])

    def build_detector(self):
        '''
        Convert the list of boundaries of TPCs that make up the detector
        into a single set of overall detector boundaries.
        '''
        self.detector = self.merge_volumes(self.tpcs)

    def build_cathodes(self):
        '''
        Convert the list of boundaries of TPCs that make up the modules into
        a list cathode plane positions for each module. The cathode position
        is expressed a simple number pair [axis, position] with axis the drift
        axis and position the cathode position along that axis/
        '''
        self.cathodes = []
        for m, module in enumerate(self.boundaries):
            # Check that the module is central-cathode style
            assert len(module) == 2, \
                    'A module with < 2 TPCs has no central cathode'

            # Define the cathode position
            centers = np.mean(module, axis=-1)
            axis = np.where(centers[1] - centers[0])[0]
            assert len(axis) == 1, \
                    'The drift direction is not aligned with an axis, abort'
            axis = axis[0]
            midpoint = np.sum(centers, axis=0)/2
            self.cathodes.append([axis, midpoint[axis]])

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

    def get_tpc_offset(self, points, module_id, tpc_id):
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
        # Compute the axis-wise distances of each point to each boundary
        tpc = self.boundaries[module_id, tpc_id]
        ranges = self.ranges[module_id, tpc_id]
        dists = points[..., None] - tpc

        # Pick the farthest away point for each axis, restrict distances
        max_ids = np.argmax(np.max(np.abs(dists), axis=-1), axis=0)
        dists = dists[max_ids, np.arange(3)]

        # Compute the necessary offsets
        max_ids = np.argmax(np.abs(dists), axis=-1)
        offsets = dists[np.arange(3), max_ids]
        offsets = np.sign(offsets) * np.clip(np.abs(offsets)-ranges, 0., np.inf)

        return offsets

    def check_containment(self, points, margin, sources=None, mode='module'):
        '''
        Check whether a point cloud comes within some distance of the boundaries
        of a certain subset of detector volumes, depending on the mode.

        If a list of sources is provided, the `mode` is ignored and the
        containement is checked against the list of TPCs that contributed
        to the point cloud only.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of falls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is specified
              individually of each wall.
        sources : np.ndarray, optional
            (S, 2) : List of [module ID, tpc ID] pairs that created the point cloud
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'detector', makes sure is is contained within the outermost walls
            - If 'module', makes sure it is contained within a single module
            - If 'tpc', makes sure it is contained within a single tpc

        Returns
        -------
        bool
            `True` if the particle is contained, `False` if not
        '''
        # Establish the volumes to check against
        if sources is not None:
            contributors = self.get_contributors(sources)
            tpcs    = self.boundaries[contributors]
            volume  = self.merge_volumes(tpcs)
            volumes = [volume]
        elif mode == 'detector':
            volumes = [self.detector]
        elif mode == 'module':
            volumes = self.modules
        elif mode == 'tpc':
            volumes = self.tpcs
        else:
            raise ValueError(f'Containement check mode not recognized: {mode}')

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
            margin = np.array(margin)

        # Loop over volumes, make sure the cloud is contained in at least one
        contained = False
        for v in volumes:
            if (points > (v[:,0] + margin[:,0])).all() \
                    and (points < (v[:,1] - margin[:,1])).all():
                contained = True
                break

        return contained

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
