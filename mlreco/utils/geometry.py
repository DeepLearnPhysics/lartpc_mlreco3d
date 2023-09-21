import os
import pathlib
import numpy as np


class Geometry:
    '''
    Class which handles very basic geometry queries based on a file
    which contains a list of TPC boundaries.
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
            self.sources = np.vstack(self.sources)

        # Build TPCs
        self.build_tpcs()

        # Build modules
        self.build_modules()

        # Build detector
        self.build_detector()

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
        for i, m in enumerate(self.boundaries):
            self.modules[i][:,0] = np.min(m, axis=0)[:,0]
            self.modules[i][:,1] = np.max(m, axis=0)[:,1]

    def build_detector(self):
        '''
        Convert the list of boundaries of TPCs that make up the detector
        into a single set of overall detector boundaries.
        '''
        self.detector = np.empty((3, 2))
        self.detector[:,0] = np.min(self.tpcs, axis=0)[:,0]
        self.detector[:,1] = np.max(self.tpcs, axis=0)[:,1]

    def check_containment(self, points, margin, sources=None, mode='module'):
        """
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
        """
        # Establish the volumes to check against
        if sources is not None:
            sources = np.unique(sources, axis=0)
            mask    = np.zeros(len(self.tpcs), dtype=bool)
            for s in sources:
                mask |= (self.sources == s).all(axis=-1).any(axis=-1)
            tpcs = self.tpcs[mask]
            volume = np.empty((3, 2))
            volume[:,0] = np.min(tpcs, axis=0)[:,0]
            volume[:,1] = np.max(tpcs, axis=0)[:,1]
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

