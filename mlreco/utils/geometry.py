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
        Convert a detector boundary file to useful detector attributes

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
        self.tpcs = np.load(boundaries)

        # Check that the sources file exists, load it
        self.sources = None
        if sources is not None:
            if not os.path.isfile(sources):
                raise FileNotFoundError(f'Could not find sources file: {sources}')
            self.sources = np.load(sources)
            assert self.sources.shape[0] == self.tpcs.shape[0], \
                    'There should be one list of sources per TPC'

        # Build modules from the TPC list (merge touching TPCs)
        self.build_modules()

        # Build an outer volume from the TPC list (find outermost boundaries)
        self.build_detector()

    def build_modules(self, tolerance=1):
        '''
        Merges touching TPCs into modules

        Parameters
        ----------
        tolerance : float, default 1 cm
            Distance within which two TPCs are considered touching
        '''
        # Build an adjacency matrix
        adj_mat = np.ones((len(self.tpcs), len(self.tpcs)))
        for i in range(len(self.tpcs)):
            for j in range(len(self.tpcs)):
                if i < j:
                    adj_mat[i, j] = \
                            (np.abs(self.tpcs[j][:,0] - self.tpcs[i][:,1]) \
                                    < tolerance).any() | \
                            (np.abs(self.tpcs[j][:,1] - self.tpcs[i][:,0]) \
                                    < tolerance).any()
                elif j < i:
                    adj_mat[i, j] = adj_mat[j, i]

        # Build modules
        leftover = np.ones(len(self.tpcs), dtype=bool)
        groups = []
        for i in range(len(self.tpcs)):
            if leftover[i]:
                mask = adj_mat[i].astype(bool)
                groups.append(np.where(mask)[0])
                leftover &= ~mask

        self.modules = np.empty((len(groups), 3, 2))
        for i, g in enumerate(groups):
            self.modules[i,:,0] = np.min(self.tpcs[g], axis=0)[:,0]
            self.modules[i,:,1] = np.max(self.tpcs[g], axis=0)[:,1]

    def build_detector(self):
        '''
        Find outermost boundaries
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
        margin : float
            Minimum distance from a detector wall to be considered contained
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

        # Loop over volumes, make sure the cloud is contained in at least one
        contained = False
        for v in volumes:
            if (points > (v[:,0] + margin)).all() \
                    and (points < (v[:,1] - margin)).all():
                contained = True
                break

        return contained

