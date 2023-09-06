import os
import pathlib
import numpy as np


class Geometry:
    '''
    Class which handles very basic geometry queries based on a file
    which contains a list of TPC boundaries.
    '''

    def __init__(self, detector=None, boundary_file=None):
        '''
        Convert a detector boundary file to useful detector attributes

        Parameters
        ----------
        detector : str, default 'icarus'
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        '''
        # Should either provide the detector name or a boundary file
        if detector is None and boundary_file is None:
            raise ValueError('Provide either the name of the detector or a boundary file')

        # If a detector name is provided, fetch a default boundary file
        if detector is not None and boundary_file is None:
            path = pathlib.Path(__file__).parent
            boundary_file = os.path.join(path, 'geo', f'{detector.lower()}_boundaries.npy')

        # Check that the boundary file exists, load it
        if not os.path.isfile(boundary_file):
            raise OSError(f'Could not find boundary file: {boundary_file}')
        self.tpcs = np.load(boundary_file)

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

    def check_containment(self, points, margin=5., mode='module'):
        """
        Check whether a point cloud comes within some distance of the boundaries
        of a certain subset of detector volumes, depending on the mode.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        margin : float, default 5 cm
            Minimum distance from a detector wall to be considered contained
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'global', makes sure is is contained within the outermost walls
            - If 'module', makes sure it is contained within a single module
            - If 'tpc', makes sure it is contained within a single tpc

        Returns
        -------
        bool
            `True` if the particle is contained, `False` if not
        """
        # Establish the volumes to check against
        if mode == 'detector':
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

