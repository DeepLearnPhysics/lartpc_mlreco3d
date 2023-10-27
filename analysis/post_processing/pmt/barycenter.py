import numpy as np

from mlreco.utils.geometry import Geometry
from mlreco.utils.numba_local import cdist

class BarycenterFlashMatcher:
    '''
    Matches interactions and flashes by matching the charge barycenter of
    TPC interactions with the light barycenter of optical flashes.
    '''
    axes = ['x', 'y', 'z']

    def __init__(self, match_method='threshold', dimensions=[1,2],
            charge_weighted=False, time_window=None, first_flash_only=False,
            min_inter_size=None, min_flash_pe=None, match_distance=None,
            detector=None, boundary_file=None, source_file=None):
        '''
        Initalize the barycenter flash matcher.

        Parameters
        ----------
        match_method: str, default 'distance'
            Matching method (one of 'threshold' or 'best')
            - 'threshold': If the two barycenters are within some distance, match
            - 'best': For each flash, pick the best matched interaction
        dimensions: list, default [0,1,2]
            Dimensions involved in the distance computation
        charge_weighted : bool, default False
            Use interaction pixel charge information to weight the centroid
        time_window : List, optional
            List of [min, max] values of optical flash times to consider
        first_flash_only : bool, default False
            Only try to match the first flash in the time window
        min_inter_size : int, optional
            Minimum number of voxel in an interaction to consider it
        min_flash_pe : float, optional
            Minimum number of total PE in a flash to consider it
        match_distance : float, optional
            If a threshold is used, specifies the acceptable distance
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        '''
        self.geo = None
        if detector is not None or boundary_file is not None:
            self.geo = Geometry(detector, boundary_file, source_file)

        # Store the flash matching parameters
        self.match_method     = match_method
        self.dims             = dimensions
        self.charge_weighted  = charge_weighted
        self.time_window      = time_window
        self.first_flash_only = first_flash_only
        self.min_inter_size   = min_inter_size
        self.min_flash_pe     = min_flash_pe
        self.match_distance   = match_distance

        # Check validity of certain parameters
        if self.match_method not in ['threshold', 'best']:
            msg = f'Barycenter flash matching method not recognized: {match_method}'
            raise ValueError(msg)
        if self.match_method == 'threshold':
            assert self.match_distance is not None, \
                    'When using the `threshold` method, must specify `match_distance`'

    def get_matches(self, interactions, flashes):
        '''
        Makes [interaction, flash] pairs that have compatible barycenters.

        Parameters
        ----------
        interactions : List[Interaction]
            List of interactions
        flashes : List[larcv.Flash]
            List of optical flashes

        Returns
        -------
        List[[Interaction, larcv.Flash, float]]
            List of [interaction, flash, distance] triplets
        '''
        # Restrict the flashes to those that fit the selection criteria
        flashes = np.asarray(flashes, dtype=object)
        if self.time_window is not None:
            t1, t2 = self.time_window
            mask = [(f.time() > t1 and f.time() < t2)  for f in flashes]
            flashes = flashes[mask]
        if self.min_flash_pe is not None:
            mask = [f.flash_total_pE > self.min_flash_pe for f in flashes]
            flashes = flashes[mask]
        if not len(flashes):
            return []
        if self.first_flash_only:
            flashes = [flashes[0]]

        # Restrict the interactions to those that fit the selection criterion
        if self.min_inter_size is not None:
            interactions = [ia for ia in interactions \
                    if ia.size > self.min_inter_size]
        if not len(interactions):
            return []

        # Get the flash centroids
        op_centroids = np.empty((len(flashes), len(self.dims)))
        op_widths = np.empty((len(flashes), len(self.dims)))
        for i, f in enumerate(flashes):
            for j, d in enumerate(self.dims):
                op_centroids[i, j] = getattr(f, f'{self.axes[d]}Center')()
                op_widths[i, j] = getattr(f, f'{self.axes[d]}Width')()

        # Get interactions centroids
        int_centroids = np.empty((len(interactions), len(self.dims)))
        for i, ia in enumerate(interactions):
            if not self.charge_weighted:
                int_centroids[i] = np.mean(ia.points[:, self.dims], axis=0)
            else:
                int_centroids[i] = np.sum(ia.depositions * \
                        ia.points[:, self.dims], axis=0) \
                            / np.sum(ia.depositions)

        # Compute the flash to interaction distance matrix
        dist_mat = cdist(op_centroids, int_centroids)

        # Produce matches
        matches = []
        if self.match_method == 'best':
            # For each flash, select the best match, save attributes
            for i, f in enumerate(flashes):
                best_match = np.argmin(dist_mat[i])
                dist = dist_mat[i, best_match]
                if self.match_distance is not None \
                        and dist > self.match_distance:
                            continue
                match.append((interactions[best_match], f, dist))
        elif self.match_method == 'threshold':
            # Find all compatible pairs
            valid_pairs = np.vstack(np.where(dist_mat <= self.match_distance)).T
            matches = [(interactions[j], flashes[i], dist_mat[i,j]) \
                    for i, j in valid_pairs]

        return matches
