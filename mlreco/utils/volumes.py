import numpy as np


class VolumeBoundaries:
    """
    VolumeBoundaries is a helper class to deal with multiple detector volumes. Assume you have N
    volumes that you want to process independently, but your input data file does not separate
    between them (maybe it is hard to make the separation at simulation level, e.g. in Supera).
    You can specify in the configuration of the collate function where the volume boundaries are
    and this helper class will take care of the following:

    1. Relabel batch ids: this will introduce "virtual" batch ids to account for each volume in
    each batch.

    2. Shift coordinates: voxel coordinates are shifted such that the origin is always the bottom
    left corner of a volume. In other words, it ensures the voxel coordinate phase space is the
    same regardless of which volume we are processing. That way you can train on a single volume
    (subpart of the detector, e.g. cryostat or TPC) and process later however many volumes make up
    your detector.

    3. Sort coordinates: there is no guarantee that concatenating coordinates of N volumes vs the
    stored coordinates for label tensors which cover all volumes already by default will yield the
    same ordering. Hence we do a np.lexsort on coordinates after 1. and 2. have happened. We sort
    by: batch id, z, y, x in this order.

    An example of configuration would be :

    ```yaml
    collate:
      collate_fn: Collatesparse
      boundaries: [[1376.3], None, None]
    ```

    `boundaries` is what defines the different volumes. It has a length equal to the spatial dimension.
    For each spatial dimension, `None` means that there is no boundary along that axis.
    A list of floating numbers specifies the volume boundaries along that axis in voxel units.
    The list of volumes will be inferred from this list of boundaries ("meshgrid" style, taking
    all possible combinations of the boundaries to generate all the volumes).
    """
    def __init__(self, definitions):
        """
        See explanation of `boundaries` above.

        Parameters
        ----------
        definitions: list
        """
        self.dim = len(definitions)
        self.boundaries = definitions

        # Quick sanity check
        for i in range(self.dim):
            assert self.boundaries[i] == 'None' or self.boundaries[i] is None or (isinstance(self.boundaries[i], list) and len(self.boundaries[i]) > 0)
            if self.boundaries[i] == 'None':
                self.boundaries[i] = None
                continue
            if self.boundaries[i] is None: continue
            self.boundaries[i].sort() # Ascending order

        n_boundaries = [len(self.boundaries[n]) if self.boundaries[n] is not None else 0 for n in range(self.dim)]
        # Generate indices that describe all volumes
        all_index = []
        for n in range(self.dim):
            all_index.append(np.arange(n_boundaries[n]+1))
        self.combo = np.array(np.meshgrid(*tuple(all_index))).T.reshape(-1, self.dim)

        # Generate coordinate shifts for each volume
        # List of list (1st dim is spatial dimension, 2nd is volume splits in a given spatial dimension)
        shifts = []
        for n in range(self.dim):
            if self.boundaries[n] is None:
                shifts.append([0.])
                continue
            dim_shifts = []
            for i in range(len(self.boundaries[n])):
                dim_shifts.append(self.boundaries[n][i-1] if i > 0 else 0.)
            dim_shifts.append(self.boundaries[n][-1])
            shifts.append(dim_shifts)
        self.shifts = shifts

    def num_volumes(self):
        """
        Returns
        -------
        int
        """
        return len(self.combo)

    def virtual_batch_ids(self, entry=0):
        """
        Parameters
        ----------
        entry: int, optional
            Which entry of the dataset you are trying to access.

        Returns
        -------
        list
            List of virtual batch ids that correspond to this entry.
        """
        return np.arange(len(self.combo)) + entry * self.num_volumes()

    def translate(self, voxels, volume):
        """
        Meant to reverse what the split method does: for voxels coordinates initially in the range of volume 0,
        translate to the range of a specific volume given in argument.

        Parameters
        ----------
        voxels: np.ndarray
            Expected shape is (D_0, ..., D_N, self.dim) with N >=0. In other words, voxels can be a list of
            coordinate or a single coordinate with shape (d,).
        volume: int

        Returns
        -------
        np.ndarray
            Translated voxels array, using internally computed shifts.
        """
        assert volume >= 0 and volume < self.num_volumes()
        assert voxels.shape[-1] == self.dim

        new_voxels = voxels.copy()
        for n in range(self.dim):
            new_voxels[..., n] += int(self.shifts[n][self.combo[volume][n]])
        return new_voxels

    def untranslate(self, voxels, volume):
        """
        Meant to reverse what the translate method does: for voxels coordinates initially in the range of full detector,
        translate to the range of 1 volume for a specific volume given in argument.

        Parameters
        ----------
        voxels: np.ndarray
            Expected shape is (D_0, ..., D_N, self.dim) with N >=0. In other words, voxels can be a list of
            coordinate or a single coordinate with shape (d,).
        volume: int

        Returns
        -------
        np.ndarray
            Translated voxels array, using internally computed shifts.
        """
        assert volume >= 0 and volume < self.num_volumes()
        assert voxels.shape[-1] == self.dim

        new_voxels = voxels.copy()
        for n in range(self.dim):
            new_voxels[..., n] -= int(self.shifts[n][self.combo[volume][n]])
        return new_voxels

    def split(self, voxels):
        """
        Parameters
        ----------
        voxels: np.array, shape (N, 4)
            It should contain (batch id, x, y, z) coordinates in this order (as an example if you are working in 3D).

        Returns
        -------
        new_voxels: np.array, shape (N, 4)
            The array contains voxels with shifted coordinates + virtual batch ids. This array is not yet permuted
            to obey the lexsort.
        perm: np.array, shape (N,)
            This is a permutation mask which can be used to apply the lexsort to both the new voxels and the features
            or data tensor (which is not passed to this function).
        """
        assert len(voxels.shape) == 2
        batch_ids = voxels[:, 0]
        coords = voxels[:, 1:]
        assert self.dim == coords.shape[1]

        # This will contain the list of boolean masks corresponding to each boundary
        # in each spatial dimension (so, list of list)
        all_boundaries = []
        for n in range(self.dim):
            if self.boundaries[n] is None:
                all_boundaries.append([np.ones((coords.shape[0],), dtype=bool)])
                continue
            dim_boundaries = []
            for i in range(len(self.boundaries[n])):
                dim_boundaries.append( coords[:, n] < self.boundaries[n][i] )
            dim_boundaries.append( coords[:, n] >= self.boundaries[n][-1] )
            all_boundaries.append(dim_boundaries)

        virtual_batch_ids = np.zeros((coords.shape[0],), dtype=np.int32)
        new_coords = coords.copy()
        for idx, c in enumerate(self.combo): # Looping over volumes
            m = all_boundaries[0][c[0]] # Building a boolean mask for this volume
            for n in range(1, self.dim):
                m = np.logical_and(m, all_boundaries[n][c[n]])
            # Now defining virtual batch id
            # We need to take into account original batch id
            virtual_batch_ids[m] = idx + batch_ids[m] * self.num_volumes()
            for n in range(self.dim):
                new_coords[m, n] -= int(self.shifts[n][c[n]])

        new_voxels = np.concatenate([virtual_batch_ids[:, None], new_coords], axis=1)
        perm = np.lexsort(new_voxels.T[list(range(1, self.dim+1)) + [0], :])
        return new_voxels, perm

