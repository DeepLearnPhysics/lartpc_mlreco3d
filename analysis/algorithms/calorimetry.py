from analysis.classes.particle import Particle
import numpy as np
import numba as nb
from sklearn.decomposition import PCA


def compute_sum_deposited(particle : Particle):
    assert hasattr(particle, 'deposition')
    sum_E = particle.deposition.sum()
    return sum_E


def compute_track_length(points, bin_size=17):
    """
    Compute track length by dividing it into segments
    and computing a local PCA axis, then summing the
    local lengths of the segments.

    Parameters
    ----------
    points: np.ndarray
        Shape (N, 3)
    bin_size: int, optional
        Size (in voxels) of the segments
    """
    pca = PCA(n_components=2)
    length = 0.
    if len(points) >= 2:
        coords_pca = pca.fit_transform(points)[:, 0]
        bins = np.arange(coords_pca.min(), coords_pca.max(), bin_size)
        # bin_inds takes values in [1, len(bins)]
        bin_inds = np.digitize(coords_pca, bins)
        for b_i in np.unique(bin_inds):
            mask = bin_inds == b_i
            if np.count_nonzero(mask) < 2: continue
            # Repeat PCA locally for better measurement of dx
            pca_axis = pca.fit_transform(points[mask])
            dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
            length += dx
    return length
# TODO:
# def proton_energy_tabular(particle: Particle):
#     assert particle.pid == 4  # Proton
#     x, y = particle.endpoints[0], particle.endpoints[1]
#     l = np.sqrt(np.power(x - y, 2).sum())

# def multiple_coulomb_scattering(particle: Particle):
#     assert particle.pid == 2  # Muon
#     pass
