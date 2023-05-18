from pprint import pprint
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline, CubicSpline
from functools import lru_cache
import pathlib

from analysis.post_processing import post_processing
from mlreco.utils.globals import *

@post_processing(data_capture=['input_data'], 
                 result_capture=['input_rescaled',
                                 'particle_clusts'])
def calorimetric_energy(data_dict,
                        result_dict,
                        conversion_factor=1.):
    """Compute calorimetric energy by summing the charge depositions and
    scaling by the ADC to MeV conversion factor.

    Parameters
    ----------
    data_dict : dict
        Data dictionary (contains one image-worth of data)
    result_dict : dict
        Result dictionary (contains one image-worth of data)
    conversion_factor : float, optional
        ADC to MeV conversion factor (MeV / ADC), by default 1.

    Returns
    -------
    update_dict: dict
        Dictionary to be included into result dictionary, containing the
        computed energy under the key 'particle_calo_energy'.
    """

    input_data     = data_dict['input_data'] if 'input_rescaled' not in result_dict else result_dict['input_rescaled']
    particles      = result_dict['particle_clusts']

    update_dict = {
        'particle_calo_energy': conversion_factor*np.array([np.sum(input_data[p, VALUE_COL]) for p in particles])
    }
            
    return update_dict


@post_processing(data_capture=['meta'], 
                 result_capture=['particles'],
                 result_capture_optional=['truth_particles'])
def range_based_track_energy(data_dict, result_dict,
                             bin_size=17, 
                             include_pids=[2, 3, 4],
                             data=False,
                             min_points=5,
                             mode='px'):
    """Compute track energy by the CSDA (continuous slowing-down approximation)
    range-based method. 

    Parameters
    ----------
    data_dict : dict
        Data dictionary (contains one image-worth of data)
    result_dict : dict
        Result dictionary (contains one image-worth of data)
    bin_size : int, optional
        Bin size used to perform local PCA along the track, by default 17
    include_pids : list, optional
        Particle PDG codes (converted to 0-5 labels) to include in 
        computing the energies, by default [2, 3, 4]
    table_path : str, optional
        Path to muon/proton/pion CSDARange vs. energy table, by default ''

    Returns
    -------
    update_dict: dict
        Dictionary to be included into result dictionary, containing the
        particle's estimated length ('particle_length') and the estimated
        CSDA energy ('particle_range_based_energy') using cubic splines. 
    """
    if data:
        particles = result_dict['particles']
        truth_particles = []
    else:
        particles       = result_dict['particles']
        truth_particles = result_dict['truth_particles']
        
    # Use meta info to convert units
    
    splines = {ptype: get_splines(ptype) for ptype in include_pids}
    meta = data_dict['meta']
    px_to_cm = np.mean(meta[6:9]) # TODO: ONLY TEMPORARY

    for i, p in enumerate(particles):
        if p.semantic_type == 1 and p.pid in include_pids:
            if mode == 'cm':
                points = p.points
                bin_size_cm = bin_size
            else:
                points = _pix_to_cm(p.points, meta)
                bin_size_cm = bin_size * px_to_cm
            if points.shape[0] > min_points:
                length = compute_track_length(points, bin_size=bin_size_cm)
                p.length = length
                p.csda_kinetic_energy = splines[p.pid](length)

    for i, p in enumerate(truth_particles):
        if p.semantic_type == 1 and p.pid in include_pids:
            if mode == 'cm':
                pts = p.points
                tng_pts = p.truth_points
                bin_size_cm = bin_size
            else:
                pts = _pix_to_cm(p.points, meta)
                tng_pts = _pix_to_cm(p.truth_points, meta)
                bin_size_cm = bin_size * px_to_cm
            if pts.shape[0] > min_points:
                length = compute_track_length(pts, bin_size=bin_size_cm)
                p.length = length
                p.csda_kinetic_energy = splines[p.pid](length)
            if tng_pts.shape[0] > min_points:
                length_tng = compute_track_length(tng_pts, bin_size=bin_size_cm)
                p.length_tng = length_tng
                p.csda_kinetic_energy_tng = splines[p.pid](length_tng)
            
    return {}


@post_processing(data_capture=['meta'], 
                 result_capture=['particles'],
                 result_capture_optional=['truth_particles'])
def range_based_track_energy_spline(data_dict,
                                    result_dict,
                                    bin_size=17,
                                    include_pids=[2,3,4],
                                    data=False,
                                    min_points=10,
                                    mode='px'):
    
    if data:
        particles = result_dict['particles']
        truth_particles = []
    else:
        particles = result_dict['particles']
        truth_particles = result_dict['truth_particles']
    
    splines = {ptype: get_splines(ptype) for ptype in include_pids}
    meta = data_dict['meta']
    px_to_cm = np.mean(meta[6:9]) # TODO: ONLY TEMPORARY
    
    for i, p in enumerate(particles):
        if p.semantic_type == 1 and p.pid in include_pids:
            if mode == 'cm':
                pts = p.points
                bin_size_cm = bin_size
            else:
                pts = _pix_to_cm(p.points, meta)
                bin_size_cm = bin_size * px_to_cm
            if pts.shape[0] > min_points:
                curve_data = compute_curve(pts, bin_size=bin_size_cm)
                length = curve_data[3]
                p.length = length
                p.csda_kinetic_energy = splines[p.pid](length)
            
    for i, p in enumerate(truth_particles):
        if p.semantic_type == 1 and p.pid in include_pids:
            if mode == 'cm':
                pts = p.points
                tng_pts = p.truth_points
                bin_size_cm = bin_size
            else:
                pts = _pix_to_cm(p.points, meta)
                tng_pts = _pix_to_cm(p.truth_points, meta)
                bin_size_cm = bin_size * px_to_cm
            if pts.shape[0] > min_points:
                curve_data = compute_curve(pts, bin_size=bin_size)
                length = curve_data[3]
                p.length = length
                p.csda_kinetic_energy = splines[p.pid](length)
            if tng_pts.shape[0] > min_points:
                curve_truth = compute_curve(tng_pts, bin_size=bin_size)
                length_tng = curve_truth[3]
                p.length_tng = length_tng
                p.csda_kinetic_energy_tng = splines[p.pid](length_tng)
            
    return {}
    

# ----------------------------- Helper functions -----------------------------

@lru_cache
def get_splines(particle_type):
    """_summary_

    Parameters
    ----------
    particle_type : int
        Particle type ID to construct splines. 
        Only one of [2,3,4] are available. 
    table_path : str
        Path to CSDARange vs Kinetic E table. 

    Returns
    -------
    f: Callable
        Function mapping CSDARange (g/cm^2) vs. Kinetic E (MeV/c^2)
    """
    path = pathlib.Path(__file__).parent
    if particle_type == PDG_TO_PID[2212]:
        path = os.path.join(path, 'tables', 'pE_liquid_argon.txt')
        tab = pd.read_csv(path, 
                          delimiter=' ',
                          index_col=False)
    elif particle_type == PDG_TO_PID[13]:
        path = os.path.join(path, 'tables', 'muE_liquid_argon.txt')
        tab = pd.read_csv(path, 
                          delimiter=' ',
                          index_col=False)
    else:
        raise ValueError("Range based energy reconstruction for particle type"\
                        " {} is not supported!".format(particle_type))
    # print(tab)
    f = CubicSpline(tab['CSDARange'] / ARGON_DENSITY, tab['T'])
    return f


def compute_curve(points, s=None, bin_size=20):
    """Estimate the best approximating curve defined by a point cloud 
    using univariate 3D splines. 
    
    The length is computed by measuring the length of the piecewise linear
    interpolation of the spline at points defined by the bin size. 

    Parameters
    ----------
    points : np.ndarray
        (N x 3) point cloud
    s : float, optional
        The smoothing factor to be used in spline regression, by default None
    bin_size : int, optional
        The subdivision length at which to sample points from the spline.
        If the track length is less than the bin_size, then the returned
        length will be computed from the farthest two projected points along
        the track's principal direction.

    Returns
    -------
    u : np.ndarray
        The principal axis parametrization (N, ) of the curve 
        C(u) = (spx(u), spy(u), spz(u))
    sppoints: np.ndarray
        The graph (N, 3) of the spline at points u.
    splines: scipy.interpolate.UnivariateSpline
        Approximating splines for the point cloud defined by points. 
    length: float
        The estimate of the total length of the curve. 
    """
    
    pca = PCA(n_components=1)
    proj_1d = pca.fit_transform(points)
    perm = np.argsort(proj_1d.squeeze())
    u = proj_1d[perm]
    spx = UnivariateSpline(u, points[perm][:, 0], s=s)
    spy = UnivariateSpline(u, points[perm][:, 1], s=s)
    spz = UnivariateSpline(u, points[perm][:, 2], s=s)
    sppoints = np.hstack([spx(u), spy(u), spz(u)])
    
    splines = [spx, spy, spz]
    
    start, end = u.min(), u.max()
    length = end - start
    
    # If track length is less than bin_size, just return length.
    # Otherwise estimate length by piecewise linear interpolation. 
    if length > bin_size:
        bins = np.arange(u.min(), u.max(), bin_size)
        bins = np.hstack([bins, np.array([u.max()])])
        pt_approx = np.hstack([sp(bins).reshape(-1, 1) for sp in splines])
        segments = np.linalg.norm(pt_approx[1:] - pt_approx[:-1], axis=1)
        length = segments.sum()

    return u.squeeze(), sppoints, splines, length


def compute_track_length(points, bin_size=17):
    """Compute track length by dividing it into segments and computing 
    a local PCA axis, then summing the local lengths of the segments.

    Parameters
    ----------
    points: np.ndarray
        Shape (N, 3)
    bin_size: int, optional
        Size (in voxels) of the segments

    Returns
    -------
    float
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
            # pca_axis = pca.fit_transform(points[mask])
            pca_axis = coords_pca[mask]
            dx = pca_axis.max() - pca_axis.min()
            length += dx
    return length
    

def compute_track_dedx(points, startpoint, endpoint, depositions, bin_size=17):
    assert len(points) >= 2
    vec = endpoint - startpoint
    vec_norm = np.linalg.norm(vec)
    vec = (vec / (vec_norm + 1e-6)).astype(np.float64)
    proj = points - startpoint
    proj = np.dot(proj, vec)
    bins = np.arange(proj.min(), proj.max(), bin_size)
    bin_inds = np.digitize(proj, bins)
    dedx = np.zeros(np.unique(bin_inds).shape[0]).astype(np.float64)
    for i, b_i in enumerate(np.unique(bin_inds)):
        mask = bin_inds == b_i
        sum_energy = depositions[mask].sum()
        if np.count_nonzero(mask) < 2: continue
        # Repeat PCA locally for better measurement of dx
        dx = proj[mask].max() - proj[mask].min()
        dedx[i] = sum_energy / dx
    return dedx

def _pix_to_cm(arr, meta):
    
    min_x        = meta[0]
    min_y        = meta[1]
    min_z        = meta[2]
    size_voxel_x = meta[6]
    size_voxel_y = meta[7]
    size_voxel_z = meta[8]
    
    arr[:, COORD_COLS[0]] = arr[:, COORD_COLS[0]] * size_voxel_x + min_x
    arr[:, COORD_COLS[1]] = arr[:, COORD_COLS[1]] * size_voxel_y + min_y
    arr[:, COORD_COLS[2]] = arr[:, COORD_COLS[2]] * size_voxel_z + min_z
    return arr