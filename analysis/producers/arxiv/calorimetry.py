import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from mlreco.utils.gnn.cluster import cluster_direction
import pandas as pd
from analysis.classes import Particle
from mlreco.utils.globals import *


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
            pca_axis = pca.fit_transform(points[mask])
            dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
            length += dx
    return length


def get_csda_range_spline(particle_type, table_path):
    '''
    Returns CSDARange (g/cm^2) vs. Kinetic E (MeV/c^2)
    '''
    if particle_type == 'proton':
        tab = pd.read_csv(table_path, 
                          delimiter=' ',
                          index_col=False)
    elif particle_type == 'muon':
        tab = pd.read_csv(table_path, 
                          delimiter=' ',
                          index_col=False)
    else:
        raise ValueError("Range based energy reconstruction for particle type\
            {} is not supported!".format(particle_type))
    # print(tab)
    f = CubicSpline(tab['CSDARange'] / ARGON_DENSITY, tab['T'])
    return f


def compute_range_based_energy(particle, f, **kwargs):
    assert particle.semantic_type == 1
    if particle.pid == 4: m = PROTON_MASS
    elif particle.pid == 2: m = MUON_MASS
    else:
        raise ValueError("For track particle {}, got {}\
             as particle type!".format(particle.pid))
    if not hasattr(particle, 'length'):
        particle.length = compute_track_length(particle.points, **kwargs)
    kinetic = f(particle.length * PIXELS_TO_CM)
    total = kinetic + m
    return total


def get_particle_direction(p: Particle, **kwargs):
    v = cluster_direction(p.points, p.startpoint, **kwargs)
    return v
    

def compute_track_dedx(p, bin_size=17):
    assert len(p.points) >= 2
    vec = p.endpoint - p.startpoint
    vec_norm = np.linalg.norm(vec)
    vec = (vec / (vec_norm + 1e-6)).astype(np.float64)
    proj = p.points - p.startpoint
    proj = np.dot(proj, vec)
    bins = np.arange(proj.min(), proj.max(), bin_size)
    bin_inds = np.digitize(proj, bins)
    dedx = np.zeros(np.unique(bin_inds).shape[0]).astype(np.float64)
    for i, b_i in enumerate(np.unique(bin_inds)):
        mask = bin_inds == b_i
        sum_energy = p.depositions[mask].sum()
        if np.count_nonzero(mask) < 2: continue
        # Repeat PCA locally for better measurement of dx
        dx = proj[mask].max() - proj[mask].min()
        dedx[i] = sum_energy / dx
    return dedx


def highland_formula(p, l, m, X_0=14, z=1):
    '''
    Highland formula for angular scattering variance.
    
    Parameters:
        - p: Initial momentum (MeV/c)
        - l: Distance traveled inside material (cm)
        - X_0: Radiation Length of target material (cm)
        - m: mass of particle
        - z: charge of particle
        
    Returns:
        - sigma (unitless)
    '''
    sigma = (13.6 * (np.sqrt(p**2 + m**2) / p**2)) * z * np.sqrt(l / X_0) * (1 + 0.038 * np.log(l / X_0))
    return sigma


def modified_highland_formula(p, l, m, X_0=14, z=1, sigma_res=0.0):
    k = 0.105 / p**2 + 11.004
    beta = 1 / np.sqrt(1 + (m / p)**2)
    rms = np.sqrt((k / (p * beta))**2 + sigma_res**2)
    return rms


def gauss(dtheta, sigma):
    return (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * (dtheta / sigma)**2)


def compute_log_likelihood(dthetas, dls, initial_energy, l=14, m=105.658):
    
    assert len(dthetas) > 2
    
    ll = 0.5 * len(dls) * np.log(2 * np.pi)
    
    E = initial_energy
    assert E > 0
    MUON_IONIZATION_CONST = 2.1
    for j in range(len(dthetas)):
        sigma_j = modified_highland_formula(E, l, m)
        eloss = dls[j] * MUON_IONIZATION_CONST
        E = E - eloss
        
        ll += np.log(sigma_j) 
        ll += 0.5 * (dthetas[j] / sigma_j)**2
        
        if E < 1e-6:
            return None
    
    return ll


def bethe_bloch_dEdx(E, m, K=0.30705, A=39.948, Z=18, m_e=511, rho=1.3982, I=0.188):
    C = K * rho * Z / A 
    beta = (E / m) * 1.0 / np.sqrt(1 + (E / m)**2 )
    gamma = 1.0 / np.sqrt(1 - beta**2)
    dEdx = (C / beta**2) * (np.log(2 * m_e * (beta * gamma)**2 / I) - beta**2 )
    return dEdx


def compute_mcs_muon_energy(particle, bin_size=17, 
                            min_E=1, max_E=1000, step_E=1):

    pca = PCA(n_components=3)
    coords_pca = pca.fit_transform(particle.points)
    proj = coords_pca[:, 0]
    global_dir = get_particle_direction(particle, optimize=True)
    if global_dir[0] < 0:
        global_dir = pca.components_[0]
    perm = np.argsort(proj.squeeze())
    bin_size = 20
    bins = np.arange(proj.min(), proj.max(), bin_size)

    binds = np.digitize(proj, bins)
    cones = np.zeros((6, np.unique(binds).shape[0]))
    for i, b_i in enumerate(np.unique(binds)):
        mask = binds == b_i
        points_segment = particle.points[mask]
        if points_segment.shape[0] < 3:
            return -1, -1
        segment_pca = pca.fit_transform(points_segment)
        segment_dir = pca.components_[0]
        if np.dot(segment_dir, global_dir) < 0:
            segment_dir *= -1
        cones[0:3, i-1] = points_segment[0]
        cones[3:6, i-1] = segment_dir

    vecs = cones[3:6, :].T
    fragments = cones[0:3, :].T
    angles = []
    lengths = []
    for i in range(vecs.shape[0]-1):
        length = np.linalg.norm(fragments[i+1] - fragments[i]) * 0.3
        angle = np.arccos(np.dot(vecs[i], vecs[i+1]))
        angles.append(angle)
        lengths.append(length)
    angles = np.array(angles)
    lengths = np.array(lengths)

    if len(lengths) <= 2:
        return -1, -1

    lls, Es = [], []
    einit, min_ll = 1, np.inf
    for i in range(min_E, max_E, step_E):
        ll = compute_log_likelihood(angles, lengths, i)
        if ll is None:
            continue
        else:
            lls.append(ll)
            Es.append(i)
            if ll < min_ll:
                min_ll = ll
                einit = i
    lls = np.array(lls)
    Es = np.array(Es)
    return einit, min_ll