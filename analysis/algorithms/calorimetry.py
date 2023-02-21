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


def compute_particle_direction(p: Particle, 
                               start_segment_radius=17, 
                               vertex=None,
                               return_explained_variance=False):
    """
    Given a Particle, compute the start direction. Within `start_segment_radius`
    of the start point, find PCA axis and measure direction.

    If not start point is found, returns (-1, -1, -1).

    Parameters
    ----------
    p: Particle
    start_segment_radius: float, optional

    Returns
    -------
    np.ndarray
        Shape (3,)
    """
    pca = PCA(n_components=2)
    direction = None
    if p.startpoint is not None and p.startpoint[0] >= 0.:
        startpoint = p.startpoint
        if p.endpoint is not None and vertex is not None: # make sure we pick the one closest to vertex
            use_end = np.argmin([
                np.sqrt(((vertex-p.startpoint)**2).sum()),
                np.sqrt(((vertex-p.endpoint)**2).sum())
            ])
            startpoint = p.endpoint if use_end else p.startpoint
        d = np.sqrt(((p.points - startpoint)**2).sum(axis=1))
        if (d < start_segment_radius).sum() >= 2:
            direction = pca.fit(p.points[d < start_segment_radius]).components_[0, :]
    if direction is None: # we could not find a startpoint
        if len(p.points) >= 2: # just all voxels
            direction = pca.fit(p.points).components_[0, :]
        else:
            direction = np.array([-1, -1, -1])
            if not return_explained_variance:
                return direction
            else:
                return direction, np.array([-1, -1])
    if not return_explained_variance:
        return direction
    else:
        return direction, pca.explained_variance_ratio_


def load_range_reco(particle_type='muon', kinetic_energy=True):
    """
    Return a function maps the residual range of a track to the kinetic
    energy of the track. The mapping is based on the Bethe-Bloch formula
    and stored per particle type in TGraph objects. The TGraph::Eval
    function is used to perform the interpolation.

    Parameters
    ----------
    particle_type: A string with the particle name.
    kinetic_energy: If true (false), return the kinetic energy (momentum)
    
    Returns
    -------
    The kinetic energy or momentum according to Bethe-Bloch.
    """
    output_var = ('_RRtoT' if kinetic_energy else '_RRtodEdx')
    if particle_type in ['muon', 'pion', 'kaon', 'proton']:
        input_file = ROOT.TFile.Open('RRInput.root', 'read')
        graph = input_file.Get(f'{particle_type}{output_var}')
        return np.vectorize(graph.Eval)
    else:
        print(f'Range-based reconstruction for particle "{particle_type}" not available.')


def make_range_based_momentum_fns():
    f_muon = load_range_reco('muon')
    f_pion = load_range_reco('pion')
    f_proton = load_range_reco('proton')
    return [f_muon, f_pion, f_proton]


def compute_range_momentum(particle, f, voxel_to_cm=0.3, **kwargs):
    assert particle.semantic_type == 1
    length = compute_track_length(particle.points, 
                                  bin_size=kwargs.get('bin_size', 17))
    E = f(length * voxel_to_cm)
    return E


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
    global_dir = compute_particle_direction(particle, 
                                            start_segment_radius=bin_size)
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