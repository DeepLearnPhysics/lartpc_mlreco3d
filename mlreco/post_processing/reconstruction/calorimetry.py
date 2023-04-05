from pprint import pprint
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from functools import lru_cache

from mlreco.utils.gnn.cluster import cluster_direction
from mlreco.post_processing import post_processing
from mlreco.utils.globals import *

@post_processing(data_capture=[], result_capture=['particle_clusts', 
                                                  'particle_seg', 
                                                  'input_rescaled', 
                                                  'particle_node_pred_type'])
def range_based_track_energy(data_dict, result_dict,
                             bin_size=17, include_pids=[2, 3, 4], table_path=''):

    input_data     = result_dict['input_rescaled']
    particles      = result_dict['particle_clusts']
    particle_seg   = result_dict['particle_seg']
    particle_types = result_dict['particle_node_pred_type']

    update_dict = {
        'particle_length': np.array([]),
        'particle_range_based_energy': np.array([])
    }
    if len(particles) == 0:
        return update_dict

    splines = {ptype: get_splines(ptype, table_path) for ptype in include_pids}

    pred_ptypes = np.argmax(particle_types, axis=1)
    particle_length = -np.ones(len(particles))
    particle_energy = -np.ones(len(particles))

    assert len(pred_ptypes) == len(particle_types)

    for i, p in enumerate(particles):
        semantic_type = particle_seg[i]
        if semantic_type == 1 and pred_ptypes[i] in include_pids:
            points = input_data[p]
            length = compute_track_length(points, bin_size=bin_size)
            particle_length[i] = length
            particle_energy[i] = splines[pred_ptypes[i]](length * PIXELS_TO_CM)
            
    update_dict['particle_length'] = particle_length
    update_dict['particle_range_based_energy'] = particle_energy
            
    return update_dict


# Helper Functions
@lru_cache
def get_splines(particle_type, table_path):
    '''
    Returns CSDARange (g/cm^2) vs. Kinetic E (MeV/c^2)
    '''
    if particle_type == PDG_TO_PID[2212]:
        path = os.path.join(table_path, 'pE_liquid_argon.txt')
        tab = pd.read_csv(path, 
                          delimiter=' ',
                          index_col=False)
    elif particle_type == PDG_TO_PID[13]:
        path = os.path.join(table_path, 'muE_liquid_argon.txt')
        tab = pd.read_csv(path, 
                          delimiter=' ',
                          index_col=False)
    else:
        raise ValueError("Range based energy reconstruction for particle type"\
                        " {} is not supported!".format(particle_type))
    # print(tab)
    f = CubicSpline(tab['CSDARange'] / ARGON_DENSITY, tab['T'])
    return f


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
            # pca_axis = pca.fit_transform(points[mask])
            pca_axis = coords_pca[mask]
            dx = pca_axis.max() - pca_axis.min()
            length += dx
    return length

