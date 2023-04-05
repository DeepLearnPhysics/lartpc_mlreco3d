import numpy as np
import numba as nb
from scipy.spatial.distance import cdist

from mlreco.utils.gnn.cluster import cluster_direction
from mlreco.post_processing import post_processing
from mlreco.utils.globals import COORD_COLS


@post_processing(data_capture=[],
                 result_capture=['particle_clusts',
                                 'input_rescaled',
                                 'particle_start_points'])
def reconstruct_direction(data_dict, result_dict,
                          max_dist=-1, optimize=True):
    """Post processing for reconstructing particle direction.
    
    """
    startpts  = result_dict['particle_start_points'][:, COORD_COLS[0]:COORD_COLS[-1]+1]
    coords    = result_dict['input_rescaled'][:, COORD_COLS[0]:COORD_COLS[-1]+1]
    particles = result_dict['particle_clusts']

    update_dict = {}

    particle_dirs = []
    for i, mask in enumerate(particles):
        pts = coords[mask]
        vec = cluster_direction(pts, startpts[i], 
                                max_dist=max_dist, optimize=optimize)
        particle_dirs.append(vec)
    if len(particle_dirs) > 0:
        particle_dirs = np.vstack(particle_dirs)
        update_dict['particle_dirs'] = particle_dirs
    else:
        update_dict['particle_dirs'] = np.array(particle_dirs)
    return update_dict