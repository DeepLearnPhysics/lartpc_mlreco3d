import numpy as np
import numba as nb
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from mlreco.utils.globals import COORD_COLS

from analysis.post_processing import post_processing
from analysis.post_processing.reconstruction.calorimetry import compute_track_dedx

@post_processing(data_capture=[], 
                 result_capture=['particle_start_points',
                                 'particle_end_points',
                                #  'input_rescaled',
                                #  'particle_seg',
                                #  'particle_clusts',
                                 'particles'])
def assign_particle_extrema(data_dict, result_dict,
                            mode='local_density',
                            radius=1.5):
    """Post processing for assigning track startpoint and endpoint, with
    added correction modules.
    
    Parameters
    ----------
    mode: algorithm to correct track startpoint/endpoint misplacement.
        The following modes are available:
        - linfit: computes local energy deposition density throughout the
        track, computes the overall slope (linear fit) of the energy density
        variation to estimate the direction.
        - local_desnity: computes local energy deposition density only at
        the extrema and chooses the higher one as the endpoint.
        - ppn: uses ppn candidate predictions (classify_endpoints) to assign
        start and endpoints.

    Returns
    -------
        update_dict: dict
            Empty dictionary (operation is in-place)
    """

    startpts       = result_dict['particle_start_points'][:, COORD_COLS]
    endpts         = result_dict['particle_end_points'][:, COORD_COLS]
    particles      = result_dict['particles']

    assert len(startpts) == len(endpts)
    assert len(startpts) == len(particles)
    
    for i, p in enumerate(particles):
        if p.semantic_type == 1:
            start_point = p.start_point
            end_point   = p.end_point
            new_start_point, new_end_point = get_track_points(p.points,
                                                              start_point,
                                                              end_point, 
                                                              p.depositions,
                                                              correction_mode=mode,
                                                              r=radius)
            p.start_point = new_start_point
            p.end_point   = new_end_point
    
    return {}



def handle_singleton_ppn_candidate(pts, ppn_candidates):
    """Function for handling ppn endpoint correction cases in which
    there's only one ppn candidate associated with a particle instance.

    Parameters
    ----------
    pts: (2 x 3 np.array)
        xyz coordinates of startpoint and endpoint
    ppn_candidates: (N x 5 np.array)
        ppn predictions associated with a single particle instance.

    Returns
    -------
    new_points: (2 x 3 np.array)
        Rearranged startpoint and endpoint based on proximity to
        ppn candidate point and endpoint score.
    
    """
    assert ppn_candidates.shape[0] == 1
    score = ppn_candidates[0][5:]
    label = np.argmax(score)
    dist = cdist(pts, ppn_candidates[:, :3])
    pt_near = pts[dist.argmin(axis=0)]
    pt_far = pts[dist.argmax(axis=0)]
    if label == 0:
        startpoint = pt_near.reshape(-1)
        endpoint = pt_far.reshape(-1)
    else:
        endpoint = pt_near.reshape(-1)
        startpoint = pt_far.reshape(-1)

    new_points = np.vstack([startpoint, endpoint])

    return new_points


def correct_track_endpoints_ppn(startpoint: np.ndarray, 
                                endpoint: np.ndarray,
                                ppn_candidates: np.ndarray):


    pts = np.vstack([startpoint, endpoint])

    new_points = np.copy(pts)
    if ppn_candidates.shape[0] == 0:
        startpoint = pts[0]
        endpoint = pts[1]
    elif ppn_candidates.shape[0] == 1:
        # If only one ppn candidate, find track endpoint closer to
        # ppn candidate and give the candidate's label to that track point
        new_points = handle_singleton_ppn_candidate(pts, ppn_candidates)
    else:
        dist1 = cdist(np.atleast_2d(ppn_candidates[:, :3]), 
                      np.atleast_2d(pts[0])).reshape(-1)
        dist2 = cdist(np.atleast_2d(ppn_candidates[:, :3]), 
                      np.atleast_2d(pts[1])).reshape(-1)
        
        ind1, ind2 = dist1.argmin(), dist2.argmin()
        if ind1 == ind2:
            ppn_candidates = ppn_candidates[dist1.argmin()].reshape(1, 7)
            new_points = handle_singleton_ppn_candidate(pts, ppn_candidates)
        else:
            pt1_score = ppn_candidates[ind1][5:]
            pt2_score = ppn_candidates[ind2][5:]
            
            labels = np.array([pt1_score.argmax(), pt2_score.argmax()])
            scores = np.array([pt1_score.max(), pt2_score.max()])
            
            if labels[0] == 0 and labels[1] == 1:
                new_points[0] = pts[0]
                new_points[1] = pts[1]
            elif labels[0] == 1 and labels[1] == 0:
                new_points[0] = pts[1]
                new_points[1] = pts[0]
            elif labels[0] == 0 and labels[1] == 0:
                # print("Particle {} has no endpoint".format(p.id))
                # Select point with larger score as startpoint
                ix = np.argmax(scores)
                iy = np.argmin(scores)
                # print(ix, iy, pts, scores)
                new_points[0] = pts[ix]
                new_points[1] = pts[iy]
            elif labels[0] == 1 and labels[1] == 1:
                ix = np.argmax(scores) # point with higher endpoint score
                iy = np.argmin(scores)
                new_points[0] = pts[iy]
                new_points[1] = pts[ix]
            else:
                raise ValueError("Classify endpoints feature dimension must be 2, got something else!")
            
    return new_points[0], new_points[1]


def correct_track_endpoints_local_density(points: np.ndarray, 
                                          startpoint: np.ndarray, 
                                          endpoint: np.ndarray, 
                                          depositions: np.ndarray,
                                          r=5):
    new_startpoint, new_endpoint = np.copy(startpoint), np.copy(endpoint)
    pca = PCA(n_components=2)
    mask_st = np.linalg.norm(startpoint - points, axis=1) < r
    if np.count_nonzero(mask_st) < 2:
        return new_startpoint, new_endpoint
    pca_axis = pca.fit_transform(points[mask_st])
    length = pca_axis[:, 0].max() - pca_axis[:, 0].min()
    local_d_start = depositions[mask_st].sum() / length
    mask_end = np.linalg.norm(endpoint - points, axis=1) < r
    if np.count_nonzero(mask_end) < 2:
        return new_startpoint, new_endpoint
    pca_axis = pca.fit_transform(points[mask_end])
    length = pca_axis[:, 0].max() - pca_axis[:, 0].min()
    local_d_end = depositions[mask_end].sum() / length
    # Startpoint must have lowest local density
    if local_d_start > local_d_end:
        p1, p2 = startpoint, endpoint
        new_startpoint = p2
        new_endpoint = p1
    return new_startpoint, new_endpoint


def correct_track_endpoints_linfit(points, 
                                   startpoint,
                                   endpoint,
                                   depositions,
                                   bin_size=17):
    if len(points) >= 2:
        dedx = compute_track_dedx(points, 
                                  startpoint,
                                  endpoint, 
                                  depositions, 
                                  bin_size=bin_size)
        new_startpoint, new_endpoint = np.copy(startpoint), np.copy(endpoint)
        if len(dedx) > 1:
            x = np.arange(len(dedx))
            params = np.polyfit(x, dedx, 1)
            if params[0] < 0:
                p1, p2 = startpoint, endpoint
                new_startpoint = p2
                new_endpoint = p1
        return new_startpoint, new_endpoint


def get_track_endpoints_max_dist(points):
    """Helper function for getting track endpoints.

    Computes track endpoints without ppn predictions by
    selecting the farthest two points from the coordinate centroid.

    Parameters
    ----------
    points: (N x 3) particle voxel coordinates

    Returns
    -------
    endpoints : (2, 3) np.array
        Xyz coordinates of two endpoints predicted or manually found
        by network.
    """
    coords = points
    dist = cdist(coords, coords)
    inds = np.unravel_index(dist.argmax(), dist.shape)
    return coords[inds[0]], coords[inds[1]]


def get_track_points(points,
                     startpoint,
                     endpoint,
                     depositions,
                     correction_mode='ppn',
                     **kwargs):
    if correction_mode == 'ppn':
        ppn_candidates = kwargs['ppn_candidates']
        new_startpoint, new_endpoint = correct_track_endpoints_ppn(startpoint, 
                                                                   endpoint, 
                                                                   ppn_candidates)
    elif correction_mode == 'local_density':
        new_startpoint, new_endpoint = correct_track_endpoints_local_density(points, 
                                                                             startpoint, 
                                                                             endpoint, 
                                                                             depositions, 
                                                                             **kwargs)
    elif correction_mode == 'linfit':
        new_startpoint, new_endpoint = correct_track_endpoints_linfit(points,
                                                                      startpoint,
                                                                      endpoint,
                                                                      depositions,
                                                                      **kwargs)
    else:
        raise ValueError("Track extrema correction mode {} not defined!".format(correction_mode))
    return new_startpoint, new_endpoint