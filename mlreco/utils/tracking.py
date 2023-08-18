import numpy as np
import numba as nb

from . import numba_local as nbl


def get_track_length(coordinates: nb.float32[:,:],
                     segment_length: nb.float32 = None,
                     point: nb.float32[:] = None,
                     method: str = 'step',
                     anchor_point: bool = True,
                     min_count: int = 1) -> nb.float32:
    '''
    Given a set of point coordinates associated with a track and one of its end
    points, compute its length.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    point : np.ndarray, optional
        (3) An end point of the track
    segment_length : float, optional
        Segment length in the units that specify the coordinates
    method : str, default 'step'
        Method used to compute the track length (one of 'displacement', 'step',
        'step_pca', 'step_end' or 'bin_pca')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 1
        Minimum number of points in a segment to use it in the stepping function

    Returns
    -------
    float
       Total length of the track
    '''
    if method in ['step', 'step_pca', 'step_end', 'bin_pca']:
        # Segment the track and sum the segment lengths
        segment_lengths = get_track_segments(coordinates, segment_length,
                point, method, anchor_point, min_count)[-1]

        return np.sum(segment_lengths)
    elif method == 'splines':
        # Fit point along the track with a spline, compute spline length
        raise NotImplementedError
    else:
        raise ValueError('Track length estimation method not recognized')


@nb.njit(cache=True)
def get_track_segment_dedxs(coordinates: nb.float32[:,:],
                            values: nb.float32[:],
                            end_point: nb.float32[:],
                            segment_length: nb.float32,
                            method: str = 'step',
                            anchor_point: bool = True,
                            min_count: int = 1) -> (nb.float32[:], nb.float32[:]):
    '''
    Given a set of point coordinates and their values associated with a track
    and one of its end points, compute the energy/charge deposition rate as
    a function of the residual range.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    values : np.ndarray
        (N) Values associated with each point
    end_point : np.ndarray
        (3) End point of the track
    segment_length : float
        Segment length in the units that specify the coordinates
    method : str, default 'step'
        Method used to segment the track (one of 'step', 'step_pca',
        'step_end' or 'bin_pca')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 1
        Minimum number of points in a segment for it to be valid

    Returns
    -------
    np.ndarray 
       (S) Array of energy/charge deposition rate values
    np.ndarray 
       (S) Array of residual ranges (center of the segment w.r.t. end point)
    '''
    # Get the segment indexes and their lengths
    segment_clusts, _, segment_lengths = get_track_segments(coordinates,
            segment_length, end_point, method, anchor_point, min_count)

    # Compute the dQdxs and residual ranges
    segment_dedxs  = np.empty(len(segment_clusts), dtype=np.float32)
    segment_rrs    = np.empty(len(segment_clusts), dtype=np.float32)
    residual_range = 0.
    for i, segment in enumerate(segment_clusts):
        # Compute the rate of energy/charge deposition
        # If the segment has insufficient content, return dummy values
        dx = segment_lengths[i]
        if len(segment) >= min_count and dx > 0.:
            de = np.sum(values[segment])
            segment_dedxs[i] = de/dx 
        else:
            segment_dedxs[i] = -1.

        # Compute the residual_range
        segment_rrs[i]  = residual_range + dx/2.
        residual_range += dx

    return segment_dedxs, segment_rrs


@nb.njit(cache=True)
def get_track_segments(coordinates: nb.float32[:,:],
                       segment_length: nb.float32,
                       point: nb.float32[:] = None,
                       method: str = 'step',
                       anchor_point: bool = True,
                       min_count: int = 1) -> (nb.types.List(nb.int64[:]), nb.float32[:], nb.float32):
    '''
    Given a set of point coordinates associated with a track and one of its end
    points, divide the track into segments of the requested length.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    segment_length : float
        Segment length in the units that specify the coordinates
    point : np.ndarray, optional
        (3) A preferred end point of the track from which to start
    method : str, default 'step'
        Method used to segment the track (one of 'step', 'step_pca' or
        'step_end' or'bin_pca')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 1
        Minimum number of point in segment to use segment to direct the next step

    Returns
    -------
    segment_clusts : List[np.ndarray]
       (S) List of indexes which correspond to each segment cluster of points
    segment_dirs : np.ndarray
       (S, 3) Array of segment direction vectors
    segment_lengths : np.ndarray 
       (S) Array of segment lengths
    '''
    if method == 'step' or method == 'step_pca' or method == 'step_end':
        # Determine which point to start stepping from
        if point is not None:
            segment_start = point
            if anchor_point:
                start_id      = np.argmin(nbl.cdist(np.atleast_2d(point), coordinates))
                segment_start = coordinates[start_id]
        else:
            # If not specified, pick a random end point of the track
            start_id      = nbl.farthest_pair(coordinates)[0]
            segment_start = coordinates[start_id]

        # If PCA is used, find the track principal axis to orient segements
        if method == 'step_pca':
            track_dir    = nbl.principal_components(coordinates)[0]
            pstart       = np.dot(segment_start, track_dir)
            pcoordinates = np.dot(coordinates, track_dir)
            if np.abs(np.min(pcoordinates) - pstart) > np.abs(np.max(pcoordinates) - pstart):
                track_dir = -track_dir

        # Step through the track iteratively
        segment_clusts    = nb.typed.List.empty_list(np.empty(0, dtype=np.int64))
        segment_dirs_l    = nb.typed.List.empty_list(np.empty(0, dtype=coordinates.dtype))
        segment_lengths_l = nb.typed.List.empty_list(np.float32)
        left_index = np.arange(len(coordinates))
        while len(left_index):
            # Compute distances from the segment start point to the leftover points
            dists = nbl.cdist(np.atleast_2d(segment_start), coordinates[left_index]).flatten()

            # Select the points that belong to this segment, store
            dist_mask      = dists <= segment_length
            pass_index     = np.where(dist_mask)[0]
            fail_index     = np.where(~dist_mask)[0]
            segment_index  = left_index[pass_index]
            segment_clusts.append(segment_index)

            # Estimate the direction w.r.t. the segment start point ('step')
            # or by taking the PCA of the segment ('step_pca'), or by finding
            # the closest point that is not in the segment ('step_end')
            # If there are no points in this segment but there are leftovers,
            # step in the direction of the next closest point
            if method != 'step_end' and len(segment_index) > min_count and np.max(dists[pass_index]) > 0.:
                if method == 'step_pca':
                    direction = nbl.principal_components(coordinates[segment_index])[0]
                    if np.dot(direction, track_dir) < 0.:
                        direction = -direction
                else:
                    direction = nbl.mean(coordinates[segment_index] - segment_start, axis=0)
            elif len(fail_index):
                direction = coordinates[left_index[fail_index][np.argmin(dists[fail_index])]] - segment_start
            else:
                direction = coordinates[segment_index[np.argmax(dists)]] - segment_start
            direction /= np.linalg.norm(direction)
            segment_dirs_l.append(direction)

            # Store the segment length. If it's the last segment,
            # track the distance to the fathest point in the segment
            if len(fail_index):
                length = segment_length
            else:
                to_end = coordinates[segment_index[np.argmax(dists)]] - segment_start
                length = np.dot(to_end, direction)
            segment_lengths_l.append(length)

            # Step the segment start point in the direction of the segment
            segment_start += length * direction

            # Update the leftover index
            left_index = left_index[fail_index]

        # Convert lists of directions and lengths to numpy.ndarray objects
        segment_dirs    = np.empty((len(segment_dirs_l), coordinates.shape[1]), dtype=coordinates.dtype)
        segment_lengths = np.empty(len(segment_lengths_l), dtype=np.float64)
        for i in range(len(segment_clusts)):
            segment_dirs[i]    = segment_dirs_l[i]
            segment_lengths[i] = segment_lengths_l[i]

        return segment_clusts, segment_dirs, segment_lengths

    elif method == 'bin_pca':
        # Find the principal component of the whole track
        track_dir = nbl.principal_components(coordinates)[0]

        # If a track end point is provided, check which end the track end point
        # is on and flip the principal axis coordinates, if needed
        pcoordinates = np.dot(coordinates, track_dir)
        if point is not None:
            pstart = np.dot(point, track_dir)
            if np.abs(np.min(pcoordinates) - pstart) > np.abs(np.max(pcoordinates) - pstart):
                pstart = -pstart
                pcoordinates = -pcoordinates

        # Bin the track along the principal component vector
        if point is not None:
            boundaries = np.arange(min(pstart, np.min(pcoordinates)), np.max(pcoordinates), segment_length)
        else:
            boundaries = np.arange(np.min(pcoordinates), np.max(pcoordinates), segment_length)

        segment_labels = np.digitize(pcoordinates, boundaries)
        segment_clusts = nb.typed.List.empty_list(np.empty(0, dtype=np.int64))
        for l in range(len(boundaries)):
            segment_clusts.append(np.where(segment_labels == l+1)[0])

        # Compute the segment directions and lengths
        segment_dirs    = np.empty((len(segment_clusts), coordinates.shape[1]), dtype=coordinates.dtype)
        segment_lengths = np.empty(len(segment_clusts), dtype=np.float64)
        for i, segment in enumerate(segment_clusts):
            # If this segment is empty, use track-level information
            if not len(segment):
                segment_dirs[i]    = track_dir
                segment_lengths[i] = segment_length
                continue

            # Compute the principal component of the segment, use it as direction
            if len(segment) > min_count:
                direction = nbl.principal_components(coordinates[segment])[0]
                if np.dot(direction, track_dir) < 0.:
                    direction = -direction
            else:
                direction = track_dir
            segment_dirs[i] = direction

            # Evaluate the length of the segment as constrained by the track
            # principal axis bin that defines it
            if i < len(segment_clusts) - 1:
                length = segment_length/np.dot(direction, track_dir)
            else:
                length = (np.max(pcoordinates)-boundaries[-1])/np.dot(direction, track_dir)
            segment_lengths[i] = length

        return segment_clusts, segment_dirs, segment_lengths

    else:
        raise ValueError('Track segmentation method not recognized')
