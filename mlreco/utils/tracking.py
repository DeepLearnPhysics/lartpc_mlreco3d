import numpy as np
import numba as nb

from scipy.interpolate import UnivariateSpline

from . import numba_local as nbl


def get_track_length(coordinates: nb.float32[:,:],
                     segment_length: nb.float32 = None,
                     point: nb.float32[:] = None,
                     method: str = 'bin_pca',
                     anchor_point: bool = True,
                     min_count: int = 10,
                     spline_smooth: float = None) -> nb.float32:
    '''
    Given a set of point coordinates associated with a track and one of its end
    points, compute its length.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    segment_length : float, optional
        Segment length in the units that specify the coordinates
    point : np.ndarray, optional
        (3) An end point of the track
    method : str, default 'bin_pca'
        Method used to compute the track length (one of 'displacement', 'step',
        'step_next', 'bin_pca' or 'spline')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 10
        Minimum number of points in a segment to use it to evaluate the
        direction of the next step along the track.
    spline_smooth : float, optional
        The smoothing factor to be used in spline regression, when used

    Returns
    -------
    float
       Total length of the track
    '''
    if method == 'displacement':
        # Project points along the principal component, compute displacement
        track_dir = nbl.principal_components(coordinates)[0]
        pcoordinates = np.dot(coordinates, track_dir)

        return np.max(pcoordinates) - np.min(pcoordinates)

    if method in ['step', 'step_next', 'bin_pca']:
        # Segment the track and sum the segment lengths
        seg_lengths = get_track_segments(coordinates, segment_length,
                point, method, anchor_point, min_count)[-1]

        return np.sum(seg_lengths)

    elif method == 'splines':
        # Fit point along the track with a spline, compute spline length
        return get_track_spline(coordinates, segment_length, spline_smooth)[-1]

    else:
        raise ValueError('Track length estimation method ' \
                f'not recognized: {method}')


@nb.njit(cache=True)
def check_track_orientation(coordinates: nb.float32[:,:],
                            values: nb.float32[:],
                            start_point: nb.float32[:],
                            end_point: nb.float32[:],
                            method: str = 'local',
                            anchor_points: bool = True,
                            local_radius: nb.float32 = 5,
                            segment_method: str = 'step_next',
                            segment_length: nb.float32 = 5,
                            segment_min_count: int = 10) -> bool:
    '''
    Given a set of track point coordinates and the track end points, checks
    which end point is most likely to be the correct start, based on the
    rate of energy deposition in the track.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    values : np.ndarray
        (N) Values associated with each point
    start_point : np.ndarray
        (3) Start point of the track
    end_point : np.ndarray
        (3) End point of the track
    method : str, default 'local'
        Method used to orient the track (one of 'local', 'gradient')
    local_radius : float, default 5
        Radius around the end points to used to evaluate the local dE/dx
    anchor_points : bool, default True
        Weather or not to collapse end point onto the closest track point
    segment_method : str, default 'step_next'
        Method used to segment the track when using the 'gradient' method
    segment_length : float, default 5
        Segment length when using the 'gradient' method
    segment_min_count : int, default 10
        Minimum number of points in a segment when using the 'gradient' method

    Returns
    -------
    bool
       Returns `True` if the start point provided is correct, `False`
       if the end point is more likely to be the start point.
    '''
    if method == 'local':
        # If requested, anchor the end points to the closest track points
        end_points = np.vstack((start_point, end_point))
        if anchor_points:
            dist_mat = nbl.cdist(end_points, coordinates)
            end_ids  = nbl.argmin(dist_mat, axis=1)
            end_points = coordinates[end_ids]

        # Compute the local dE/dx around each end, pick the end with the lowest
        start_dedx  = np.sum(values[dist_mat[0] < local_radius])/local_radius
        end_dedx    = np.sum(values[dist_mat[1] < local_radius])/local_radius

        return start_dedx < end_dedx

    elif method == 'gradient':
        # Compute the track gradient with respect to either ends
        grad_start = get_track_deposition_gradient(coordinates,
                values, start_point, segment_length, segment_method,
                anchor_points, segment_min_count)[0]
        grad_end = get_track_deposition_gradient(coordinates,
                values, end_point, segment_length, segment_method,
                anchor_points, segment_min_count)[0]

        # Compute the deposition gradient as an average of the two
        gradient = (grad_start - grad_end) / 2.

        return bool(gradient >= 0.)

    else:
        raise ValueError('Track orientation method not recognized')


@nb.njit(cache=True)
def get_track_deposition_gradient(coordinates: nb.float32[:,:],
                                  values: nb.float32[:],
                                  start_point: nb.float32[:],
                                  segment_length: nb.float32 = 5.,
                                  method: str = 'step_next',
                                  anchor_point: bool = True,
                                  min_count: int = 10) -> (nb.float32,
                                          nb.float32[:], nb.float32[:],
                                          nb.float32[:]):
    '''
    Given a set of point coordinates and their values associated with a track
    and a start point, compute the deposition gradient with respect to the
    start point.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    values : np.ndarray
        (N) Values associated with each point
    start_point : np.ndarray
        (3) End point of the track
    segment_length : float, default 5
        Segment length in the units that specify the coordinates
    method : str, default 'step_next'
        Method used to segment the track (one of 'step', 'step_next'
        or 'bin_pca')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 10
        Minimum number of points in a segment for it to be valid. If not valid,
        the dedx value for that segment is not used to compute the gradient.

    Returns
    -------
    gradient : float
       Deposition gradient along the track from the start point
    segment_dedxs : np.ndarray
       (S) Array of energy/charge deposition rate values
    segment_rrs : np.ndarray
       (S) Array of residual ranges (center of the segment w.r.t. end point)
    segment_lengths : np.ndarray
       (S) Array of segment lengths
    '''
    # Compute the track segment dedxs
    seg_dedxs, seg_rrs, _, _, seg_lengths = \
            get_track_segment_dedxs(coordinates, values, start_point,
                    segment_length, method, anchor_point, min_count)

    valid_index = np.where(seg_dedxs > -1)[0]
    if not len(valid_index):
        return 0., seg_dedxs, seg_rrs, seg_lengths

    seg_dedxs = seg_dedxs[valid_index]
    seg_rrs = seg_rrs[valid_index]

    # Compute the dE/dx gradient
    gradient = np.cov(seg_rrs, seg_dedxs)[0,1]/np.std(seg_rrs)**2 \
            if np.std(seg_rrs) > 0. else 0.

    return gradient, seg_dedxs, seg_rrs, seg_lengths


@nb.njit(cache=True)
def get_track_segment_dedxs(coordinates: nb.float32[:,:],
                            values: nb.float32[:],
                            end_point: nb.float32[:] = None,
                            segment_length: nb.float32 = 5.,
                            method: str = 'step_next',
                            anchor_point: bool = True,
                            min_count: int = 10) -> (nb.float32[:],
                                    nb.float32[:], nb.float32[:],
                                    nb.float32[:], nb.float32[:]):
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
    end_point : np.ndarray, optional
        (3) End point of the track
    segment_length : float, default 5.
        Segment length in the units that specify the coordinates
    method : str, default 'step_next'
        Method used to segment the track (one of 'step', 'step_next'
        or 'bin_pca')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 10
        Minimum number of points in a segment for it to be valid. If not valid,
        the dedx value returned for the segment is -1.

    Returns
    -------
    seg_dedxs : np.ndarray
       (S) Array of energy/charge deposition rate values
    seg_rrs : np.ndarray
       (S) Array of residual ranges (center of the segment w.r.t. end point)
    seg_clusts : List[np.ndarray]
       (S) List of indexes which correspond to each segment cluster of points
    seg_dirs : np.ndarray
       (S, 3) Array of segment direction vectors
    seg_lengths : np.ndarray
       (S) Array of segment lengths
    '''
    # Get the segment indexes and their lengths
    seg_clusts, seg_dirs, seg_lengths = get_track_segments(coordinates,
            segment_length, end_point, method, anchor_point, min_count)

    # Compute the dQdxs and residual ranges
    seg_dedxs = np.empty(len(seg_clusts), dtype=np.float32)
    seg_rrs  = np.empty(len(seg_clusts), dtype=np.float32)
    residual_range = 0.
    for i, seg in enumerate(seg_clusts):
        # Compute the rate of energy/charge deposition
        # If the segment has insufficient content, return dummy values
        dx = seg_lengths[i]
        if len(seg) >= min_count and dx > 0.:
            de = np.sum(values[seg])
            seg_dedxs[i] = de/dx
        else:
            seg_dedxs[i] = -1.

        # Compute the residual_range
        seg_rrs[i]  = residual_range + dx/2.
        residual_range += dx

    return seg_dedxs, seg_rrs, seg_clusts, seg_dirs, seg_lengths


@nb.njit(cache=True)
def get_track_segments(coordinates: nb.float32[:,:],
                       segment_length: nb.float32,
                       point: nb.float32[:] = None,
                       method: str = 'step_next',
                       anchor_point: bool = True,
                       min_count: int = 10) -> (nb.types.List(nb.int64[:]),
                               nb.float32[:], nb.float32):
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
    method : str, default 'step_next'
        Method used to segment the track (one of 'step', 'step_next'
        or 'bin_pca')
    anchor_point : bool, default True
        Weather or not to collapse end point onto the closest track point
    min_count : int, default 10
        Minimum number of points in a segment to use it to evaluate the
        direction of the next step along the track.

    Returns
    -------
    segment_clusts : List[np.ndarray]
       (S) List of indexes which correspond to each segment cluster of points
    segment_dirs : np.ndarray
       (S, 3) Array of segment direction vectors
    segment_lengths : np.ndarray
       (S) Array of segment lengths
    '''
    if method == 'step' or method == 'step_next':
        # Determine which point to start stepping from
        if point is not None:
            start_point = point
            if anchor_point:
                start_id    = np.argmin(nbl.cdist(np.atleast_2d(point),
                    coordinates))
                start_point = coordinates[start_id]
        else:
            # If not specified, pick a random end point of the track
            start_id    = nbl.farthest_pair(coordinates)[0]
            start_point = coordinates[start_id]

        # Step through the track iteratively
        empty_list    = nb.typed.List.empty_list
        seg_start     = np.copy(start_point)
        seg_clusts    = empty_list(np.empty(0, dtype=np.int64))
        seg_dirs_l    = empty_list(np.empty(0, dtype=coordinates.dtype))
        seg_lengths_l = empty_list(np.empty(1, dtype=coordinates.dtype).item())
        left_index = np.arange(len(coordinates))
        while len(left_index):
            # Compute distances from the segment start point to the all
            # the leftover points
            dists = nbl.cdist(np.atleast_2d(seg_start),
                    coordinates[left_index]).flatten()

            # Select the points that belong to this segment
            dist_mask  = dists <= segment_length
            pass_index = np.where(dist_mask)[0]
            fail_index = np.where(~dist_mask)[0]
            seg_index  = left_index[pass_index]

            # If the next closest point is backwards, make it the last segment
            if len(fail_index):
                next_id = np.argmin(dists[fail_index])
                next_index = left_index[fail_index][next_id]
                next_closest = coordinates[next_index]
                if np.dot(next_closest - seg_start,
                        seg_start - start_point) < 0.:
                    fail_index = fail_index[:0]
                    if not len(pass_index): break

            # Estimate the direction of the segment
            seg_coords = coordinates[seg_index]
            if method == 'step' \
                    and len(seg_index) > min_count \
                    and np.max(dists[pass_index]) > 0.:
                # Estimate direction w.r.t. the segment start point ('step')
                direction = nbl.mean(seg_coords - seg_start, axis=0)
            elif len(fail_index):
                # Take direction as the vector joining the next closest point
                # ('step_next'). Also apply this method is the `min_count`
                # condition is not satisfied for this segment.
                direction = next_closest - seg_start
            else:
                # If this is the last segment, find the farthest point from
                # its start and make it the last point.
                last_id = np.argmax(dists[pass_index])
                direction = seg_coords[last_id] - seg_start

            if np.linalg.norm(direction):
                direction /= np.linalg.norm(direction)
            else:
                direction = np.array([1.,0.,0.], dtype=coordinates.dtype)

            # Compute the segment length. If it's the last segment,
            # track the distance to the fathest point in the segment
            if len(fail_index):
                length = segment_length
            else:
                last_id = np.argmax(dists[pass_index])
                to_end = seg_coords[last_id] - seg_start
                length = np.dot(to_end, direction)

            # Step the segment start point in the direction of the segment
            seg_start += length * direction

            # Update the leftover index
            left_index = left_index[fail_index]

            # Store the segment information
            seg_clusts.append(seg_index)
            seg_dirs_l.append(direction)
            seg_lengths_l.append(length)

        # Convert lists of directions and lengths to numpy.ndarray objects
        shape = (len(seg_dirs_l), coordinates.shape[1])
        seg_dirs    = np.empty(shape, dtype=coordinates.dtype)
        seg_lengths = np.empty(shape[0], dtype=coordinates.dtype)
        for i in range(len(seg_clusts)):
            seg_dirs[i]    = seg_dirs_l[i]
            seg_lengths[i] = seg_lengths_l[i]

        return seg_clusts, seg_dirs, seg_lengths

    elif method == 'bin_pca':
        # Find the principal component of the whole track
        track_dir = nbl.principal_components(coordinates)[0]

        # If a track end point is provided, check which end the track end point
        # is on and flip the principal axis coordinates, if needed
        pcoordinates = np.dot(coordinates, track_dir)
        if point is not None:
            pstart = np.dot(point, track_dir)
            if np.abs(np.min(pcoordinates) - pstart) \
                    > np.abs(np.max(pcoordinates) - pstart):
                pstart = -pstart
                pcoordinates = -pcoordinates

        # Bin the track along the principal component vector
        if point is not None:
            boundaries = np.arange(min(pstart, np.min(pcoordinates)),
                    np.max(pcoordinates), segment_length)
        else:
            boundaries = np.arange(np.min(pcoordinates),
                    np.max(pcoordinates), segment_length)

        seg_labels = np.digitize(pcoordinates, boundaries)
        seg_clusts = nb.typed.List.empty_list(np.empty(0, dtype=np.int64))
        for l in range(len(boundaries)):
            seg_clusts.append(np.where(seg_labels == l+1)[0])

        # Compute the segment directions and lengths
        seg_dirs    = np.empty((len(seg_clusts), coordinates.shape[1]),
                dtype=coordinates.dtype)
        seg_lengths = np.empty(len(seg_clusts), dtype=coordinates.dtype)
        for i, seg in enumerate(seg_clusts):
            # If this segment is empty, use track-level information
            if not len(seg):
                seg_dirs[i]    = track_dir
                seg_lengths[i] = segment_length
                continue

            # Compute principal component of the segment, use it as direction
            if len(seg) > min_count:
                direction = nbl.principal_components(coordinates[seg])[0]
                if np.dot(direction, track_dir) < 0.:
                    direction = -direction
            else:
                direction = track_dir
            seg_dirs[i] = direction

            # Evaluate length of the segment as constrained by the track
            # principal axis bin that defines it
            if i < len(seg_clusts) - 1:
                length = segment_length/np.dot(direction, track_dir)
            else:
                length = (np.max(pcoordinates)-boundaries[-1]) \
                        / np.dot(direction, track_dir)
            seg_lengths[i] = length

        return seg_clusts, seg_dirs, seg_lengths

    else:
        raise ValueError('Track segmentation method not recognized')


def get_track_spline(coordinates, segment_length, s=None):
    '''
    Estimate the best approximating curve defined by a point cloud
    using univariate 3D splines.

    The length is computed by measuring the length of the piecewise linear
    interpolation of the spline at points defined by the bin size.

    Parameters
    ----------
    coordinatea : np.ndarray
        (N, 3) point cloud
    segment_length : float
        The subdivision length at which to sample points from the spline.
        If the track length is less than the segment_length, then the returned
        length will be computed from the farthest two projected points along
        the track's principal direction.
    s : float, optional
        The smoothing factor to be used in spline regression, by default None

    Returns
    -------
    u : np.ndarray
        (N) The principal axis parametrization of the curve
        C(u) = (spx(u), spy(u), spz(u))
    sppoints : np.ndarray
        (N, 3) The graph of the spline at points u
    splines : scipy.interpolate.UnivariateSpline
        Approximating splines for the point cloud defined by points
    length : float
        The estimate of the total length of the curve
    '''
    # Compute the principal component along which to segment the track
    track_dir = nbl.principal_components(coordinates)[0]
    pcoords   = np.dot(coordinates, track_dir)
    perm      = np.argsort(pcoords.squeeze())
    u         = pcoords[perm]

    # If there is less than four points, cannot fit a 3D spline
    if len(coordinates) < 4:
        # Fall back on displacement to estimate length
        length = np.max(pcoords) - np.min(pcoords)
        return u.squeeze, None, None, length

    # Compute the univariate splines along each axis
    spx = UnivariateSpline(u, coordinates[perm][:, 0], s=s)
    spy = UnivariateSpline(u, coordinates[perm][:, 1], s=s)
    spz = UnivariateSpline(u, coordinates[perm][:, 2], s=s)
    sppoints = np.hstack([spx(u), spy(u), spz(u)])
    splines  = [spx, spy, spz]

    # If track length is less than segment_length, just return length.
    # Otherwise estimate length by piecewise linear interpolation
    start, end = u.min(), u.max()
    length = end - start
    if length > segment_length:
        bins = np.arange(u.min(), u.max(), segment_length)
        bins = np.hstack([bins, np.array([u.max()])])
        pt_approx = np.hstack([sp(bins).reshape(-1, 1) for sp in splines])
        segments = np.linalg.norm(pt_approx[1:] - pt_approx[:-1], axis=1)
        length = segments.sum()

    return u.squeeze(), sppoints, splines, length
