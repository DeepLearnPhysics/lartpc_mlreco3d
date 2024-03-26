"""Module with functions/classes used to identify particle species."""

import numpy as np

from .globals import PID_MASSES
from .tracking import get_track_segment_dedxs
from .energy_loss import (
        csda_table_spline, csda_ke_lar, bethe_bloch_lar, bethe_bloch_mpv_lar)


def get_track_deposition_chi2(coordinates, values, end_point, pid,
                              segment_length=5.0, method='step_next',
                              anchor_point=True, min_count=10, use_table=False,
                              use_mpv=False):
    '''
    Computes the Chi-squared measure of the agreement between dE/dxs
    measurements taken from track segment and the expected dE/dx for those.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) Coordinates of the points that make up the track
    values : np.ndarray
        (N) Values associated with each point
    end_point : np.ndarray, optional
        (3) End point of the track
    pid : int
        Particle species, as defined in globals
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
    use_table : bool, default False
        Use tabulated values of dE/dx vs residual range
    use_mpv : bool, default False
        Use most-probable energy deposition value

    Returns
    -------
    chi2 : float
       Measure of the agreement between the PID hypothesis and the dE/dx profile
    seg_dedxs : np.ndarray
       (S) Array of energy/charge deposition rate values
    seg_errs : np.ndarray
       (S) Array of uncertainties on the energy/charge deposition rate
    seg_rrs : np.ndarray
       (S) Array of residual ranges (center of the segment w.r.t. end point)
    '''
    # No tabulated dE/dx for MPV, throw if needed
    assert not use_table or not use_mpv, (
            "No available tabulated dE/dx for MPV values.")

    # Compute the track segment dE/dxs
    seg_dedxs, seg_errs, seg_rrs, _, _, seg_lengths = \
            get_track_segment_dedxs(coordinates, values, end_point,
                    segment_length, method, anchor_point, min_count)

    # Get the expected value of dE/dx for each value of the residual range
    exp_dedxs = np.empty(len(seg_rrs), dtype=seg_dedxs.dtype)
    if use_table:
        table = csda_table_spline(pid, value='dE/dx')
        for i in range(len(seg_rrs)):
            dedx = table(seg_rrs[i])
            exp_dedxs[i] = dedx
    else:
        mass = PID_MASSES[pid]
        for i in range(len(seg_rrs)):
            T = csda_ke_lar(seg_rrs[i], mass)
            if not use_mpv:
                dedx = -bethe_bloch_lar(T, mass)
            else:
                dedx = -bethe_bloch_mpv_lar(T, mass, 1)
            exp_dedxs[i] = dedx

    # Evaluate the agreement between observation and theory
    mask = np.where(seg_dedxs > -1)[0]
    chi2 = np.sum((seg_dedxs[mask] - exp_dedxs[mask])**2/seg_errs[mask]**2)
    if len(mask):
        chi2 /= len(mask)

    return chi2, seg_dedxs, seg_errs, seg_rrs
