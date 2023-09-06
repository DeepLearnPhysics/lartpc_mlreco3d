import numpy as np
from functools import partial 

from analysis.post_processing import post_processing
from analysis.post_processing.reconstruction.calorimetry import _pix_to_cm, get_splines, compute_track_length, compute_track_length_splines
from mlreco.utils.globals import *
from mlreco.utils.tracking import get_track_segments, get_track_segment_dedxs, get_track_length


@post_processing(data_capture=['meta'], 
                 result_capture=['particles'],
                 result_capture_optional=['truth_particles'])
def reconstruct_track_energy(data_dict, result_dict,
                             data=False,
                             include_pids=[2,4],
                             tracking_mode='default',
                             truth_point_mode='points',
                             convert_to_cm=False,
                             **kwargs):
    
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
    
    bin_size = kwargs.get('bin_size', 20)
    
    for p in particles:
        if not ((p.semantic_type == 1) and (p.pid in include_pids)):
            continue
        if convert_to_cm:
            coordinates = _pix_to_cm(p.points, meta)
            startpoint  = _pix_to_cm(p.start_point, meta)
            bin_size    = bin_size * px_to_cm
        else:
            coordinates = p.points
            startpoint  = p.start_point
        if tracking_mode == 'default':
            length = compute_track_length(coordinates, bin_size=bin_size)
        elif tracking_mode == 'spline':
            if coordinates.shape[0] > bin_size:
                length = compute_track_length_splines(coordinates, bin_size=bin_size)
            else:
                length = compute_track_length(coordinates, bin_size=bin_size)
        elif tracking_mode == 'numba':
            coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
            startpoint = np.ascontiguousarray(startpoint, dtype=np.float32)
            length = get_track_length(coordinates, segment_length=bin_size,
                                      point=startpoint, 
                                      method=kwargs.get('method', 'step'),
                                      anchor_point=kwargs.get('anchor_point', True))
        else:
            raise ValueError(f"Track length reconstruction module {tracking_mode} is not supported!")
        p.length = length
        p.csda_kinetic_energy = splines[p.pid](length)
        
    if len(truth_particles) > 0:
        for p in truth_particles:
            if not ((p.semantic_type == 1) and (p.pid in include_pids)):
                continue
            if convert_to_cm:
                coordinates = getattr(p, truth_point_mode)
                coordinates = _pix_to_cm(coordinates, meta)
                startpoint  = _pix_to_cm(p.start_point, meta)
                bin_size    = bin_size * px_to_cm
            else:
                # assert p.units == 'cm'
                coordinates = getattr(p, truth_point_mode)
                startpoint  = p.start_point
                if tracking_mode == 'default':
                    length = compute_track_length(coordinates, bin_size=bin_size)
                elif tracking_mode == 'spline':
                    if coordinates.shape[0] > bin_size:
                        length = compute_track_length_splines(coordinates, bin_size=bin_size)
                    else:
                        length = compute_track_length(coordinates, bin_size=bin_size)
                elif tracking_mode == 'numba':
                    coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
                    startpoint = np.ascontiguousarray(startpoint, dtype=np.float32)
                    length = get_track_length(coordinates, segment_length=bin_size,
                                            point=startpoint, 
                                            method=kwargs.get('method', 'step'),
                                            anchor_point=kwargs.get('anchor_point', True))
                else:
                    raise ValueError(f"Track length reconstruction module {tracking_mode} is not supported!")
                p.length = length
                p.csda_kinetic_energy = splines[p.pid](length)
            
    return {}