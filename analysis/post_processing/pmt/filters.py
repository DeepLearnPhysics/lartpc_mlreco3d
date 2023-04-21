import numpy as np
from collections import defaultdict

def filter_opflashes(opflashes, beam_window=(0, 1.6)):
    """Python implementation for filtering opflashes.
    
    Only meant to be temporary, will be implemented in C++ to OpT0Finder. 

    Parameters
    ----------
    opflashes : dict
        Dictionary of List[larcv.Flash], corresponding to each 
        east and west cryostat.

    Returns
    -------
    out_flashes : dict
        filtered List[larcv.Flash] dictionary.
    """
    
    out_flashes = defaultdict(list)
    flash_dist_dict = {}
    
    for key in opflashes:
        for flash in opflashes[key]:
            if (flash.time() < beam_window[1]) and (flash.time() > beam_window[0]):
                flash_dist_dict[flash.id()] = 0
            else:
                dt1 = flash.time() - beam_window[0]
                dt2 = flash.time() - beam_window[1]
                dt = max(dt1, dt2)
                flash_dist_dict[flash.id()] = dt
                
            
    return out_flashes