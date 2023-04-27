import numpy as np
from collections import defaultdict

def filter_opflashes(opflashes, beam_window=(0, 1.6), tolerance=0.4):
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
    
    for key in opflashes:
        for flash in opflashes[key]:
            if (flash.time() < beam_window[1] + tolerance) and \
               (flash.time() > beam_window[0] - tolerance):
                out_flashes[key].append(flash)
                
    return out_flashes