import numpy as np
from collections import defaultdict
from analysis.post_processing import post_processing
from mlreco.utils.globals import *

@post_processing(data_capture=['index', 'opflash_cryoE', 'opflash_cryoW'], 
                 result_capture=['interactions'])
def run_flash_matching(data_dict, result_dict, 
                       fm=None,
                       opflash_keys=[]):
    """
    Post processor for running flash matching using OpT0Finder.
    
    Parameters
    ----------
    fm : FlashManager
    opflash_keys : List[str]

    Returns
    -------
    Empty dict (operation is in-place)
        
    NOTE: This post-processor also modifies the list of Interactions
    in-place by adding the following attributes:
        interaction.fmatched: (bool)
            Indicator for whether the given interaction has a flash match
        interaction.fmatch_time: float
            The flash time in microseconds 
        interaction.fmatch_total_pE: float
        interaction.fmatch_id: int
    """
    print("Running flash matching...")
    opflashes = {}
    assert len(opflash_keys) > 0
    for key in opflash_keys:
        opflashes[key] = data_dict[key]
    
    interactions = result_dict['interactions']
    entry        = data_dict['index']
    
    fmatches_E = fm.get_flash_matches(int(entry), 
                                      interactions,
                                      opflashes,
                                      volume=0,
                                      restrict_interactions=[], 
                                      cache=False)
    fmatches_W = fm.get_flash_matches(int(entry), 
                                      interactions,
                                      opflashes,
                                      volume=1,
                                      restrict_interactions=[],
                                      cache=False)

    flash_dict_E = {}
    for ia, flash, match in fmatches_E:
        flash_dict_E[ia.id] = (flash, match)
        ia.fmatched = True
        ia.flash_time = float(flash.time())
        ia.flash_total_pE = float(flash.TotalPE())
        ia.flash_id = int(flash.id())
        ia.flash_hypothesis = float(np.array(match.hypothesis).sum())
        
    flash_dict_W = {}
    for ia, flash, match in fmatches_W:
        flash_dict_W[ia.id] = (flash, match)
        ia.fmatched = True
        ia.flash_time = float(flash.time())
        ia.flash_total_pE = float(flash.TotalPE())
        ia.flash_id = int(flash.id())
        ia.flash_hypothesis = float(np.array(match.hypothesis).sum())

    print("Done flash matching.")
    return {}