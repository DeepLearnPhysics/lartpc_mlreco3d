import numpy as np
from collections import defaultdict
from analysis.post_processing import post_processing
from mlreco.utils.globals import *
from .filters import filter_opflashes

@post_processing(data_capture=['meta', 'index', 'opflash_cryoE', 'opflash_cryoW'], 
                 result_capture=['Interactions'])
def run_flash_matching(data_dict, result_dict, 
                       fm=None,
                       opflash_keys=[]):
    """
    Post processor for running flash matching using OpT0Finder.
    
    Parameters
    ----------
    config_path: str
        Path to current model's .cfg file.
    fmatch_config: str
        Path to flash matching config
    reflash_merging_window: float
    volume_boundaries: np.ndarray or list
    ADC_to_MeV: float
    opflash_keys: list of str

    Returns
    -------
    update_dict: dict of list
        Dictionary of a list of length batch_size, where each entry in 
        the list is a mapping:
            interaction_id : (larcv.Flash, flashmatch.FlashMatch_t)
        
    NOTE: This post-processor also modifies the list of Interactions
    in-place by adding the following attributes:
        interaction.fmatched: (bool)
            Indicator for whether the given interaction has a flash match
        interaction.fmatch_time: float
            The flash time in microseconds 
        interaction.fmatch_total_pE: float
        interaction.fmatch_id: int
    """
    
    opflashes = {}
    assert len(opflash_keys) > 0
    for key in opflash_keys:
        opflashes[key] = data_dict[key]

    update_dict = {}
    
    interactions = result_dict['Interactions']
    entry        = data_dict['index']
    
    opflashes = filter_opflashes(opflashes)
    
    fmatches_E = fm.get_flash_matches(entry, 
                                      interactions,
                                      opflashes,
                                      volume=0,
                                      restrict_interactions=[])
    fmatches_W = fm.get_flash_matches(entry, 
                                      interactions,
                                      opflashes,
                                      volume=1,
                                      restrict_interactions=[])

    update_dict = defaultdict(list)

    flash_dict_E = {}
    for ia, flash, match in fmatches_E:
        flash_dict_E[ia.id] = (flash, match)
        ia.fmatched = True
        ia.fmatch_time = flash.time()
        ia.fmatch_total_pE = flash.TotalPE()
        ia.fmatch_id = flash.id()
    update_dict['flash_matches_cryoE'].append(flash_dict_E)
        
    flash_dict_W = {}
    for ia, flash, match in fmatches_W:
        flash_dict_W[ia.id] = (flash, match)
        ia.fmatched = True
        ia.fmatch_time = flash.time()
        ia.fmatch_total_pE = flash.TotalPE()
        ia.fmatch_id = flash.id()
    update_dict['flash_matches_cryoW'].append(flash_dict_W)

    assert len(update_dict['flash_matches_cryoE'])\
           == len(update_dict['flash_matches_cryoW'])

    return update_dict