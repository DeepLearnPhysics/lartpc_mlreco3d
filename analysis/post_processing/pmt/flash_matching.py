import numpy as np
import yaml
from pprint import pprint
from collections import defaultdict

from mlreco.utils.gnn.cluster import get_cluster_directions
from analysis.post_processing import post_processing
from mlreco.utils.globals import *
from mlreco.main_funcs import process_config
from . import FlashMatcherInterface


@post_processing(data_capture=['meta', 'index', 'opflash_cryoE', 'opflash_cryoW'], 
                 result_capture=['Interactions'])
def run_flash_matching(data_dict, result_dict, 
                       config_path=None,
                       fmatch_config=None, 
                       reflash_merging_window=None,
                       volume_boundaries=None,
                       ADC_to_MeV=1.,
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
    for key in opflash_keys:
        opflashes[key] = data_dict[key]

    ADC_to_MeV = ADC_TO_MEV
    
    if config_path is None:
        raise ValueError("You need to give the path to your full chain config.")
    if fmatch_config is None:
        raise ValueError("You need a flash matching config to run flash matching.")
    if volume_boundaries is None:
        raise ValueError("You need to set volume boundaries to run flash matching.")
    
    config = yaml.safe_load(open(config_path, 'r'))
    process_config(config, verbose=False)

    fm = FlashMatcherInterface(config, fmatch_config, 
                               boundaries=volume_boundaries,
                               opflash_keys=opflash_keys, 
                               reflash_merging_window=reflash_merging_window)
    fm.initialize_flash_manager(data_dict['meta'][0])

    update_dict = {}

    flash_matches_cryoE = []
    flash_matches_cryoW = []
    
    for entry, image_id in enumerate(data_dict['index']):
        interactions = result_dict['Interactions'][entry]

        fmatches_E = fm.get_flash_matches(entry, 
                                          interactions,
                                          opflashes,
                                          use_true_tpc_objects=False,
                                          volume=0,
                                          use_depositions_MeV=False,
                                          ADC_to_MeV=ADC_to_MeV,
                                          restrict_interactions=[])
        fmatches_W = fm.get_flash_matches(entry, 
                                          interactions,
                                          opflashes,
                                          use_true_tpc_objects=False,
                                          volume=1,
                                          use_depositions_MeV=False,
                                          ADC_to_MeV=ADC_to_MeV,
                                          restrict_interactions=[])
        flash_matches_cryoE.append(fmatches_E)
        flash_matches_cryoW.append(fmatches_W)

    update_dict = defaultdict(list)

    for tuple_list in flash_matches_cryoE:
        flash_dict_E = {}
        for ia, flash, match in tuple_list:
            flash_dict_E[ia.id] = (flash, match)
            ia.fmatched = True
            ia.fmatch_time = flash.time()
            ia.fmatch_total_pE = flash.TotalPE()
            ia.fmatch_id = flash.id()
        update_dict['flash_matches_cryoE'].append(flash_dict_E)
        
    for tuple_list in flash_matches_cryoW:
        flash_dict_W = {}
        for ia, flash, match in tuple_list:
            flash_dict_W[ia.id] = (flash, match)
            ia.fmatched = True
            ia.fmatch_time = flash.time()
            ia.fmatch_total_pE = flash.TotalPE()
            ia.fmatch_id = flash.id()
        update_dict['flash_matches_cryoW'].append(flash_dict_W)

    assert len(update_dict['flash_matches_cryoE'])\
           == len(update_dict['flash_matches_cryoW'])

    return update_dict