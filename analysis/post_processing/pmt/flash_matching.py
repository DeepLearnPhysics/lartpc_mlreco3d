import numpy as np

from mlreco.utils.gnn.cluster import get_cluster_directions
from analysis.post_processing import post_processing
from mlreco.utils.globals import *
from . import FlashManager


@post_processing(data_capture=['meta'], result_capture=[])
def run_flash_matching(data_dict, result_dict, 
                       fmatch_config=None, 
                       reflash_merging_window=None,
                       volume_boundaries=None):
    
    if fmatch_config is None:
        raise ValueError("You need a flash matching config to run flash matching.")
    if volume_boundaries is None:
        raise ValueError("You need to set volume boundaries to run flash matching.")
    
    opflash_keys = []