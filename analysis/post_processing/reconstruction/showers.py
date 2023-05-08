import numpy as np
import numba as nb

from analysis.post_processing import post_processing
from mlreco.utils.globals import *

# @post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
#                                                               'particle_clusts',
#                                                               'particle_start_points',
#                                                               'particle_end_points'])