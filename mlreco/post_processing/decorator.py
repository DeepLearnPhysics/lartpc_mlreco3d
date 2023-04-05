from mlreco.utils import CSVData
import os
import numpy as np
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels

from functools import wraps
from pprint import pprint

def post_processing(data_capture, result_capture):
    """
    Decorator for common post-processing boilerplate.

    <post_processing> functions take information in data and result and
    modifies the result dictionary (output of full chain) in-place, usually
    adding a new key, value pair corresponding to some reconstructed quantity.
    ----------
    data_capture: list of string
        List of data keys needed. 
    result_capture: list of string
        List of result keys needed. 
    """
    def decorator(func):
        # This mapping is hardcoded for now...
        @wraps(func)
        def wrapper(data_dict, result_dict, **kwargs):

            # TODO: Handle unwrap/non-unwrap

            out = func(data_dict, result_dict, **kwargs)

            return out
        
        wrapper._data_capture = data_capture
        wrapper._result_capture = result_capture

        return wrapper
    return decorator