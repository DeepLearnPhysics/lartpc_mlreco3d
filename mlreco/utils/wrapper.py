import numpy as np
import numba as nb
import torch
import inspect
from functools import wraps


def numba_wrapper(cast_args=[], list_args=[], keep_torch=False, ref_arg=None):
    '''
    Function which wraps a *numba* function with some checks on the input
    to make the relevant conversions to numpy where necessary.

    Args:
        cast_args ([str]): List of arguments to be cast to numpy
        list_args ([str]): List of arguments which need to be cast to a numba typed list
        keep_torch (bool): Make the output a torch object, if the reference argument is one
        ref_arg (str)    : Reference argument used to assign a type and device to the torch output
    Returns:
        Function
    '''
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            # Convert the positional arguments in args into key:value pairs in kwargs
            keys = list(inspect.signature(fn).parameters.keys())
            for i, val in enumerate(args):
                kwargs[keys[i]] = val

            # Extract the default values for the remaining parameters
            for key, val in inspect.signature(fn).parameters.items():
                if key not in kwargs and val.default != inspect.Parameter.empty:
                    kwargs[key] = val.default

            # If a torch output is request, register the input dtype and device location
            if keep_torch:
                assert ref_arg in kwargs
                dtype, device = None, None
                if isinstance(kwargs[ref_arg], torch.Tensor):
                    dtype = kwargs[ref_arg].dtype
                    device = kwargs[ref_arg].device

            # If the cast data is not a numpy array, cast it
            for arg in cast_args:
                assert arg in kwargs
                if not isinstance(kwargs[arg], np.ndarray):
                    assert isinstance(kwargs[arg], torch.Tensor)
                    kwargs[arg] = kwargs[arg].detach().cpu().numpy() # For now cast to CPU only

            # If there is a reflected list in the input, type it
            for arg in list_args:
                assert arg in kwargs
                kwargs[arg] = nb.typed.List(kwargs[arg])

            # Get the output
            ret = fn(**kwargs)
            if keep_torch and dtype:
                if isinstance(ret, np.ndarray):
                    ret = torch.tensor(ret, dtype=dtype, device=device)
                elif isinstance(ret, list):
                    ret = [torch.tensor(r, dtype=dtype, device=device) for r in ret]
                else:
                    raise TypeError('Return type not recognized, cannot cast to torch')
            return ret
        return inner
    return outer
