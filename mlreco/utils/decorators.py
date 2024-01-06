import numpy as np
import numba as nb
import torch
import inspect
from time import time
from functools import wraps


def timing(fn):
    '''
    Function which wraps any function and times it.

    Parameters
    ----------
    fn : callable
        Function to time

    Returns
    -------
    callable
        Timed function
    '''
    @wraps(fn)
    def wrap(*args, **kwargs):
        ts = time()
        result = fn(*args, **kwargs)
        te = time()
        print('func:%r args:[%r, %r] took: %2.f sec' % \
          (fn.__name__, args, kwargs, te-ts))
        return result
    return wrap


def inherit_docstring(parent):
    '''
    Inherits docstring attributes of a parent class.
    Only handles numpy-style docstrings.

    Parameters
    ----------
    parent : object
        Parent class to inherit attributes from

    Returns
    -------
    callable
        Class with updated docstring
    '''
    def inherit(obj):
        tab       = '    '
        underline = '----'
        header    = f'Attributes\n{tab}----------\n'

        # If there is no attribute or method yet, add the header
        if not header in obj.__doc__:
            obj.__doc__ += f'\n\n{tab}{header}'

        # Get the parent attribute docstring block
        docstr = parent.__doc__
        substr = docstr.split(header)[-1].rstrip() + '\n'
        if len(substr.split(underline)) > 1:
            substr = ''.join(substr.split(underline)[0].split('\n')[:-1]).rstrip()

        # Append it to the relevant block
        split_doc   = obj.__doc__.split(header)
        obj.__doc__ = split_doc[0] + header + substr + split_doc[1]

        return obj

    return inherit


def numbafy(cast_args=[], list_args=[], keep_torch=False, ref_arg=None):
    '''
    Function which wraps a *numba* function with some checks on the input
    to make the relevant conversions to numpy where necessary.

    Parameters
    ----------
    cast_args : list(str), optional
        List of arguments to be cast to numpy
    list_args : list(str), optional
        List of arguments which need to be cast to a numba typed list
    keep_torch : bool, default False
        Make the output a torch object, if the reference argument is one
    ref_arg : str, optional
        Reference argument used to assign a type and device to the torch output

    Returns
    -------
    callable
        Wrapped function which ensures input type compatibility with numba
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
