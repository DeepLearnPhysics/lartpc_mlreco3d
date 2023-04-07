from functools import wraps

def write_to(filenames=[]):
    """
    Decorator for handling analysis tools script savefiles.

    Parameters
    ----------
    filenames: list of output filenames
    """
    def decorator(func):
        @wraps(func)
        def wrapper(data_dict, result_dict, **kwargs):

            # TODO: Handle unwrap/non-unwrap

            out = func(data_dict, result_dict, **kwargs)
            return out
        
        wrapper._filenames = filenames

        return wrapper
    return decorator