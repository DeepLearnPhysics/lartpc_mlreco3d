from functools import wraps

def post_processing(data_capture, result_capture,
                    data_capture_optional=[],
                    result_capture_optional=[]):
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
        @wraps(func)
        def wrapper(data_dict, result_dict, **kwargs):

            # TODO: Handle unwrap/non-unwrap

            out = func(data_dict, result_dict, **kwargs)
            return out
        
        wrapper._data_capture            = data_capture
        wrapper._result_capture          = result_capture
        wrapper._data_capture_optional   = data_capture_optional
        wrapper._result_capture_optional = result_capture_optional

        return wrapper
    return decorator
