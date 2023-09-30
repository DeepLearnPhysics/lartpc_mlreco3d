import numpy as np
from functools import partial, wraps
from collections import defaultdict, OrderedDict
import warnings
import time


class PostProcessor:
    """Manager for handling post-processing scripts.
    
    """
    def __init__(self, data, result, debug=True, profile=False):
        self._funcs = defaultdict(list)
        # self._batch_funcs = defaultdict(list)
        self._num_batches = len(data['index'])
        self.data = data
        self.result = result
        self.debug = debug
        
        self._profile = defaultdict(float)
        
    def profile(self, func): 
        '''Decorator that reports the execution time. '''
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs) 
            end = time.time() 
            dt = end - start
            self._profile[func.__name__] += dt
            return result
        return wrapper

    def register_function(self, f, priority, 
                          processor_cfg={}, 
                          profile=False,
                          verbose=False):
        data_capture, result_capture = f._data_capture, f._result_capture
        data_capture_optional        = f._data_capture_optional
        result_capture_optional      = f._result_capture_optional
        pf                           = partial(f, **processor_cfg)
        pf.__name__                  = f.__name__
        pf._data_capture             = data_capture
        pf._result_capture           = result_capture
        pf._data_capture_optional    = data_capture_optional
        pf._result_capture_optional  = result_capture_optional
        if profile:
            pf = self.profile(pf)
        self._funcs[priority].append(pf)
        if verbose:
            print(f"Registered post-processor {f.__name__}")

    def process_event(self, image_id, f_list):

        image_dict = {}
        
        for f in f_list:
            data_one_event, result_one_event = {}, {}
            for data_key in f._data_capture:
                if data_key in self.data:
                    if data_key == 'meta':  # Handle special case for meta
                        data_one_event[data_key] = self.data[data_key][0]
                    else:
                        data_one_event[data_key] = self.data[data_key][image_id]
                else:
                    msg = f"Unable to find {data_key} in data dictionary while "\
                        f"running post-processor {f.__name__}."
                    warnings.warn(msg)
            for result_key in f._result_capture:
                if result_key in self.result:
                    result_one_event[result_key] = self.result[result_key][image_id]
                else:
                    msg = f"Unable to find {result_key} in result dictionary while "\
                        f"running post-processor {f.__name__}."
                    warnings.warn(msg)

            for data_key in f._data_capture_optional:
                if data_key in self.data:
                    data_one_event[data_key] = self.data[data_key][image_id]
            for result_key in f._result_capture_optional:
                if result_key in self.result:
                    result_one_event[result_key] = self.result[result_key][image_id]

            update_dict = f(data_one_event, result_one_event)
            for key, val in update_dict.items():
                if key in image_dict:
                    msg = 'Output {} in post-processing function {},'\
                         ' caused a dictionary key conflict. You may '\
                         'want to change the output dict key for that function.'
                    raise ValueError(msg.format(key, f.__name__))
                else:
                    image_dict[key] = val

        return image_dict
    
    def process_and_modify(self):
        """
        
        """
        sorted_processors = sorted([x for x in self._funcs.items()], reverse=True)
        for priority, f_list in sorted_processors:
            out_dict = defaultdict(list)
            for image_id in range(self._num_batches):
                image_dict = self.process_event(image_id, f_list)
                for key, val in image_dict.items():
                    out_dict[key].append(val)
            
            if self.debug:
                for key, val in out_dict.items():
                    assert len(out_dict[key]) == self._num_batches

            for key, val in out_dict.items():
                assert len(val) == self._num_batches
                if key in self.result:
                    msg = "Post processing script output key {} "\
                    "is already in result_dict, it will be overwritten "\
                    "unless you rename it.".format(key)
                    # print(msg)
                self.result[key] = val


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)
