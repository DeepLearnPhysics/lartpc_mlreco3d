import numpy as np
from functools import partial
from collections import defaultdict, OrderedDict

class PostProcessor:

    def __init__(self, data, result, debug=True):
        self._funcs = defaultdict(list)
        self._num_batches = len(data['index'])
        self.data = data
        self.result = result
        self.debug = debug

    def register_function(self, f, priority, processor_cfg={}):
        data_capture, result_capture = f._data_capture, f._result_capture
        result_capture_optional      = f._result_capture_optional
        pf = partial(f, **processor_cfg)
        pf._data_capture            = data_capture
        pf._result_capture          = result_capture
        pf._result_capture_optional = result_capture_optional
        self._funcs[priority].append(pf)

    def process_event(self, image_id, f_list):

        image_dict = {}
        
        for f in f_list:
            data_dict, result_dict = {}, {}
            for data_key in f._data_capture:
                data_dict[data_key] = self.data[data_key][image_id]
            for result_key in f._result_capture:
                result_dict[result_key] = self.result[result_key][image_id]
            for result_key in f._result_capture_optional:
                if result_key in self.result:
                    result_dict[result_key] = self.result[result_key][image_id]
            update_dict = f(data_dict, result_dict)
            for key, val in update_dict.items():
                if key in image_dict:
                    msg = 'Output {} in post-processing function {},'\
                         ' caused a dictionary key conflict. You may '\
                         'want to change the output dict key for that function.'
                    raise ValueError(msg)
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
                if key in self.result:
                    msg = "Post processing script output key {} "\
                    "is already in result_dict, you may want"\
                    "to rename it.".format(key)
                    raise RuntimeError(msg)
                else:
                    self.result[key] = val


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)
