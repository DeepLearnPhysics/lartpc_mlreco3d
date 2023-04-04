import numpy as np
from functools import partial
from collections import defaultdict

class PostProcessor:

    def __init__(self, cfg, data, result, debug=True):
        self._funcs = []
        self._num_batches = cfg['iotool']['batch_size']
        self.data = data
        self.result = result
        self.debug = debug

    def register_function(self, f, processor_cfg={}):
        data_capture, result_capture = f._data_capture, f._result_capture
        pf = partial(f, **processor_cfg)
        pf._data_capture = data_capture
        pf._result_capture = result_capture
        self._funcs.append(pf)

    def process_event(self, image_id):

        image_dict = {}
        
        for f in self._funcs:
            data_dict, result_dict = {}, {}
            for data_key in f._data_capture:
                data_dict[data_key] = self.data[data_key][image_id]
            for result_key in f._result_capture:
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
    
    def process(self):

        out_dict = defaultdict(list)

        for image_id in range(self._num_batches):
            image_dict = self.process_event(image_id)
            for key, val in image_dict.items():
                out_dict[key].append(val)
        
        if self.debug:
            for key, val in out_dict.items():
                assert len(out_dict[key]) == self._num_batches

        return out_dict


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)
