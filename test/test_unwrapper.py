from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pytest

@pytest.mark.parametrize("dim", [2,3])
def test_unwrap_scn(dim, num_device=4, minibatch_size=8):

    from mlreco.utils import unwrap_3d_scn
    from mlreco.utils import unwrap_2d_scn

    unwrapper = {2:unwrap_2d_scn,3:unwrap_3d_scn}
    
    data_blob = {'x':[]}

    for j in range(num_device):
        batch_ids = np.arange(num_device*minibatch_size+j*minibatch_size,num_device*minibatch_size+(j+1)*minibatch_size)
        minibatch = np.zeros(shape=(minibatch_size,dim+1))
        minibatch[:,dim] = batch_ids
        data_blob['x'].append(minibatch)

    outputs = {'y':[]}
    for j in range(num_device):
        batch_ids = np.arange(num_device*minibatch_size+j*minibatch_size,num_device*minibatch_size+(j+1)*minibatch_size)
        minibatch = np.zeros(shape=(minibatch_size,dim+1))
        minibatch[:,dim] = batch_ids
        outputs['y'].append(minibatch)

    data_blob,outputs = unwrapper[dim](data_blob,outputs)
    assert(len(data_blob['x']) == num_device * minibatch_size)
    assert(len(data_blob['x']) == len(outputs['y']))
    assert(data_blob['x'][0].mean() == outputs['y'][0].mean())

if __name__ == '__main__':
    test_unwrap_scn(2)
    test_unwrap_scn(3)
