from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pytest

@pytest.mark.parametrize("dim", [2,3])
def test_unwrap_scn(dim, batch_size=64, num_device=4, minibatch_size=8):

    from mlreco.utils import unwrap_3d_scn
    from mlreco.utils import unwrap_2d_scn

    unwrapper = {2:unwrap_2d_scn,3:unwrap_3d_scn}
    
    data_blob = {'x':[]}

    num_forwards = int(batch_size / (num_device * minibatch_size))
    for i in range(num_forwards):
        data = []
        for j in range(num_device):
            batch_ids = np.arange(i*num_device*minibatch_size+j*minibatch_size,i*num_device*minibatch_size+(j+1)*minibatch_size)
            minibatch = np.zeros(shape=(minibatch_size,dim+1))
            minibatch[:,dim] = batch_ids
            data.append(minibatch)
        data_blob['x'].append(data)

    outputs = {'y':[]}
    for i in range(num_forwards):
        for j in range(num_device):
            batch_ids = np.arange(i*num_device*minibatch_size+j*minibatch_size,i*num_device*minibatch_size+(j+1)*minibatch_size)
            minibatch = np.zeros(shape=(minibatch_size,dim+1))
            minibatch[:,dim] = batch_ids
            outputs['y'].append(minibatch)

    parsed = unwrapper[dim](data_blob,outputs)
    assert(len(parsed) == 2)
    assert(len(parsed[0]) == len(parsed[1]))
    assert(len(parsed[0]) == batch_size)
    assert(parsed[0][0]['x'].mean() == parsed[1][0]['y'].mean())
    assert(parsed[0][batch_size-1]['x'].mean() == parsed[1][batch_size-1]['y'].mean())

if __name__ == '__main__':
    test_unwrap_scn(2)
    test_unwrap_scn(3)
