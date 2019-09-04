from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import pytest


@pytest.mark.parametrize("data_size,minibatch_size", [(100, 10), (100, 1)])
def test_sampler(data_size, minibatch_size):
    """
    Tests that RandomSequenceSampler runs.
    TODO: check that it runs *correctly*.
    """
    TOP_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR = os.path.dirname(TOP_DIR)
    sys.path.insert(0, TOP_DIR)

    from mlreco.iotools.samplers import RandomSequenceSampler
    import numpy as np

    s = RandomSequenceSampler(data_size, minibatch_size)
    used = np.ones(shape=[data_size], dtype=np.int32)
    used2 = np.zeros(shape=[data_size], dtype=np.int32)
    used3 = np.array([v for v in range(data_size)])
    for idx in s:
        used[idx] = 0
        used2[idx] += 1
        used3[idx] = 0
    print('...remaining:', used.sum())
    print('...max reuse:', used2.max(), 'for', used2.argmax())
    print('...average:', used3[np.where(used > 0)].mean())
    return True
