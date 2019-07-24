from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pytest


@pytest.fixture(params=[(1, 1), (1, 4), (4, 1), (4, 4)])
def batch(request):
    batch_size = request.param[0]
    # channels = request.param[1]
    num_products = request.param[1]
    res = []
    for _ in range(batch_size):
        event = {}
        num_points = np.random.randint(low=0, high=100)
        for name in range(num_products):
            data = np.random.uniform(low=0.0, high=100.0, size=(num_points,))
            event[name]=data
        # Append index
        # TODO
        res.append(event)
    return res


def test_collate_sparse(batch):
    from mlreco.iotools.collates import CollateSparse
    batch_size = len(batch)
    num_products = len(batch[0])
    result = CollateSparse(batch)

    assert len(result) == num_products
