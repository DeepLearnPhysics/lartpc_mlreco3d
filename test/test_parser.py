from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pytest


@pytest.fixture(params=[1, 2])
def event_tensor3d(request):
    """
    This fixture generates a list of larcv::EventSparseTensor3D, the event
    count is given by request.param.
    """
    from larcv import larcv
    import random
    meta = larcv.Voxel3DMeta()
    xmin, ymin, zmin = random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-500, 500)
    meta.set(xmin, ymin, zmin,
             xmin+230.4, ymin+230.4, zmin+230.4,
             768, 768, 768)

    event_list = []
    for _ in range(request.param):
        data = np.zeros(shape=(10, 4), dtype=np.float32)
        voxel_set = larcv.as_tensor3d(data, meta)

        event = larcv.EventSparseTensor3D()
        event.set(voxel_set, meta)
        event_list.append(event)
    return event_list


# @pytest.mark.parametrize('event_tensor3d', [1, 2], indirect=True)
def test_parse_sparse3d_scn(event_tensor3d):
    from mlreco.iotools.parsers import parse_sparse3d_scn
    np_voxels, np_data = parse_sparse3d_scn(event_tensor3d)

    assert np_voxels.shape[1] == 3
    assert np_data.shape[1] == len(event_tensor3d)
    assert np_voxels.shape[0] == np_data.shape[0]
