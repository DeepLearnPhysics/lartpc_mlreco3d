from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pytest


#######################################################################
# Fixtures for the tests
# The idea is to generate dummy data products that can be used to run
# the parsers and check basic assumptions.
#######################################################################


@pytest.fixture(params=[1, 2])
def event_tensor3d(request, N):
    """
    This fixture generates a list of larcv::EventSparseTensor3D.

    Parameters: event count and image size
    """
    from larcv import larcv
    import random
    meta = larcv.Voxel3DMeta()
    xmin, ymin, zmin = random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-500, 500)
    meta.set(xmin, ymin, zmin,
             xmin+N*0.3, ymin+N*0.3, zmin+N*0.3,
             N, N, N)

    event_list = []
    for _ in range(request.param):
        data = np.concatenate([
            np.random.random((10, 1)) * N * 0.3 + xmin,
            np.random.random((10, 1)) * N * 0.3 + ymin,
            np.random.random((10, 1)) * N * 0.3 + zmin,
            np.random.random((10, 1))
        ], axis=1).astype(np.float32)
        voxel_set = larcv.as_tensor3d(data, meta, -0.01)

        event = larcv.EventSparseTensor3D()
        event.set(voxel_set, meta)
        event_list.append(event)
    return event_list


@pytest.fixture(params=[7])
def event_particles(request):
    """
    This fixture generates one larcv::EventParticle.
    It attempts to generate some particles that will pass the filters of
    `get_ppn_info`.

    Parameters: number of particles.
    """
    from larcv import larcv
    num_particles = request.param
    particles = larcv.EventParticle()
    for idx in range(num_particles):
        p = larcv.Particle()
        p.shape(0)
        p.energy_deposit(np.random.random()*100)
        p.num_voxels(10)
        particle_type = np.random.randint(low=0, high=5)
        if particle_type == 0:
            p.pdg_code(2212)
        elif particle_type == 1:
            p.pdg_code(int(np.random.choice([13, -13])))
        elif particle_type == 2:
            p.pdg_code(int(np.random.choice([22, 11])))
            p.creation_process(np.random.choice(["primary", "nCapture", "conv"]))
        elif particle_type == 3:
            p.parent_pdg_code(13)
            p.creation_process(np.random.choice(["muIoni", "hIoni"]))
            p.pdg_code(11)
        elif particle_type == 4:
            p.pdg_code(11)
            p.creation_process(np.random.choice(["muMinusCaptureAtRest", "muPlusCaptureAtRest", "Decay"]))
        p.group_id(idx)
        particles.append(p)

    return particles


@pytest.fixture(params=[4])
def event_cluster3d(request, N):
    """
    This fixture generates a larcv::EventClusterVoxel3D.

    Parameters: number of clusters and image size
    """
    from larcv import larcv
    import random
    meta = larcv.Voxel3DMeta()
    xmin, ymin, zmin = random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-500, 500)
    meta.set(xmin, ymin, zmin,
             xmin+N, ymin+N, zmin+N,
             N, N, N)
    print(meta.size_voxel_x())

    num_clusters = request.param
    event = larcv.EventClusterVoxel3D()
    event.resize(num_clusters)
    for i in range(num_clusters):
        num_voxels = np.random.randint(low=5, high=20)
        # data = np.concatenate([
        #     np.random.random((num_voxels, 1)) * N + xmin,
        #     np.random.random((num_voxels, 1)) * N + ymin,
        #     np.random.random((num_voxels, 1)) * N + zmin,
        #     np.random.random((num_voxels, 1))
        # ], axis=1).astype(np.float32)
        data = np.concatenate([
            np.random.randint(0, high=N-1, size=(num_voxels, 1)) + xmin,
            np.random.randint(0, high=N-1, size=(num_voxels, 1)) + ymin,
            np.random.randint(0, high=N-1, size=(num_voxels, 1)) + zmin,
            np.random.random((num_voxels, 1))
        ], axis=1).astype(np.float32)
        voxel_set = larcv.as_tensor3d(data, meta, -0.01)
        # Write cluster i
        v = event.writeable_voxel_set(i)
        for vox in voxel_set.as_vector():
            if meta.invalid_voxel_id() != vox.id():
                v.insert(vox)
    event.meta(meta)
    return event


# def event_tensor3d_from_cluster3d(ev):
#     """
#     ev is an EventClusterVoxel3D
#     """
#     from larcv import larcv
#     meta = ev.meta()
#     event = larcv.EventSparseTensor3D()
#     data = []
#     print(meta.size_voxel_x())
#     for voxel_set in ev.as_vector():
#         event.emplace(voxel_set, meta)
#     #     num_points = voxel_set.as_vector().size()
#     #     if num_points > 0:
#     #         x = np.empty(shape=(num_points,), dtype=np.int32)
#     #         y = np.empty(shape=(num_points,), dtype=np.int32)
#     #         z = np.empty(shape=(num_points,), dtype=np.int32)
#     #         value = np.empty(shape=(num_points,), dtype=np.float32)
#     #         larcv.as_flat_arrays(voxel_set,meta,x, y, z, value)
#     #         data.append(np.stack([x*meta.size_voxel_x()+meta.min_x(),
#     #                             y*meta.size_voxel_y()+meta.min_y() ,
#     #                             z*meta.size_voxel_z()+meta.min_z(), value], axis=1))
#     # data = np.concatenate(data, axis=0).astype(np.float32)
#     # print("tensor", data)
#     # voxel_set = larcv.as_tensor3d(data, meta, -0.01)
#     # event.set(voxel_set, meta)
#     return event

#######################################################################
# Parsers tests
#######################################################################

def test_parse_sparse3d_scn(event_tensor3d):
    from mlreco.iotools.parsers import parse_sparse3d_scn
    output = parse_sparse3d_scn(event_tensor3d)
    assert len(output) == 2

    np_voxels, np_data = output
    assert len(np_voxels.shape) == 2
    assert len(np_data.shape) == 2
    assert np_voxels.shape[1] == 3
    assert np_data.shape[1] == len(event_tensor3d)
    assert np_voxels.shape[0] == np_data.shape[0]
    assert np_voxels.shape[0] > 0


def test_parse_sparse3d(event_tensor3d):
    from mlreco.iotools.parsers import parse_sparse3d
    output = parse_sparse3d(event_tensor3d)
    assert len(output.shape) == 2
    assert output.shape[1] == 3 + len(event_tensor3d)
    assert output.shape[0] > 0


# event_tensor3d fixture also depends on the fixture N.
# But, the value of N will be in agreement with N used to run event_tensor3d.
def test_parse_tensor3d(event_tensor3d, N):
    from mlreco.iotools.parsers import parse_tensor3d
    output = parse_tensor3d(event_tensor3d)
    assert len(output.shape) == 4
    assert output.shape[0] == N
    assert output.shape[0] == output.shape[1]
    assert output.shape[0] == output.shape[2]
    assert output.shape[3] == len(event_tensor3d)


@pytest.mark.parametrize("event_tensor3d", [1], indirect=True)
def test_parse_particle_points(event_tensor3d, event_particles):
    from mlreco.iotools.parsers import parse_particle_points
    output = parse_particle_points((event_tensor3d[0], event_particles))
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 2
    assert output[0].shape[0] == output[1].shape[0]


@pytest.mark.parametrize("event_tensor3d", [1], indirect=True)
def test_parse_dbscan(event_tensor3d):
    from mlreco.iotools.parsers import parse_dbscan
    output = parse_dbscan(event_tensor3d)
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 1
    assert output[0].shape[0] == output[1].shape[0]


def test_parse_cluster3d(event_cluster3d):
    from mlreco.iotools.parsers import parse_cluster3d
    output = parse_cluster3d([event_cluster3d])
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 2
    assert output[0].shape[0] == output[1].shape[0]


# TODO in larcv add a function to generate EventSparseTensor3D from EventClusterVoxel3D
# Otherwise rounding errors when converting through numpy cause voxels to disappear in parse_cluster3d_full

# Make sure there are the same number of clusters in
# event_cluster3d and particles in event_particles
#@pytest.mark.parametrize("event_tensor3d", [1], indirect=True)
# @pytest.mark.parametrize("event_particles", [5], indirect=True)
# @pytest.mark.parametrize("event_cluster3d_and_event_tensor3d", [5], indirect=True)
# def test_parse_cluster3d_clean(event_cluster3d_and_event_tensor3d, event_particles):
#     from mlreco.iotools.parsers import parse_cluster3d_clean
#     event_tensor3d = event_tensor3d_from_cluster3d(event_cluster3d)
#     output = parse_cluster3d_clean([event_cluster3d, event_particles, event_tensor3d])
#     assert len(output) == 2
#     assert len(output[0].shape) == 2
#     assert len(output[1].shape) == 2
#     assert output[0].shape[1] == 3
#     assert output[1].shape[1] == 6
#     assert output[0].shape[0] == output[1].shape[0]
def test_parse_cluster3d_clean():
    pass

# TODO
def test_parse_particle_singlep_pdg():
    pass

def test_parse_particle_singlep_einit():
    pass

def test_parse_sparse2d_meta():
    pass

def test_parse_sparse2d_scn():
    pass

def test_parse_semantics():
    pass

def test_parse_weights():
    pass

def test_parse_particle_asis():
    pass

@pytest.mark.parametrize("event_cluster3d", [3], indirect=True)
@pytest.mark.parametrize("event_particles", [3], indirect=True)
def test_parse_particle_coords(event_cluster3d, event_particles):
    from mlreco.iotools.parsers import parse_particle_coords
    output = parse_particle_coords((event_particles, event_cluster3d))
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 4
    assert output[0].shape[0] == output[1].shape[0]

def test_parse_particle_graph():
    pass

def test_parse_particle_graph_corrected():
    pass

def test_parse_cluster2d():
    pass

def test_parse_cluster3d_full():
    pass

@pytest.mark.parametrize("event_cluster3d", [3], indirect=True)
@pytest.mark.parametrize("event_particles", [3], indirect=True)
def test_parse_cluster3d_types(event_cluster3d, event_particles):
    from mlreco.iotools.parsers import parse_cluster3d_types
    output = parse_cluster3d_types((event_cluster3d, event_particles))
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 4
    assert output[0].shape[0] == output[1].shape[0]

@pytest.mark.parametrize("event_cluster3d", [3], indirect=True)
@pytest.mark.parametrize("event_particles", [3], indirect=True)
def test_parse_cluster3d_kinematics(event_cluster3d, event_particles):
    from mlreco.iotools.parsers import parse_cluster3d_kinematics
    output = parse_cluster3d_kinematics((event_cluster3d, event_particles))
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 5
    assert output[0].shape[0] == output[1].shape[0]

def test_parse_cluster3d_kinematics_clean():
    pass

def test_parse_cluster3d_full_fragment():
    pass

def test_parse_cluster3d_fragment():
    pass

# Make sure input only has 1 feature
@pytest.mark.parametrize("event_tensor3d", [1], indirect=True)
def test_parse_sparse3d_fragment(event_tensor3d):
    from mlreco.iotools.parsers import parse_sparse3d_fragment
    output = parse_sparse3d_fragment(event_tensor3d)
    assert len(output) == 2
    assert len(output[0].shape) == 2
    assert len(output[1].shape) == 2
    assert output[0].shape[1] == 3
    assert output[1].shape[1] == 1
    assert output[0].shape[0] == output[1].shape[0]

def test_clean_data():
    pass

def test_parse_cluster3d_clean_full():
    pass

def test_parse_sparse3d_clean():
    pass

def test_parse_cluster3d_scales():
    pass

# Make sure input only has 2 features
@pytest.mark.parametrize("event_tensor3d", [2], indirect=True)
def test_parse_sparse3d_scn_scales(event_tensor3d):
    from mlreco.iotools.parsers import parse_sparse3d_scn_scales
    output = parse_sparse3d_scn_scales(event_tensor3d)
    spatial_size = event_tensor3d[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size))-1)
    assert len(output) == max_depth
    for d, x in enumerate(output):
        assert len(x) == 2
        assert isinstance(x, tuple)

        assert len(x[0].shape) == 2
        assert len(x[1].shape) == 2
        assert x[0].shape[1] == 3
        assert x[1].shape[1] == 2
        assert x[0].shape[0] == x[1].shape[0]

        assert x[0].min() >= 0
        assert x[0].max() < spatial_size/2**d
