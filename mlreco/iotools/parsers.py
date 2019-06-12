from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from larcv import larcv
from mlreco.utils.ppn import get_ppn_info


def parse_sparse3d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Returns the data in format to pass to SCN
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,1) - pixel value
    """
    event_tensor3d = data[0]
    num_point = event_tensor3d.as_vector().size()
    np_voxels = np.empty(shape=(num_point,3),dtype=np.int32)
    np_data   = np.empty(shape=(num_point,1),dtype=np.float32)
    larcv.fill_3d_voxels(event_tensor3d, np_voxels)
    larcv.fill_3d_pcloud(event_tensor3d, np_data  )
    return np_voxels, np_data


def parse_sparse3d(data):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor3D object
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        a numpy array with the shape (N,4) where 4=3+1 represents (x,y,z) coordinate and stored pixel value.
    """
    event_tensor3d = data[0]
    num_point = event_tensor3d.as_vector().size()
    np_data   = np.empty(shape=(num_point,4),dtype=np.float32)
    larcv.fill_3d_pcloud(event_tensor3d, np_data)
    return np_data


def parse_tensor3d(data):
    """
    A function to retrieve larcv::EventSparseTensor3D as a numpy array
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        a numpy array of a dense 3d tensor object
    """
    from larcv import larcv
    event_tensor3d = data[0]
    return np.array(larcv.as_ndarray(event_tensor3d))


def parse_particles(data):
    """
    A function to retrieve particles ground truth points tensor
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, 1) where 1 represents class of
        the ground truth point (track vs shower).
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    if part_info.shape[0] > 0:
        return part_info[:, :-1], part_info[:, -1][:, None]
    else:
        return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 1), dtype=np.float32)


def parse_cluster3d(data):
    """
    A function to retrieve clusters tensor
    Args:
        length 1 array of larcv::EventClusterVoxel3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,1) where 1 is cluster id
    """
    cluster_event = data[0]
    from larcv import larcv
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_data = [], []
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.int32)
            larcv.as_flat_arrays(cluster_event.as_vector()[i],
                                 cluster_event.meta(),
                                 x, y, z, value)
            value = np.full(shape=(cluster.as_vector().size(), 1),
                            fill_value=i, dtype=np.int32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_data.append(value)
    np_voxels = np.concatenate(clusters_voxels, axis=0)
    np_data = np.concatenate(clusters_data, axis=0)
    return np_voxels, np_data
