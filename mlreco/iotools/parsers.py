from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from larcv import larcv
from mlreco.utils.ppn import get_ppn_info
from mlreco.utils.gnn.primary import get_em_primary_info
from mlreco.utils.dbscan import dbscan_types, dbscan_groups
from mlreco.utils.groups import get_group_types, filter_duplicate_voxels, filter_nonimg_voxels


def parse_sparse3d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Returns the data in format to pass to SCN
    Args:
        array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,C) - pixel values/channels
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d(data):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor3D object
    Args:
        array of larcv::EventSparseTensor3D (one per channel)
    Return:
        a numpy array with the shape (N,3+C) where 3+C represents
        (x,y,z) coordinate and C stored pixel values (channels).
    """
    meta = None
    output = []
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
            output.append(np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_values = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_values)
        output.append(np_values)
    return np.concatenate(output, axis=-1)


def parse_tensor3d(data):
    """
    A function to retrieve larcv::EventSparseTensor3D as a dense numpy array
    Args:
        array of larcv::EventSparseTensor3D
    Return:
        a numpy array of a dense 3d tensor object, last dimension = channels
    """
    np_data = []
    meta = None
    for event_tensor3d in data:
        if meta is None:
            meta = event_tensor3d.meta()
        else:
            assert meta == event_tensor3d.meta()
        np_data.append(np.array(larcv.as_ndarray(event_tensor3d)))
    return np.stack(np_data, axis=-1)


def parse_particle_points(data):
    """
    A function to retrieve particles ground truth points tensor, includes
    spatial coordinates and point type.
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, 1) where 1 represents class of
        the ground truth point.
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    if part_info.shape[0] > 0:
        return part_info[:, :3], part_info[:, 3][:, None]
    else:
        return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 1), dtype=np.float32)


def parse_particle_infos(data):
    """
    A function to retrieve particles ground truth points tensor, includes
    spatial coordinates, point type and other infos (creation energy, pdg, etc)
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, C) where C represents class of
        the ground truth point + other infos.
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    if part_info.shape[0] > 0:
        return part_info[:, :3], part_info[:, 3:]
    else:
        return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 7), dtype=np.float32)


def parse_em_primaries(data):
    """
    A function to retrieve primary ionization points from grond truth
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, 1) containing group id for the primary
    """
    particles_v = data[1].as_vector()
    part_info = get_em_primary_info(particles_v, data[0].meta(), min_voxel_count=30, min_energy_deposit=5)
    if part_info.shape[0] > 0:
        return part_info[:, :-1], part_info[:, -1][:, None]
    else:
        return np.empty(shape=(0, 6), dtype=np.int32), np.empty(shape=(0, 1), dtype=np.float32)


def parse_dbscan(data):
    """
    A function to create dbscan tensor
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,1) - dbscan cluster. -1 if not assigned
    """
    np_voxels, np_types = parse_sparse3d_scn(data)
    # now run dbscan on data
    clusts = dbscan_types(np_voxels, np_types)
    # start with no clusters assigned.
    np_types.fill(-1)
    for i, c in enumerate(clusts):
        np_types[c] = i
    return np_voxels, np_types


def parse_dbscan_groups(data):
    """
    A function to create dbscan tensor
    Args:
        length 2 array of larcv::EventClusterVoxel3 and larcv::EventParticle
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,1) - dbscan cluster. -1 if not assigned
    """
    np_voxels, np_groups = parse_cluster3d(data)
    # get the particle types for each group
    particles_v = data[1].as_vector()
    part_types = get_group_types(particles_v, data[0].meta())
    # now run dbscan on data
    num_groups = data[0].as_vector().size()
    clusts = dbscan_groups(np_voxels, np_groups, part_types)
    # start with no clusters assigned.
    np_groups.fill(-1)
    for i, c in enumerate(clusts):
        np_groups[c] = i
    return np_voxels, np_groups


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


def parse_cluster3d_clean(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)

    Args:
        length 2 array of larcv::EventClusterVoxel3D and larcv::EventSparseTensor3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,1) where 1 is cluster id
    """
    grp_voxels, grp_data = parse_cluster3d([data[0]])
    img_voxels, img_data = parse_sparse3d_scn([data[1]])

    # step 1: lexicographically sort group data
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm,:]
    grp_data = grp_data[perm]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels(grp_voxels, usebatch=False)
    inds1 = np.where(sel1)[0]
    grp_voxels = grp_voxels[inds1,:]
    grp_data = grp_data[inds1]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(grp_voxels, img_voxels, usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2,:]
    grp_data = grp_data[inds2]

    return grp_voxels, grp_data
