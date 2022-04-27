import numpy as np
from larcv import larcv
from mlreco.utils.dbscan import dbscan_types
from mlreco.iotools.parsers.sparse import parse_sparse3d_scn


def parse_meta3d(data):
    """
    Get the meta information to translate into real world coordinates (3D).

    Each entry in a dataset is a cube, where pixel coordinates typically go
    from 0 to some integer N in each dimension. If you wish to translate
    these voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block:: yaml

        schema:
          meta:
            - parse_meta3d
            - sparse3d_pcluster

    Configuration
    ----------
    sparse3d_pcluster : larcv::EventSparseTensor3D or larcv::EventClusterVoxel3D

    Returns
    -------
    np.ndarray
        Contains in order:

        * `min_x`, `min_y`, `min_z` (real world coordinates)
        * `max_x`, `max_y`, `max_z` (real world coordinates)
        * `size_voxel_x`, `size_voxel_y`, `size_voxel_z` the size of each voxel
        in real world units
    """
    event_tensor3d = data[0]
    meta = event_tensor3d.meta()
    return [
        meta.min_x(),
        meta.min_y(),
        meta.min_z(),
        meta.max_x(),
        meta.max_y(),
        meta.max_z(),
        meta.size_voxel_x(),
        meta.size_voxel_y(),
        meta.size_voxel_z()
    ]


def parse_meta2d(data):
    """
    Get the meta information to translate into real world coordinates (2D).

    Each entry in a dataset is a cube, where pixel coordinates typically go
    from 0 to some integer N in each dimension. If you wish to translate
    these voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block:: yaml

        schema:
          meta:
            - parse_meta2d
            - sparse2d_pcluster

    Configuration
    ----------
    sparse2d_pcluster : larcv::EventSparseTensor2D or larcv::EventClusterVoxel2D

    Returns
    -------
    np.ndarray
        Contains in order:

        * `min_x`, `min_y` (real world coordinates)
        * `max_x`, `max_y` (real world coordinates)
        * `size_voxel_x`, `size_voxel_y` the size of each voxel
        in real world units

    Note
    ----
    TODO document how to specify projection id.
    """
    event_tensor2d = data[0]
    projection_id = 0  # default
    if isinstance(event_tensor2d, tuple):
        projection_id = event_tensor2d[1]
        event_tensor2d = event_tensor2d[0]

    tensor2d = event_tensor2d.sparse_tensor_2d(projection_id)
    meta = tensor2d.meta()
    return [
        meta.min_x(),
        meta.min_y(),
        meta.max_x(),
        meta.max_y(),
        meta.pixel_width(),
        meta.pixel_height()
    ]


def parse_dbscan(data):
    """
    A function to create dbscan tensor

    .. code-block:: yaml

        schema:
          meta:
            - parse_dbscan
            - sparse3d_pcluster

    Configuration
    ----------
    sparse3d_pcluster : larcv::EventSparseTensor3D

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,1)
        dbscan cluster. -1 if not assigned
    """
    np_voxels, np_types = parse_sparse3d_scn(data)
    # now run dbscan on data
    clusts = dbscan_types(np_voxels, np_types)
    # start with no clusters assigned.
    np_types.fill(-1)
    for i, c in enumerate(clusts):
        np_types[c] = i
    return np_voxels, np_types


def parse_run_info(data):
    """
    Parse run info (run, subrun, event number)

    .. code-block:: yaml

        schema:
          meta:
            - parse_run_info
            - sparse3d_pcluster

    Configuration
    ----------
    sparse3d_pcluster : larcv::EventSparseTensor3D or larcv::EventClusterVoxel3D
        data to get run info from

    Returns
    -------
    tuple
         (run, subrun, event)
    """
    return data[0].run(), data[0].subrun(), data[0].event()


def parse_tensor3d(data):
    """
    A function to retrieve larcv::EventSparseTensor3D as a dense numpy array

    .. code-block:: yaml

        schema:
          meta:
            - parse_tensor3d
            - sparse3d_pcluster

    Configuration
    ----------
    sparse3d_pcluster : larcv::EventSparseTensor3D

    Returns
    -------
    np.ndarray
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
