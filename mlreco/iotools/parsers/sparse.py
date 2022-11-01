import numpy as np
from larcv import larcv


def parse_sparse2d(sparse_event_list):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor2D object

    Returns the data in format to pass to SCN

    .. code-block:: yaml

        schema:
          input_data:
            parser: parse_sparse2d
            args:
              sparse_event_list:
                - sparse2d_pcluster_0 (, 0)
                - sparse2d_pcluster_1 (, 1)
                - ...

    Configuration
    -------------
    sparse_event_list: list of larcv::EventSparseTensor2D
        Optionally, give an array of (larcv::EventSparseTensor2D, int) for projection id

    Returns
    -------
    voxels: np.ndarray(int32)
        Coordinates with shape (N,2)
    data: np.ndarray(float32)
        Pixel values/channels with shape (N,C)
    """
    meta = None
    output = []
    np_voxels = None
    for sparse_event in sparse_event_list:
        projection_id = 0  # default
        if isinstance(sparse_event, tuple):
            projection_id = sparse_event[1]
            sparse_event = sparse_event[0]

        tensor = sparse_event.sparse_tensor_2d(projection_id)
        num_point = tensor.as_vector().size()

        if meta is None:

            meta = tensor.meta()
            np_voxels = np.empty(shape=(num_point, 2), dtype=np.int32)
            larcv.fill_2d_voxels(tensor, np_voxels)

        # else:
        #     assert meta == tensor.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_2d_pcloud(tensor, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d(sparse_event_list):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object

    Returns the data in format to pass to DataLoader

    .. code-block:: yaml

        schema:
          input_data:
            parser: parse_sparse3d
            args:
              sparse_event_list:
                - sparse3d_pcluster_0
                - sparse3d_pcluster_1
                - ...

    Configuration
    -------------
    sparse_event_list: list of larcv::EventSparseTensor3D
        Can be repeated to load more features (one per feature).

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,C)
        Pixel values/channels, as many channels as specified larcv::EventSparseTensor3D.
    """
    meta = None
    output = []
    np_voxels = None
    for sparse_event in sparse_event_list:
        num_point = sparse_event.as_vector().size()
        if meta is None:
            meta = sparse_event.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(sparse_event, np_voxels)
        else:
            assert meta == sparse_event.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(sparse_event, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d_ghost(sparse_event_semantics):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object

    Converts the sematic class to a ghost vs non-ghost label.

    .. code-block:: yaml

        schema:
          ghost_label:
            parser: parse_sparse3d
            args:
              sparse_event_semantics: sparse3d_semantics

    Configuration
    -------------
    sparse_event_semantics: larcv::EventSparseTensor3D

    Returns
    -------
    np.ndarray
        a numpy array with the shape (N,3+1) where 3+1 represents
        (x,y,z) coordinate and 1 stored ghost labels (channels).
    """
    np_voxels, np_data = parse_sparse3d([sparse_event_semantics])
    return np_voxels, (np_data==5).astype(np.float32)


def parse_sparse3d_charge_rescaled(sparse_event_list):
    # Produces sparse3d_reco_rescaled on the fly on datasets that do not have it
    np_voxels, output = parse_sparse3d(sparse_event_list)

    deghost      = output[:, -1] < 5
    hit_charges  = output[deghost,  :3]
    hit_ids      = output[deghost, 3:6]
    pmask        = hit_ids > -1

    _, inverse, counts = np.unique(hit_ids, return_inverse=True, return_counts=True)
    multiplicity = counts[inverse].reshape(-1,3)
    charges = np.sum((hit_charges*pmask)/multiplicity, axis=1)/np.sum(pmask, axis=1)

    return np_voxels[deghost], charges.reshape(-1,1)


def parse_sparse2d_scn(sparse_event_list):
    from warnings import warn
    warn("Deprecated: parse_sparse2d_scn deprecated, use parse_sparse2d instead", DeprecationWarning)
    return parse_sparse2d(sparse_event_list)


def parse_sparse3d_scn(sparse_event_list):
    from warnings import warn
    warn("Deprecated: parse_sparse3d_scn deprecated, use parse_sparse3d instead", DeprecationWarning)
    return parse_sparse3d(sparse_event_list)
