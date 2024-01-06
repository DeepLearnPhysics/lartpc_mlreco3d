import numpy as np
from larcv import larcv

from mlreco.utils.globals import GHOST_SHP


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


def parse_sparse3d(sparse_event_list, features=None, hit_keys=[], nhits_idx=None):
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
    features: int, optional
        Default is None (ignored). If a positive integer is specified,
        the sparse_event_list will be split in equal lists of length
        `features`. Each list will be concatenated along the feature
        dimension separately. Then all lists are concatenated along the
        first dimension (voxels). For example, this lets you work with
        distinct detector volumes whose input data is stored in separate
        TTrees.`features` is required to be a divider of the `sparse_event_list`
        length.
    hit_keys: list of int, optional
        Indices among the input features of the _hit_key_ TTrees that can be
        used to infer the _nhits_ quantity (doublet vs triplet point).
    nhits_idx: int, optional
        Index among the input features where the _nhits_ feature (doublet vs triplet)
        should be inserted.

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,C)
        Pixel values/channels, as many channels as specified larcv::EventSparseTensor3D.
    """
    split_sparse_event_list = [sparse_event_list]
    if features is not None and features > 0:
        if len(sparse_event_list) % features > 0:
            raise Exception("features number in parse_sparse3d should be a divider of the sparse_event_list length.")
        split_sparse_event_list = np.split(np.array(sparse_event_list), len(sparse_event_list) / features)

    voxels, features = [], []
    features_count = None
    compute_nhits = len(hit_keys) > 0
    if compute_nhits and nhits_idx is None:
        raise Exception("nhits_idx needs to be specified if you want to compute the _nhits_ feature.")

    for sparse_event_list in split_sparse_event_list:
        if features_count is None:
            features_count = len(sparse_event_list)
        assert len(sparse_event_list) == features_count

        meta = None
        output = []
        np_voxels = None
        hit_key_array = []
        for idx, sparse_event in enumerate(sparse_event_list):
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

            if compute_nhits:
                if idx in hit_keys:
                    hit_key_array.append(np_data)

        voxels.append(np_voxels)
        features_array = np.concatenate(output, axis=-1)

        if compute_nhits:
            hit_key_array = np.concatenate(hit_key_array, axis=-1)
            doublets = (hit_key_array < 0).any(axis=1)
            nhits = 3. * np.ones((np_voxels.shape[0],), dtype=np.float32)
            nhits[doublets] = 2.
            if nhits_idx < 0 or nhits_idx > features_array.shape[1]:
                raise Exception("nhits_idx is out of range")
            features_array = np.concatenate([features_array[..., :nhits_idx], nhits[:, None], features_array[..., nhits_idx:]], axis=-1)

        features.append(features_array)

    return np.concatenate(voxels, axis=0), np.concatenate(features, axis=0)


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


def parse_sparse3d_charge_rescaled(sparse_event_list, collection_only=False):
    # Produces sparse3d_reco_rescaled on the fly on datasets that do not have it
    from mlreco.utils.ghost import compute_rescaled_charge
    np_voxels, output = parse_sparse3d(sparse_event_list)

    deghost_mask = np.where(output[:, -1] < GHOST_SHP)[0]
    charges = compute_rescaled_charge(output[:, :-1], deghost_mask,
            last_index=0, collection_only=collection_only, use_batch=False)

    return np_voxels[deghost_mask], charges.reshape(-1,1)
