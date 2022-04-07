import numpy as np
from larcv import larcv
from mlreco.utils.groups import filter_duplicate_voxels, filter_duplicate_voxels_ref, filter_nonimg_voxels
from mlreco.iotools.parsers.sparse import parse_sparse3d_scn
from mlreco.iotools.parsers.particles import parse_particle_asis
from mlreco.iotools.parsers.clean_data import clean_data


def parse_cluster2d(data):
    """
    A function to retrieve a 2D clusters tensor

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster2d
            - cluster2d_pcluster

    Configuration
    -------------
    cluster2d_pcluster: larcv::EventClusterPixel2D

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (N,2) where 2 represents (x,y)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (N,2) where 2 is pixel value and cluster id, respectively
    """
    cluster_event = data[0].as_vector().front()
    meta = cluster_event.meta()
    num_clusters = cluster_event.size()
    clusters_voxels, clusters_features = [], []
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, value)
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=i, dtype=np.float32)
            clusters_voxels.append(np.stack([x, y], axis=1))
            clusters_features.append(np.column_stack([value, cluster_id]))
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)

    return np_voxels, np_features


def parse_cluster3d(data):
    """
    a function to retrieve a 3D clusters tensor

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d
            - cluster3d_pcluster

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (n,2) where 2 is voxel value and cluster id, respectively

    See Also
    --------
    parse_cluster3d_full
    """
    cluster_event = data[0]
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_features = [], []
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, z, value)
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=i, dtype=np.float32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_features.append(np.column_stack([value,cluster_id]))
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)
    return np_voxels, np_features


def parse_cluster3d_full(data):
    """
    a function to retrieve a 3D clusters tensor with full features list

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d_full
            - cluster3d_pcluster
            - particle_mpv

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D
    particle_mpv: larcv::EventParticle, optional
        To determine neutrino vs cosmic labels

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (n,8) where 8 is respectively

        * voxel value,
        * cluster id,
        * group id,
        * interaction id,
        * nu id,
        * particle type,
        * primary id,
        * semantic type,
    """
    cluster_event = data[0]
    particles_v = data[1].as_vector()
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_features = [], []
    particle_mpv = None
    if len(data) > 2:
        particle_mpv = data[2].as_vector()

    from mlreco.utils.groups import get_interaction_id, get_nu_id, get_particle_id, get_primary_id
    group_ids =   np.array([p.group_id() for p in particles_v])
    inter_ids =   get_interaction_id(particles_v)
    nu_ids      = get_nu_id(cluster_event, particles_v, inter_ids, particle_mpv=particle_mpv)
    pids        = get_particle_id(particles_v, nu_ids)
    primary_ids = get_primary_id(cluster_event, particles_v)

    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, z, value)
            assert i == particles_v[i].id()
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=particles_v[i].id(), dtype=np.float32)
            group_id = np.full(shape=(cluster.as_vector().size()),
                               #fill_value=particles_v[i].group_id(), dtype=np.float32)
                               fill_value=group_ids[i], dtype=np.float32)
            inter_id = np.full(shape=(cluster.as_vector().size()),
                               fill_value=inter_ids[i], dtype=np.float32)
            nu_id = np.full(shape=(cluster.as_vector().size()),
                            fill_value=nu_ids[i], dtype=np.float32)
            pdg = np.full(shape=(cluster.as_vector().size()),
                          fill_value=pids[i], dtype=np.float32)
            primary_id = np.full(shape=(cluster.as_vector().size()),
                            fill_value=primary_ids[i], dtype=np.float32)
            sem_type = np.full(shape=(cluster.as_vector().size()),
                               fill_value=particles_v[i].shape(), dtype=np.float32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_features.append(np.column_stack([value,cluster_id,group_id,inter_id,nu_id,pdg,primary_id,sem_type]))

    if len(clusters_voxels):
        np_voxels   = np.concatenate(clusters_voxels, axis=0)
        np_features = np.concatenate(clusters_features, axis=0)
    else:
        np_voxels   = np.empty(shape=(0, 3), dtype=np.float32)
        np_features = np.empty(shape=(0, 8), dtype=np.float32)

    return np_voxels, np_features


def parse_cluster3d_types(data):
    """
    a function to retrieve a 3D clusters tensor and PDG information

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d_types
            - cluster3d_pcluster
            - particle_mpv

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D
    particle_mpv: larcv::EventParticle, optional
        To determine neutrino vs cosmic labels

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (n,4) where 4 is voxel value,
        cluster id, group id, pdg, respectively

    See Also
    --------
    parse_cluster3d_full
    """
    cluster_event = data[0]
    particles_v = data[1].as_vector()
    # print(cluster_event)
    # assert False
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_features = [], []
    particle_mpv = None
    if len(data) > 2:
        particle_mpv = data[2].as_vector()

    from mlreco.utils.groups import get_interaction_id, get_nu_id
    group_ids = np.array([p.group_id() for p in particles_v])
    inter_ids = get_interaction_id(particles_v)
    nu_ids    = get_nu_id(cluster_event, particles_v, inter_ids, particle_mpv = particle_mpv)

    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, z, value)
            assert i == particles_v[i].id()
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=particles_v[i].id(), dtype=np.float32)
            group_id = np.full(shape=(cluster.as_vector().size()),
                               #fill_value=particles_v[i].group_id(), dtype=np.float32)
                               fill_value=group_ids[i], dtype=np.float32)
            t = int(particles_v[i].pdg_code())
            if t in TYPE_LABELS.keys():
                pdg = np.full(shape=(cluster.as_vector().size()),
                                fill_value=TYPE_LABELS[t], dtype=np.float32)
            else:
                pdg = np.full(shape=(cluster.as_vector().size()),
                                fill_value=-1, dtype=np.float32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_features.append(np.column_stack([value, cluster_id, group_id, pdg]))
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)
    # mask = np_features[:, 6] == np.unique(np_features[:, 6])[0]

    # print(np_features[mask][:, [0, 1, 5, 6]])
    return np_voxels, np_features


def parse_cluster3d_kinematics(data):
    """
    a function to retrieve a 3D clusters tensor with kinematics features
    (including vertex information and primary particle tagging).

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d_kinematics
            - cluster3d_pcluster
            - particle_pcluster
            - particle_mpv

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D
    particle_pcluster: larcv::EventParticle
    particle_mpv: larcv::EventParticle, optional
        To determine neutrino vs cosmic labels

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (n,9) where 9 is respectively

        * voxel value,
        * cluster id,
        * group id,
        * pdg,
        * momentum,
        * vtx_x,
        * vtx_y,
        * vtx_z,
        * is_primary

    See Also
    --------
    parse_cluster3d_full
    parse_cluster3d_kinematics_clean

    Note
    ----
    Likely to be merged with `parse_cluster3d_full` soon.
    """
    cluster_event = data[0]
    particles_v = data[1].as_vector()
    particles_v_asis = parse_particle_asis([data[1], data[0]])

    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_features = [], []
    particle_mpv = None
    if len(data) > 2:
        particle_mpv = data[2].as_vector()

    from mlreco.utils.groups import get_interaction_id, get_nu_id, get_particle_id
    group_ids = np.array([p.group_id() for p in particles_v])
    inter_ids = get_interaction_id(particles_v)
    nu_ids    = get_nu_id(cluster_event, particles_v, inter_ids, particle_mpv = particle_mpv)
    pids      = get_particle_id(particles_v, nu_ids)

    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, z, value)
            assert i == particles_v[i].id()
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=particles_v[i].id(), dtype=np.float32)
            group_id = np.full(shape=(cluster.as_vector().size()),
                               #fill_value=particles_v[i].group_id(), dtype=np.float32)
                               fill_value=group_ids[i], dtype=np.float32)
            px = particles_v[i].px()
            py = particles_v[i].py()
            pz = particles_v[i].pz()
            p = np.sqrt(px**2 + py**2 + pz**2) / 1000.0
            p = np.full(shape=(cluster.as_vector().size()),
                                fill_value=p, dtype=np.float32)
            pdg = np.full(shape=(cluster.as_vector().size()),
                            fill_value=pids[i], dtype=np.float32)
            vtx_x = np.full(shape=(cluster.as_vector().size()),
                            fill_value=particles_v_asis[i].ancestor_position().x(), dtype=np.float32)
            vtx_y = np.full(shape=(cluster.as_vector().size()),
                            fill_value=particles_v_asis[i].ancestor_position().y(), dtype=np.float32)
            vtx_z = np.full(shape=(cluster.as_vector().size()),
                            fill_value=particles_v_asis[i].ancestor_position().z(), dtype=np.float32)
            # is_primary = np.full(shape=(cluster.as_vector().size()),
            #             fill_value=float((nu_ids[i] > 0) and (particles_v[i].parent_id() == particles_v[i].id()) and (particles_v[i].group_id() == particles_v[i].id())),
            #             dtype=np.float32)
            is_primary = np.full(shape=(cluster.as_vector().size()),
                        fill_value=float((nu_ids[i] > 0) and (particles_v[i].group_id() == particles_v[i].parent_id())),
                        dtype=np.float32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_features.append(np.column_stack([value, cluster_id, group_id, pdg, p, vtx_x, vtx_y, vtx_z, is_primary]))
    if len(clusters_voxels) > 0:
        np_voxels   = np.concatenate(clusters_voxels, axis=0)
        np_features = np.concatenate(clusters_features, axis=0)
    else:
        np_voxels = np.empty((0, 3), dtype=np.int32)
        np_features = np.empty((0, 9), dtype=np.float32)
    # mask = np_features[:, 6] == np.unique(np_features[:, 6])[0]

    # print(np_features[mask][:, [0, 1, 5, 6]])
    return np_voxels, np_features


def parse_cluster3d_kinematics_clean(data):
    """
    Similar to parse_cluster3d_kinematics, but removes overlap voxels.

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d_kinematics_clean
            - cluster3d_pcluster
            - particle_pcluster
            - particle_mpv
            - sparse3d_pcluster

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D
    particle_pcluster: larcv::EventParticle
    particle_mpv: larcv::EventParticle, optional
        To determine neutrino vs cosmic labels
    sparse3d_pcluster: larcv::EventSparseTensor3D
        This tensor will help determine overlap voxels and final shape.

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (n, 10) where 10 is respectively

        * voxel value,
        * cluster id,
        * group id,
        * pdg,
        * momentum,
        * vtx_x,
        * vtx_y,
        * vtx_z,
        * is_primary,
        * semantic type

    See Also
    --------
    parse_cluster3d_full
    parse_cluster3d_kinematics
    """
    grp_voxels, grp_data = parse_cluster3d_kinematics(data[:-1])
    _, cluster_data = parse_cluster3d_full(data[:-1])
    img_voxels, img_data = parse_sparse3d_scn([data[-1]])

    grp_data = np.concatenate([grp_data, cluster_data[:, -1][:, None]], axis=1)
    grp_voxels, grp_data = clean_data(grp_voxels, grp_data, img_voxels, img_data, data[0].meta())
    return grp_voxels, grp_data#[:, :-1]


def parse_cluster3d_clean_full(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:

    1) lexicographically sort group data (images are lexicographically sorted)

    2) remove voxels from group data that are not in image

    3) choose only one group per voxel (by lexicographic order)

    4) override semantic labels with those from sparse3d
    and give labels -1 to all voxels of class 4 and above

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d_clean_full
            - cluster3d_pcluster
            - particle_mpv
            - sparse3d_pcluster

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D
    particle_mpv: larcv::EventParticle, optional
        To determine neutrino vs cosmic labels
    sparse3d_pcluster: larcv::EventSparseTensor3D
        Will determine final shape and overlap voxels.

    Returns
    -------
    grp_voxels: np.ndarray
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
    grp_data: np.ndarray
        a numpy array with the shape (n,8) where 8 is respectively

        * voxel value,
        * cluster id,
        * group id,
        * interaction id,
        * nu id,
        * particle type,
        * primary id,
        * semantic type

    See Also
    --------
    parse_cluster3d_full
    """
    grp_voxels, grp_data = parse_cluster3d_full(data[:-1])
    img_voxels, img_data = parse_sparse3d_scn([data[-1]])

    grp_voxels, grp_data = clean_data(grp_voxels, grp_data, img_voxels, img_data, data[0].meta())

    # step 4: override semantic labels with those from sparse3d
    # and give labels -1 to all voxels of class 4 and above
    grp_data[:,-1] = img_data[:,-1]
    grp_data[img_data[:,-1] > 3,1:5] = -1
    return grp_voxels, grp_data


def parse_cluster3d_scales(data):
    """
    Retrieves clusters tensors at different spatial sizes.

    .. code-block:: yaml

        schema:
          cluster_label:
            - parse_cluster3d_scales
            - cluster3d_pcluster
            - sparse3d_pcluster

    Configuration
    -------------
    cluster3d_pcluster: larcv::EventClusterVoxel3D
    sparse3d_pcluster: larcv::EventSparseTensor3D
        Will determine final shape and overlap voxels.

    Returns
    -------
    list of tuples
    """
    grp_voxels, grp_data = parse_cluster3d_clean_full(data)
    spatial_size = data[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size))-1)
    scales = []
    for d in range(max_depth):
        scale_voxels = np.floor(grp_voxels/2**d)#.astype(int)
        scale_voxels, unique_indices = np.unique(scale_voxels, axis=0, return_index=True)
        scale_data = grp_data[unique_indices]
        scales.append((scale_voxels, scale_data))
    return scales
