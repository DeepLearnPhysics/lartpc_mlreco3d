from collections import OrderedDict
import numpy as np
from larcv import larcv
from sklearn.cluster import DBSCAN
from mlreco.utils.groups import get_interaction_id, get_nu_id, get_particle_id, get_shower_primary_id, get_group_primary_id
from mlreco.utils.groups import type_labels as TYPE_LABELS
from mlreco.iotools.parsers.sparse import parse_sparse3d
from mlreco.iotools.parsers.particles import parse_particles
from mlreco.iotools.parsers.clean_data import clean_sparse_data


def parse_cluster2d(cluster_event):
    """
    A function to retrieve a 2D clusters tensor

    .. code-block:: yaml

        schema:
          cluster_label:
            parser: parse_cluster2d
            args:
              cluster_event: cluster2d_pcluster

    Configuration
    -------------
    cluster_event: larcv::EventClusterPixel2D

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (N,2) where 2 represents (x,y)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (N,2) where 2 is pixel value and cluster id, respectively
    """
    cluster_event = cluster_event.as_vector().front()
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


def parse_cluster3d(cluster_event,
                    particle_event = None,
                    particle_mpv_event = None,
                    sparse_semantics_event = None,
                    sparse_value_event = None,
                    add_particle_info = False,
                    add_kinematics_info = False,
                    clean_data = True,
                    precedence = [1,2,0,3,4],
                    type_include_mpr = False,
                    type_include_secondary = False,
                    primary_include_mpr = True,
                    break_clusters = False,
                    min_size = -1):
    """
    a function to retrieve a 3D clusters tensor

    .. code-block:: yaml

        schema:
          cluster_label:
            parser: parse_cluster3d
            args:
              cluster_event: cluster3d_pcluster
              particle_event: particle_pcluster
              particle_mpv_event: particle_mpv
              sparse_semantics_event: sparse3d_semantics
              sparse_value_event: sparse3d_pcluster
              add_particle_info: true
              add_kinematics_info: false
              clean_data: true
              precedence: [1,2,0,3,4]
              type_include_mpr: false
              type_include_secondary: false
              primary_include_mpr: true
              break_clusters: True

    Configuration
    -------------
    cluster_event: larcv::EventClusterVoxel3D
    particle_event: larcv::EventParticle
    particle_mpv_event: larcv::EventParticle
    sparse_semantics_event: larcv::EventSparseTensor3D
    sparse_value_event: larcv::EventSparseTensor3D
    add_particle_info: bool
    add_kinematics_info: bool
    clean_data: bool
    precedence: list
    type_include_mpr: bool
    type_include_secondary: bool
    primary_include_mpr: bool
    break_clusters: bool

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
    np_features: np.ndarray
        a numpy array with the shape (n,m) where m (2-13) includes:
        * voxel value,
        * cluster id
        if add_particle_info is true, it also includes
        * group id,
        * interaction id,
        * nu id,
        * particle type,
        * primary id
        if add_kinematics_info is true, it also includes
        * group id,
        * particle type,
        * momentum,
        * vtx (x,y,z),
        * primary group id
        if either add_* is true, it includes last:
        * semantic type
    """

    # Get the cluster-wise information
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    labels = OrderedDict()
    labels['cluster'] = np.arange(num_clusters)
    if add_particle_info or add_kinematics_info:
        assert particle_event is not None, "Must provide particle tree if particle/kinematics information is included"
        particles_v     = particle_event.as_vector()
        particles_mpv_v = particle_mpv_event.as_vector() if particle_mpv_event is not None else None
        inter_ids       = get_interaction_id(particles_v)
        nu_ids          = get_nu_id(cluster_event, particles_v, inter_ids, particle_mpv=particles_mpv_v)

        labels['cluster'] = np.array([p.id() for p in particles_v])
        labels['group']   = np.array([p.group_id() for p in particles_v])
        if add_particle_info:
            labels['inter']   = inter_ids
            labels['nu']      = nu_ids
            labels['type']    = get_particle_id(particles_v, nu_ids, type_include_mpr, type_include_secondary)
            labels['primary_shower'] = get_shower_primary_id(cluster_event, particles_v)
        if add_kinematics_info:
            primary_ids       = get_group_primary_id(particles_v, nu_ids, primary_include_mpr)
            labels['type']    = get_particle_id(particles_v, nu_ids, type_include_mpr, type_include_secondary)
            labels['p']       = np.array([p.p()/1e3 for p in particles_v]) # In GeV
            particles_v       = parse_particles(particle_event, cluster_event)
            labels['vtx_x']   = np.array([p.ancestor_position().x() for p in particles_v])
            labels['vtx_y']   = np.array([p.ancestor_position().y() for p in particles_v])
            labels['vtx_z']   = np.array([p.ancestor_position().z() for p in particles_v])
            labels['primary_group'] = primary_ids
        labels['sem'] = np.array([p.shape() for p in particles_v])

    # Loop over clusters, store info
    clusters_voxels, clusters_features = [], []
    id_offset = 0
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points >= max(min_size, 1):
            # Get the position and pixel value from EventSparseTensor3D, append positions
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster, meta, x, y, z, value)
            voxels = np.stack([x, y, z], axis=1)
            clusters_voxels.append(voxels)

            # Append the cluster-wise information
            features = [value]
            for k, l in labels.items():
                size = cluster.as_vector().size()
                features.append(np.full(shape=(size), fill_value=l[i], dtype=np.float32))

            # If requested, break cluster into pieces that do not touch each other
            if break_clusters:
                dbscan = DBSCAN(eps=1.1, min_samples=1, metric='chebyshev')
                frag_labels = np.unique(dbscan.fit(voxels).labels_, return_inverse=True)[-1]
                features[1] = id_offset + frag_labels
                id_offset += max(frag_labels) + 1

            clusters_features.append(np.column_stack(features))

    # If there are no non-empty clusters, return. Concatenate otherwise
    if not len(clusters_voxels):
        return np.empty(shape=(0, 3), dtype=np.float32), np.empty(shape=(0, len(labels)+1), dtype=np.float32)
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)

    # If requested, remove duplicate voxels (cluster overlaps) and account for semantics
    if clean_data:
        assert sparse_semantics_event is not None, 'Need to provide a semantics tensor to clean up output'
        sem_voxels, sem_features = parse_sparse3d([sparse_semantics_event])
        np_voxels,  np_features  = clean_sparse_data(np_voxels, np_features, sem_voxels, sem_features, meta, precedence)
        np_features[:,-1] = sem_features[:,-1] # Match semantic column to semantic tensor
        np_features[sem_features[:,-1] > 3, 1:-1] = -1 # Set all cluster labels to -1 if semantic class is LE or ghost

        # If a value tree is provided, override value colum
        if sparse_value_event:
            _, val_features  = parse_sparse3d([sparse_value_event])
            np_features[:,0] = val_features[:,-1]

    return np_voxels, np_features


def parse_cluster3d_charge_rescaled(cluster_event,
                                    particle_event = None,
                                    particle_mpv_event = None,
                                    sparse_semantics_event = None,
                                    sparse_value_event_list = None,
                                    add_particle_info = False,
                                    add_kinematics_info = False,
                                    clean_data = True,
                                    precedence = [1,2,0,3,4],
                                    type_include_mpr = False,
                                    type_include_secondary = False,
                                    primary_include_mpr = True,
                                    break_clusters = False,
                                    min_size = -1):

    # Produces cluster3d labels with sparse3d_reco_rescaled on the fly on datasets that do not have it
    np_voxels, np_features = parse_cluster3d(cluster_event, particle_event, particle_mpv_event, sparse_semantics_event, None,
                                             add_particle_info, add_kinematics_info, clean_data, precedence, 
                                             type_include_mpr, type_include_secondary, primary_include_mpr, break_clusters, min_size)

    from .sparse import parse_sparse3d_charge_rescaled
    _, val_features  = parse_sparse3d_charge_rescaled(sparse_value_event_list)
    np_features[:,0] = val_features[:,-1]

    return np_voxels, np_features
