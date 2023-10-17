from collections import OrderedDict
import numpy as np
from larcv import larcv
from sklearn.cluster import DBSCAN

from .sparse import parse_sparse3d
from .particles import parse_particles
from .clean_data import clean_sparse_data

from mlreco.utils.globals import UNKWN_SHP
from mlreco.utils.particles import get_interaction_ids, get_nu_ids, get_particle_ids, get_shower_primary_ids, get_group_primary_ids


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
                    neutrino_event = None,
                    sparse_semantics_event = None,
                    sparse_value_event = None,
                    add_particle_info = False,
                    add_kinematics_info = False,
                    clean_data = False,
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
              neutrino_event: neutrino_mpv
              sparse_semantics_event: sparse3d_semantics
              sparse_value_event: sparse3d_pcluster
              add_particle_info: true
              clean_data: true
              type_include_mpr: false
              type_include_secondary: false
              primary_include_mpr: true
              break_clusters: false

    Configuration
    -------------
    cluster_event: larcv::EventClusterVoxel3D
    particle_event: larcv::EventParticle
    particle_mpv_event: larcv::EventParticle
    particle_mpv_event: larcv::EventNeutrino
    sparse_semantics_event: larcv::EventSparseTensor3D
    sparse_value_event: larcv::EventSparseTensor3D
    add_particle_info: bool
    clean_data: bool
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
        * shower primary id,
        * primary group id,
        * vtx (x,y,z),
        * momentum,
        * semantic type
    """
    # Temporary deprecation warning
    if add_kinematics_info:
        from warnings import warn
        warn('add_kinematics_info is deprecated, simply use add_particle_info')
        add_particle_info = True

    # Get the cluster-wise information
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    labels = OrderedDict()
    labels['cluster'] = np.arange(num_clusters)
    if add_particle_info:
        assert particle_event is not None,\
                'Must provide particle tree if particle information is included'
        num_particles = particle_event.size()
        assert num_particles == num_clusters or num_particles == num_clusters-1,\
                'The number of particles must be aligned with the number of clusters'

        particles     = list(particle_event.as_vector())
        particles_mpv = list(particle_mpv_event.as_vector()) if particle_mpv_event is not None else None
        neutrinos     = list(neutrino_event.as_vector()) if neutrino_event is not None else None

        particles_p   = parse_particles(particle_event, cluster_event)

        labels['cluster']  = np.array([p.id() for p in particles])
        labels['group']    = np.array([p.group_id() for p in particles])
        labels['inter']    = get_interaction_ids(particles)
        labels['nu']       = get_nu_ids(particles, labels['inter'], particles_mpv, neutrinos)
        labels['type']     = get_particle_ids(particles, labels['nu'], type_include_mpr, type_include_secondary)
        labels['pshower']  = get_shower_primary_ids(particles)
        labels['pgroup']   = get_group_primary_ids(particles, labels['nu'], primary_include_mpr)
        labels['vtx_x']    = np.array([p.ancestor_position().x() for p in particles_p])
        labels['vtx_y']    = np.array([p.ancestor_position().y() for p in particles_p])
        labels['vtx_z']    = np.array([p.ancestor_position().z() for p in particles_p])
        labels['p']        = np.array([p.p()/1e3 for p in particles]) # In GeV
        labels['particle'] = np.array([p.id() for p in particles]) # TODO: change order
        labels['shape']    = np.array([p.shape() for p in particles])

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
                value = l[i] if i < len(l) else (-1 if k != 'shape' else UNKWN_SHP)
                features.append(np.full(shape=(size), fill_value=value, dtype=np.float32))

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
    if (sparse_semantics_event is not None or sparse_value_event is not None) and not clean_data:
        from warnings import warn
        warn('You should set `clean_data` to True if you specify a sparse tensor in parse_cluster3d')
        clean_data = True

    if clean_data:
        assert add_particle_info, 'Need to add particle info to fetch particle semantics for each voxel'
        assert sparse_semantics_event is not None, 'Need to provide a semantics tensor to clean up output'
        sem_voxels, sem_features = parse_sparse3d([sparse_semantics_event])
        np_voxels,  np_features  = clean_sparse_data(np_voxels, np_features, sem_voxels)
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
                                    neutrino_event = None,
                                    sparse_semantics_event = None,
                                    sparse_value_event_list = None,
                                    add_particle_info = False,
                                    add_kinematics_info = False,
                                    clean_data = False,
                                    type_include_mpr = False,
                                    type_include_secondary = False,
                                    primary_include_mpr = True,
                                    break_clusters = False,
                                    min_size = -1,
                                    collection_only=False):

    # Produces cluster3d labels with sparse3d_reco_rescaled on the fly on datasets that do not have it
    np_voxels, np_features = parse_cluster3d(cluster_event,
                                             particle_event,
                                             particle_mpv_event,
                                             neutrino_event,
                                             sparse_semantics_event,
                                             None,
                                             add_particle_info,
                                             add_kinematics_info,
                                             clean_data,
                                             type_include_mpr,
                                             type_include_secondary,
                                             primary_include_mpr,
                                             break_clusters,
                                             min_size)

    from .sparse import parse_sparse3d_charge_rescaled
    _, val_features  = parse_sparse3d_charge_rescaled(sparse_value_event_list, collection_only)
    np_features[:,0] = val_features[:,-1]

    return np_voxels, np_features

def parse_cluster3d_2cryos(cluster_event,
                                    particle_event = None,
                                    particle_mpv_event = None,
                                    neutrino_event = None,
                                    sparse_semantics_event = None,
                                    sparse_value_event_list = None,
                                    add_particle_info = False,
                                    add_kinematics_info = False,
                                    clean_data = False,
                                    type_include_mpr = False,
                                    type_include_secondary = False,
                                    primary_include_mpr = True,
                                    break_clusters = False,
                                    min_size = -1):

    # Produces cluster3d labels with sparse3d_reco_rescaled on the fly on datasets that do not have it
    np_voxels, np_features = parse_cluster3d(cluster_event,
                                             particle_event,
                                             particle_mpv_event,
                                             neutrino_event,
                                             sparse_semantics_event,
                                             None,
                                             add_particle_info,
                                             add_kinematics_info,
                                             clean_data,
                                             type_include_mpr,
                                             type_include_secondary,
                                             primary_include_mpr,
                                             break_clusters,
                                             min_size)

    from .sparse import parse_sparse3d_charge_rescaled
    _, charge0 = parse_sparse3d([sparse_value_event_list[0]])
    _, charge1 = parse_sparse3d([sparse_value_event_list[1]])
    charge0[charge0 == 0.] = charge1[charge0 == 0.]
    np_features[:,0] = charge0.flatten()

    return np_voxels, np_features
