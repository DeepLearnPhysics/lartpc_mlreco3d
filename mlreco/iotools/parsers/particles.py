import numpy as np
from larcv import larcv
from mlreco.utils.ppn import get_ppn_info
from mlreco.utils.groups import type_labels as TYPE_LABELS
# Global type labels for PDG to Particle Type Label (nominal) conversion.


def parse_particle_singlep_pdg(data):
    """
    Get each true particle's PDG code.

    .. code-block:: yaml

        schema:
          pdg_list:
            - parse_particle_singlep_pdg
            - particle_pcluster

    Configuration
    ----------
    particle_pcluster : larcv::EventParticle

    Returns
    -------
    np.ndarray
        List of PDG codes for each particle in TTree.
    """
    parts = data[0]
    pdgs = []
    pdg = -1
    for p in parts.as_vector():
        # print(p.track_id())
        if not p.track_id() == 1: continue
        if int(p.pdg_code()) in TYPE_LABELS.keys():
            pdg = TYPE_LABELS[int(p.pdg_code())]
        else: pdg = -1
        return np.asarray([pdg])

    return np.asarray([pdg])


def parse_particle_singlep_einit(data):
    """
    Get each true particle's true initial energy.

    .. code-block:: yaml

        schema:
          pdg_list:
            - parse_particle_singlep_einit
            - particle_pcluster

    Configuration
    ----------
    particle_pcluster : larcv::EventParticle

    Returns
    -------
    np.ndarray
        List of true initial energy for each particle in TTree.
    """
    parts = data[0]
    for p in parts.as_vector():
        is_primary = p.track_id() == p.parent_track_id()
        if not p.track_id() == 1: continue
        return p.energy_init()
    return -1


def parse_particle_asis(data):
    """
    A function to copy construct & return an array of larcv::Particle

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_particle_asis
            - particle_pcluster
            - cluster3d_pcluster

    Configuration
    -------------
    particle_pcluster: larcv::EventParticle
    cluster3d_pcluster: larcv::EventClusterVoxel3D
        to translate coordinates

    Returns
    -------
    list
        a python list of larcv::Particle object
    """
    particles = data[0]
    particles = [larcv.Particle(p) for p in data[0].as_vector()]

    clusters  = data[1]
    #assert data[0].as_vector().size() in [clusters.as_vector().size(),clusters.as_vector().size()-1]

    meta = clusters.meta()


    funcs = ["first_step","last_step","position","end_position","ancestor_position"]
    for p in particles:
        for f in funcs:
            pos = getattr(p,f)()
            x = (pos.x() - meta.min_x()) / meta.size_voxel_x()
            y = (pos.y() - meta.min_y()) / meta.size_voxel_y()
            z = (pos.z() - meta.min_z()) / meta.size_voxel_z()
            # x = (pos.x() - meta.origin().x) / meta.size_voxel_x()
            # y = (pos.y() - meta.origin().y) / meta.size_voxel_y()
            # z = (pos.z() - meta.origin().z) / meta.size_voxel_z()
            # x = pos.x() * meta.size_voxel_x() + meta.origin().x
            # y = pos.y() * meta.size_voxel_y() + meta.origin().y
            # z = pos.z() * meta.size_voxel_z() + meta.origin().z
            getattr(p,f)(x,y,z,pos.t())
    return particles


def parse_neutrino_asis(data):
    """
    A function to copy construct & return an array of larcv::Neutrino

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_neutrino_asis
            - neutrino_pcluster
            - cluster3d_pcluster

    Configuration
    -------------
    neutrino_pcluster: larcv::EventNeutrino
    cluster3d_pcluster: larcv::EventClusterVoxel3D
        to translate coordinates

    Returns
    -------
    list
        a python list of larcv::Neutrino object
    """
    neutrinos = data[0]
    neutrinos = [larcv.Neutrino(p) for p in data[0].as_vector()]

    clusters  = data[1]
    #assert data[0].as_vector().size() in [clusters.as_vector().size(),clusters.as_vector().size()-1]

    meta = clusters.meta()


    #funcs = ["first_step","last_step","position","end_position","ancestor_position"]
    funcs = ["position"]
    for p in neutrinos:
        for f in funcs:
            pos = getattr(p,f)()
            x = (pos.x() - meta.min_x()) / meta.size_voxel_x()
            y = (pos.y() - meta.min_y()) / meta.size_voxel_y()
            z = (pos.z() - meta.min_z()) / meta.size_voxel_z()
            # x = (pos.x() - meta.origin().x) / meta.size_voxel_x()
            # y = (pos.y() - meta.origin().y) / meta.size_voxel_y()
            # z = (pos.z() - meta.origin().z) / meta.size_voxel_z()
            # x = pos.x() * meta.size_voxel_x() + meta.origin().x
            # y = pos.y() * meta.size_voxel_y() + meta.origin().y
            # z = pos.z() * meta.size_voxel_z() + meta.origin().z
            getattr(p,f)(x,y,z,pos.t())
    return neutrinos


def parse_particle_coords(data):
    '''
    Function that returns particle coordinates (start and end) and start time.

    This is used for particle clustering into interactions

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_particle_coords
            - particle_pcluster
            - cluster3d_pcluster

    Configuration
    -------------
    particle_pcluster: larcv::EventParticle
    cluster3d_pcluster: larcv::EventClusterVoxel3D
        to translate coordinates

    Returns
    -------
    numpy.ndarray
        Shape (N,8) containing: [first_step_x, first_step_y, first_step_z,
        last_step_x, last_step_y, last_step_z, first_step_t, shape_id]
    '''
    # Scale particle coordinates to image size
    particles = parse_particle_asis(data)

    # Make features
    particle_feats = []
    for i, p in enumerate(particles):
        start_point = last_point = [p.first_step().x(), p.first_step().y(), p.first_step().z()]
        if p.shape() == 1: # End point only meaningful and thought out for tracks
            last_point  = [p.last_step().x(), p.last_step().y(), p.last_step().z()]
        particle_feats.append(np.concatenate((start_point, last_point, [p.first_step().t(), p.shape()])))

    particle_feats = np.vstack(particle_feats)
    return particle_feats[:,:3], particle_feats[:,3:]


def parse_particle_points(data, include_point_tagging=False):
    """
    A function to retrieve particles ground truth points tensor, returns
    points coordinates, types and particle index.

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_particle_points
            - sparse3d_pcluster
            - particle_pcluster

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
    particle_pcluster: larcv::EventParticle

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
    np_values: np.ndarray
        a numpy array with the shape (N, 2) where 2 represents the class of the ground truth point
        and the particle data index in this order. (optionally: end/start tagging)
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    # For open data - to reproduce
    # part_info = get_ppn_info(particles_v, data[0].meta(), min_voxel_count=7, min_energy_deposit=10, use_particle_shape=False)
    # part_info = get_ppn_info(particles_v, data[0].meta(), min_voxel_count=5, min_energy_deposit=10, use_particle_shape=False)
    np_values = np.column_stack([part_info[:, 3], part_info[:, 8]]) if part_info.shape[0] > 0 else np.empty(shape=(0, 2), dtype=np.float32)
    if include_point_tagging:
        np_values = np.column_stack([part_info[:, 3], part_info[:, 8], part_info[:, 9]]) if part_info.shape[0] > 0 else np.empty(shape=(0, 3), dtype=np.float32)

    if part_info.shape[0] > 0:
        #return part_info[:, :3], part_info[:, 3][:, None]
        return part_info[:, :3], np_values
    else:
        #return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 1), dtype=np.float32)
        return np.empty(shape=(0, 3), dtype=np.int32), np_values


def parse_particle_points_with_tagging(data):
    """
    Same as `parse_particle_points` including start vs end point tagging.

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_particle_points_with_tagging
            - sparse3d_pcluster
            - particle_pcluster

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
    particle_pcluster: larcv::EventParticle

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
    np_values: np.ndarray
        a numpy array with the shape (N, 3) where 3 represents the class of the ground truth point,
        the particle data index and end/start tagging in this order.

    See Also
    ---------
    parse_particle_points
    """
    return parse_particle_points(data, include_point_tagging=True)


def parse_particle_graph(data):
    """
    A function to parse larcv::EventParticle to construct edges between particles (i.e. clusters)

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_particle_graph
            - particle_pcluster

    Configuration
    -------------
    particle_pcluster: larcv::EventParticle

    Returns
    -------
    np.ndarray
        a numpy array of directed edges where each edge is (parent,child) batch index ID.

    See Also
    --------
    parse_particle_graph_corrected: in addition, remove empty clusters.
    """
    particles = data[0]

    # For convention, construct particle id => cluster id mapping
    particle_to_cluster = np.zeros(shape=[particles.as_vector().size()],dtype=np.int32)

    # Fill edges (directed, [parent,child] pair)
    edges = np.empty((0,2), dtype = np.int32)
    for cluster_id in range(particles.as_vector().size()):
        p = particles.as_vector()[cluster_id]
        #print(p.id(), p.parent_id(), p.group_id())
        if p.parent_id() != p.id():
            edges = np.vstack((edges, [int(p.parent_id()),cluster_id]))
        if p.parent_id() == p.id() and p.group_id() != p.id():
            edges = np.vstack((edges, [int(p.group_id()),cluster_id]))

    return edges


def parse_particle_graph_corrected(data):
    """
    A function to parse larcv::EventParticle to construct edges between particles (i.e. clusters)

    Also removes edges to clusters that have a zero pixel count.

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_particle_graph_corrected
            - particle_pcluster
            - cluster3d_pcluster

    Configuration
    -------------
    particle_pcluster: larcv::EventParticle
    cluster3d_pcluster: larcv::EventClusterVoxel3D

    Returns
    -------
    np.ndarray
        a numpy array of directed edges where each edge is (parent,child) batch index ID.

    See Also
    --------
    parse_particle_graph: same parser without correcting for empty clusters.
    """
    particles = data[0]
    cluster_event = data[1]

    # For convention, construct particle id => cluster id mapping
    num_clusters = cluster_event.size()
    num_particles = particles.as_vector().size()
    assert num_clusters == num_particles

    zero_nodes = []
    zero_nodes_pid = []

    # Fill edges (directed, [parent,child] pair)
    edges = np.empty((0,2), dtype = np.int32)
    for cluster_id in range(num_particles):
        cluster = cluster_event.as_vector()[cluster_id]
        num_points = cluster.as_vector().size()
        p = particles.as_vector()[cluster_id]
        #print(p.id(), p.parent_id(), p.group_id())
        if p.id() != p.group_id():
            continue
        if p.parent_id() != p.group_id():
            edges = np.vstack((edges, [int(p.parent_id()),p.group_id()]))
        if num_points == 0:
            zero_nodes.append(p.group_id())
            zero_nodes_pid.append(cluster_id)

    # Remove zero pixel nodes:
    # print('------------------------------')
    # print(edges)
    # print(zero_nodes)
    for i, zn in enumerate(zero_nodes):
        children = np.where(edges[:, 0] == zn)[0]
        if len(children) == 0:
            edges = edges[edges[:, 0] != zn]
            edges = edges[edges[:, 1] != zn]
            continue
        parent = np.where(edges[:, 1] == zn)[0]
        assert len(parent) <= 1
        # If zero node has a parent, then assign children to that parent
        if len(parent) == 1:
            parent_id = edges[parent][0][0]
            edges[:, 0][children] = parent_id
        else:
            edges = edges[edges[:, 0] != zn]
        edges = edges[edges[:, 1] != zn]
    # print(edges)

    return edges
