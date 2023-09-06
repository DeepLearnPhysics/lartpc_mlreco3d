import numpy as np
from larcv import larcv

from mlreco.utils.globals import PDG_TO_PID
from mlreco.utils.ppn import get_ppn_labels

def parse_particles(particle_event, sparse_event=None, cluster_event=None, voxel_coordinates=True):
    """
    A function to copy construct & return an array of larcv::Particle.

    If `voxel_coordinates` is set to `True`, the parser rescales the truth
    positions (start, end, etc.) to voxel coordinates.

    .. code-block:: yaml

        schema:
          particles:
            parser: parse_particles
            args:
              particle_event: particle_pcluster
              cluster_event: cluster3d_pcluster
              voxel_coordinates: True

    Configuration
    -------------
    particle_event: larcv::EventParticle
    sparse_event: larcv::EventSparseTensor3D
    cluster_event: larcv::EventClusterVoxel3D
        to translate coordinates
    voxel_coordinates: bool

    Returns
    -------
    list
        a python list of larcv::Particle objects
    """
    particles = [larcv.Particle(p) for p in particle_event.as_vector()]
    if voxel_coordinates:
        assert (sparse_event is not None) ^ (cluster_event is not None)
        meta = sparse_event.meta() if sparse_event is not None else cluster_event.meta()
        funcs = ['first_step', 'last_step', 'position', 'end_position', 'parent_position', 'ancestor_position']
        for p in particles:
            for f in funcs:
                pos = getattr(p,f)()
                x = (pos.x() - meta.min_x()) / meta.size_voxel_x()
                y = (pos.y() - meta.min_y()) / meta.size_voxel_y()
                z = (pos.z() - meta.min_z()) / meta.size_voxel_z()
                getattr(p,f)(x,y,z,pos.t())

    return particles


def parse_neutrinos(neutrino_event, sparse_event=None, cluster_event=None, voxel_coordinates=True):
    """
    A function to copy construct & return an array of larcv::Neutrino.

    If `voxel_coordinates` is set to `True`, the parser rescales the truth
    position information to voxel coordinates.

    .. code-block:: yaml

        schema:
          neutrinos:
            parser: parse_neutrinos
            args:
              neutrino_event: neutrino_mpv
              cluster_event: cluster3d_pcluster
              voxel_coordinates: True

    Configuration
    -------------
    neutrino_pcluster: larcv::EventNeutrino
    sparse_event: larcv::EventSparseTensor3D
    cluster3d_pcluster: larcv::EventClusterVoxel3D
        to translate coordinates
    voxel_coordinates: bool

    Returns
    -------
    list
        a python list of larcv::Neutrino objects
    """
    neutrinos = [larcv.Neutrino(p) for p in neutrino_event.as_vector()]
    if voxel_coordinates:
        assert (sparse_event is not None) ^ (cluster_event is not None)
        meta = sparse_event.meta() if sparse_event is not None else cluster_event.meta()
        funcs = ['position']
        for p in neutrinos:
            for f in funcs:
                pos = getattr(p,f)()
                x = (pos.x() - meta.min_x()) / meta.size_voxel_x()
                y = (pos.y() - meta.min_y()) / meta.size_voxel_y()
                z = (pos.z() - meta.min_z()) / meta.size_voxel_z()
                getattr(p,f)(x,y,z,pos.t())

    return neutrinos


def parse_particle_points(sparse_event, particle_event, include_point_tagging=True):
    """
    A function to retrieve particles ground truth points tensor, returns
    points coordinates, types and particle index.
    If include_point_tagging is true, it includes start vs end point tagging.

    .. code-block:: yaml

        schema:
          points:
            parser: parse_particle_points
            args:
              sprase_event: sparse3d_pcluster
              particle_event: particle_pcluster
              include_point_tagging: True

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
    particle_pcluster: larcv::EventParticle
    include_point_tagging: bool

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
    np_values: np.ndarray
        a numpy array with the shape (N, 2) where 2 represents the class of the ground truth point
        and the particle data index in this order. (optionally: end/start tagging)
    """
    particles_v = particle_event.as_vector()
    part_labels = get_ppn_labels(particles_v, sparse_event.meta(),
            include_point_tagging=include_point_tagging)

    return part_labels[:,:3], part_labels[:,3:]


def parse_particle_coords(particle_event, cluster_event):
    '''
    Function that returns particle coordinates (start and end) and start time.

    This is used for particle clustering into interactions

    .. code-block:: yaml

        schema:
          coords:
            parser: parse_particle_coords
            args:
              particle_event: particle_pcluster
              cluster_event: cluster3d_pcluster

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
    particles = parse_particles(particle_event, cluster_event)

    # Make features
    particle_feats = []
    for i, p in enumerate(particles):
        start_point = last_point = [p.first_step().x(), p.first_step().y(), p.first_step().z()]
        if p.shape() == 1: # End point only meaningful and thought out for tracks
            last_point  = [p.last_step().x(), p.last_step().y(), p.last_step().z()]
        particle_feats.append(np.concatenate((start_point, last_point, [p.first_step().t(), p.shape()])))

    particle_feats = np.vstack(particle_feats)
    return particle_feats[:,:3], particle_feats[:,3:]


def parse_particle_graph(particle_event, cluster_event=None):
    """
    A function to parse larcv::EventParticle to construct edges between particles (i.e. clusters)

    If cluster_event is provided, it also removes edges to clusters
    that have a zero pixel count and patches subsequently broken parentage.

    .. code-block:: yaml

        schema:
          graph:
            parser: parse_particle_graph
            args:
              particle_event: particle_pcluster
              cluster_event: cluster3d_pcluster

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
    particles_v   = particle_event.as_vector()
    num_particles = particles_v.size()
    if cluster_event is None:
        # Fill edges (directed [parent,child] pair)
        edges = np.empty((0,2), dtype = np.int32)
        for cluster_id in range(num_particles):
            p = particles_v[cluster_id]
            if p.parent_id() != p.id():
                edges = np.vstack((edges, [int(p.parent_id()), cluster_id]))
            if p.parent_id() == p.id() and p.group_id() != p.id():
                edges = np.vstack((edges, [int(p.group_id()), cluster_id]))
    else:
        # Check that the cluster and particle objects are consistent
        num_clusters = cluster_event.size()
        assert num_clusters == num_particles

        # Fill edges (directed [parent,child] pair)
        zero_nodes, zero_nodes_pid = [], []
        edges = np.empty((0,2), dtype = np.int32)
        for cluster_id in range(num_particles):
            cluster = cluster_event.as_vector()[cluster_id]
            num_points = cluster.as_vector().size()
            p = particles_v[cluster_id]
            if p.id() != p.group_id():
                continue
            if p.parent_id() != p.group_id():
                edges = np.vstack((edges, [int(p.parent_id()),p.group_id()]))
            if num_points == 0:
                zero_nodes.append(p.group_id())
                zero_nodes_pid.append(cluster_id)

        # Remove zero pixel nodes
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

    return edges


def parse_particle_singlep_pdg(particle_event):
    """
    Get each true particle's PDG code.

    .. code-block:: yaml

        schema:
          pdg_list:
            parser: parse_particle_singlep_pdg
            args:
              particle_event: particle_pcluster

    Configuration
    ----------
    particle_event : larcv::EventParticle

    Returns
    -------
    np.ndarray
        List of PDG codes for each particle in TTree.
    """
    pdgs = []
    pdg = -1
    for p in particle_event.as_vector():
        if not p.track_id() == 1: continue
        if int(p.pdg_code()) in PDG_TO_PID.keys():
            pdg = PDG_TO_PID[int(p.pdg_code())]
        else: pdg = -1
        return np.asarray([pdg])

    return np.asarray([pdg])


def parse_particle_singlep_einit(particle_event):
    """
    Get each true particle's true initial energy.

    .. code-block:: yaml

        schema:
          einit_list:
            parser: parse_particle_singlep_pdg
            args:
              particle_event: particle_pcluster

    Configuration
    ----------
    particle_event : larcv::EventParticle

    Returns
    -------
    np.ndarray
        List of true initial energy for each particle in TTree.
    """
    einits = []
    einit = -1
    for p in particle_event.as_vector():
        is_primary = p.track_id() == p.parent_track_id()
        if not p.track_id() == 1: continue
        return np.asarray([p.energy_init()])

    return np.asarray([einit])
