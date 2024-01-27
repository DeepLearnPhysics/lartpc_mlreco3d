import numpy as np

from mlreco.utils.globals import TRACK_SHP, PID_LABELS
from mlreco.utils.geometry import Geometry
from mlreco.utils.gnn.cluster import cluster_direction

from analysis.classes import Interaction, TruthInteraction
from analysis.post_processing import PostProcessor


class DirectionProcessor(PostProcessor):
    '''
    Reconstruct the direction of particles w.r.t. to their end points.
    '''
    name = 'reconstruct_directions'
    result_cap = ['particles']
    result_cap_opt = ['truth_particles']

    def __init__(self,
                 neighborhood_radius = -1,
                 optimize = True,
                 truth_point_mode = 'points',
                 run_mode = 'both'):
        '''
        Store the particle direction recosntruction parameters

        Parameters
        ----------
        neighborhood_radius : float, default 5
            Max distance between start voxel and other voxels
        optimize : bool, default True
            Optimizes the number of points involved in the estimate
        '''
        # Initialize the parent class
        super().__init__(run_mode, truth_point_mode)

        # Store the direction reconstruction parameters
        self.neighborhood_radius = neighborhood_radius
        self.optimize = optimize

    def process(self, data_dict, result_dict):
        '''
        Reconstruct the directions of all particles in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over particle objects
        for k in self.part_keys:
            for p in result_dict[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(p)

                # Get point coordinates
                points = self.get_points(p)
                if not len(points):
                    continue

                # Reconstruct directions from either end of the particle
                p.start_dir = cluster_direction(points, p.start_point,
                        self.neighborhood_radius, self.optimize)
                if p.semantic_type == TRACK_SHP:
                    p.end_dir   = cluster_direction(points, p.end_point,
                            self.neighborhood_radius, self.optimize)

        return {}, {}


class ContainmentProcessor(PostProcessor):
    '''
    Check whether a particle or interaction comes within some distance
    of the boundaries of the detector and assign the `is_contained`
    attribute accordingly.
    '''
    name = 'check_containment'
    result_cap = ['particles', 'interactions']
    result_cap_opt = ['truth_particles', 'truth_interactions']

    def __init__(self,
                 margin,
                 cathode_margin = None,
                 detector = None,
                 boundary_file = None,
                 source_file = None,
                 mode = 'module',
                 allow_multi_module = False,
                 min_particle_sizes = 0,
                 truth_point_mode = 'points',
                 run_mode = 'both'):
        '''
        Initialize the containment conditions.

        If the `source` method is used, the cut will be based on the source of
        the point cloud, i.e. if a point cloud was produced by TPCs i and j, it
        must be contained within the volume bound by the set of TPCs i and j,
        and whichever volume is present between them.

        Parameters
        ----------
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of falls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is
              specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        source_file : str, optional
            Path to a detector source file. Supersedes `detector` if set
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'tpc', makes sure it is contained within a single tpc
            - If 'module', makes sure it is contained within a single module
            - If 'detector', makes sure it is contained within the
              outermost walls
            - If 'source', use the origin of voxels to determine which TPC(s)
              contributed to them, and define volumes accordingly
        allow_multi_module : bool, default False
            Whether to allow particles/interactions to span multiple modules
        min_particle_sizes : Union[int, dict], default 0
            When checking interaction containment, ignore particles below the
            size (in voxel count) specified by this parameter. If specified
            as a dictionary, it maps a specific particle type to its own cut.
        '''
        # Initialize the parent class
        super().__init__(run_mode, truth_point_mode)

        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file, source_file)
        self.geo.define_containment_volumes(margin, cathode_margin, mode)

        # Store parameters
        self.allow_multi_module = allow_multi_module

        # Store the particle size thresholds in a dictionary
        if np.isscalar(min_particle_sizes):
            min_particle_sizes = {'default': min_particle_sizes}

        self.min_particle_sizes = {}
        for pid in PID_LABELS.keys():
            if pid in min_particle_sizes:
                self.min_particle_sizes[pid] = min_particle_sizes[pid]
            elif 'default' in min_particle_sizes:
                self.min_particle_sizes[pid] = min_particle_sizes['default']
            else:
                self.min_particle_sizes[pid] = 0

    def process(self, data_dict, result_dict):
        '''
        Check the containment of all particles/interactions in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over particle objects
        for k in self.part_keys:
            for p in result_dict[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(p)

                # Get point coordinates
                points = self.get_points(p)
                if not len(points):
                    p.is_contained = True
                    continue

                # Check particle containment
                p.is_contained = self.geo.check_containment(points,
                        p.sources, self.allow_multi_module)

        # Loop over interaction objects
        for k in self.inter_keys:
            for ii in result_dict[k]:
                # Check that all the particles in the interaction are contained
                ii.is_contained = True
                for p in ii.particles:
                    if not p.is_contained:
                        # Do not account for particles below a certain size
                        if p.pid > -1 \
                                and p.size < self.min_particle_sizes[p.pid]:
                            continue

                        ii.is_contained = False
                        break

        return {}, {}


class FiducialProcessor(PostProcessor):
    '''
    Check whether an interaction vertex is within some fiducial volume defined
    as margin distances from each of the detector walls.
    '''
    name = 'check_fiducial'
    result_cap = ['interactions']
    result_cap_opt = ['truth_interactions']

    def __init__(self,
                 margin,
                 cathode_margin = None,
                 detector = None,
                 boundary_file = None,
                 mode = 'module',
                 truth_vertex_mode = 'truth_vertex',
                 run_mode = 'both'):
        '''
        Initialize the fiducial conditions

        Parameters
        ----------
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of falls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is
              specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        detector : str, default 'icarus'
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'tpc', makes sure it is contained within a single tpc
            - If 'module', makes sure it is contained within a single module
            - If 'detector', makes sure it is contained within the
              outermost walls
        truth_vertex_mode : str, default 'truth_vertex'
             Vertex attribute to use to check containment of true interactions
        '''
        # Initialize the parent class
        super().__init__(run_mode)

        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file)
        self.geo.define_containment_volumes(margin, cathode_margin, mode)

        # Store the true vertex source
        self.truth_vertex_mode = truth_vertex_mode

    def process(self, data_dict, result_dict):
        '''
        Check the fiducial status of all interactions in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over interaction objects
        for k in self.inter_keys:
            for ia in result_dict[k]:
                # Make sure the interaction coordinates are expressed in cm
                self.check_units(ia)

                # Get point coordinates
                if not isinstance(ia, TruthInteraction):
                    vertex = ia.vertex
                else:
                    vertex = getattr(ia, self.truth_vertex_mode)
                vertex = vertex.reshape(-1,3)

                # Check containment
                ia.is_fiducial = self.geo.check_containment(vertex)

        return {}, {}
