import numpy as np

from mlreco.utils.globals import SHOWR_SHP, TRACK_SHP
from mlreco.utils.vertex import get_vertex

from analysis.post_processing import PostProcessor


class VertexProcessor(PostProcessor):
    '''
    Reconstruct one vertex for each interaction in the provided list
    '''
    name = 'reconstruct_vertex'
    result_cap = ['interactions']
    result_cap_optional = ['truth_interactions']

    def __init__(self,
                 include_semantics = [SHOWR_SHP, TRACK_SHP],
                 use_primaries = True,
                 update_primaries = False,
                 anchor_vertex = True,
                 touching_threshold = 2.0,
                 angle_threshold = 0.3,
                 truth_point_mode = 'points',
                 run_mode = 'both'):
        '''
        Initialize the vertex finder properties

        Parameters
        ----------
        include_semantics : List[int]
            List of semantic classes to consider for vertex reconstruction
        use_primaries : bool, default True
            If true, only considers primary particles to reconstruct the vertex
        update_primaries : bool, default False
            Use the reconstructed vertex to update primaries
        anchor_vertex : bool, default True
            If true, anchor the candidate vertex to particle objects,
            with the expection of interactions only composed of showers
        touching_threshold : float, default 2 cm
            Maximum distance for two track points to be considered touching
        angle_threshold : float, default 0.3 radians
            Maximum angle between the vertex-to-start-point vector and a shower
            direction to consider that a shower originated from the vertex
        truth_point_mode : str, default 'points'
            Point attribute to use for true particles
        run_mode : str, default 'both'
            Which output to run on (one of 'both', 'reco' or 'truth')
        '''
        # Store the relevant parameters
        self.include_semantics = include_semantics
        self.use_primaries = use_primaries
        self.update_primaries = update_primaries
        self.anchor_vertex = anchor_vertex
        self.touching_threshold = touching_threshold
        self.angle_threshold = angle_threshold

        # List objects for which to reconstruct the vertex
        if run_mode not in ['reco', 'truth', 'both']:
            raise ValueError('`run_mode` must be either `reco`, ' \
                    '`truth` or `both`')

        self.key_list = []
        if run_mode in ['reco', 'both']:
            self.key_list += ['interactions']
        if run_mode in ['truth', 'both']:
            self.key_list += ['truth_interactions']
        self.truth_point_mode = truth_point_mode

    def process(self, data_dict, result_dict):
        '''
        Reconstruct the CSDA KE estimates for each particle in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over interaction objects
        for k in self.key_list:
            for ia in result_dict[k]:
                self.reconstruct_vertex_single(ia)

        return {}, {}

    def reconstruct_vertex_single(self, interaction):
        '''
        Post-processor which reconstructs one vertex for each
        interaction in the provided list. It modifies the input list
        of interactions in place.

        Parameters
        ----------
        interaction : List[Interaction, TruthInteraction]
            Reconstructed/truth interaction object
        '''
        # Selected the set of particles to use as a basis for vertex prediction
        if self.use_primaries:
            particles = [p for p in interaction.particles if p.is_primary \
                    and (p.semantic_type in self.include_semantics) \
                    and p.size > 0]
        if not self.use_primaries or not len(particles):
            particles = [p for p in interaction.particles \
                if p.semantic_type in self.include_semantics and p.size > 0]
        if not len(particles):
            particles = [p for p in interaction.particles if p.size > 0]

        if len(particles) > 0:
            # Collapse particle objects to start, end points and directions
            start_points = np.vstack([p.start_point for p in particles])
            end_points   = np.vstack([p.end_point for p in particles])
            directions   = np.vstack([p.start_dir for p in particles])
            semantics    = np.array([p.semantic_type for p in particles])

            # Reconstruct the vertex for this interaction
            vtx, vtx_mode = get_vertex(start_points, end_points, directions,
                semantics, self.anchor_vertex, self.touching_threshold,
                return_mode=True)
            interaction.vertex = vtx
            interaction.vertex_mode = vtx_mode

            # If requested, update primaries on the basis of the vertex
            if self.update_primaries:
                for p in interaction.particles:
                    if p.semantic_type not in [SHOWR_SHP, TRACK_SHP]:
                        p.is_primary = False
                    elif np.linalg.norm(p.start_point - interaction.vertex) \
                            < self.touching_threshold:
                        p.is_primary = True
                    elif p.semantic_type == SHOWR_SHP and \
                            np.dot(p.start_point, interaction.vertex) \
                            < angle_threshold:
                        p.is_primary = True
