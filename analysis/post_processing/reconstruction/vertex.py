import numpy as np

from mlreco.utils.globals import SHOWR_SHP, TRACK_SHP
from mlreco.utils.vertex import get_vertex

from analysis.post_processing import post_processing


@post_processing(data_capture=[],
                 result_capture=['interactions'],
                 result_capture_optional=['truth_interactions'])
def reconstruct_vertex(data_dict, result_dict,
                       include_semantics = [SHOWR_SHP, TRACK_SHP],
                       use_primaries = True,
                       update_primaries = False,
                       anchor_vertex = True,
                       touching_threshold = 2.0,
                       angle_threshold = 0.3,
                       run_mode = 'both'):
    '''
    Post-processor which reconstructs one vertex for each
    interaction in the provided list. It modifies the input list
    of interactions in place.

    Parameters
    ----------
    interactions : List[Interaction]
        List of reconstructed particle interactions
    truth_interactions : List[TruthInteractions], optional
        List of true interactions
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
        Maximum angle between the vertex-to-start-point vector and a
        shower direction to consider that a shower originated from the vertex
    run_mode : str, default 'both'
        One of `reco`, `truth`, `both` to tell which interaction types to
        apply this algorithm to.
    '''
    # Loop over interactions
    if run_mode not in ['reco', 'truth', 'both']:
        raise ValueError('`run_mode` must be either `reco`, `truth` or `both`')

    if run_mode in ['reco', 'both']:
        for ia in result_dict['interactions']:
            reconstruct_vertex_single(ia, include_semantics, use_primaries, update_primaries,
                    anchor_vertex, touching_threshold, angle_threshold)

    if run_mode in ['truth', 'both']:
        assert 'truth_interactions' in result_dict,\
                'Need truth interaction to apply vertex reconstruction to them'
        for ia in result_dict['truth_interactions']:
            reconstruct_vertex_single(ia, include_semantics, use_primaries, False,
                    anchor_vertex, touching_threshold, angle_threshold)

    return {}


def reconstruct_vertex_single(interaction,
                              include_semantics,
                              use_primaries,
                              update_primaries,
                              anchor_vertex,
                              touching_threshold,
                              angle_threshold):

    '''
    Post-processor which reconstructs one vertex for each
    interaction in the provided list. It modifies the input list
    of interactions in place.

    Parameters
    ----------
    interaction : List[Interaction, TruthInteraction]
        Reconstructed/truth interaction object
    include_semantics : List[int]
        List of semantic classes to consider for vertex reconstruction
    use_primaries : bool
        If true, only considers primary particles to reconstruct the vertex
    update_primaries : bool
        Use the reconstructed vertex to update primaries
    anchor_vertex : bool
        If true, anchor the candidate vertex to particle objects,
        with the expection of interactions only composed of showers
    touching_threshold : float
        Maximum distance for two track points to be considered touching
    angle_threshold : float
        Maximum angle between the vertex-to-start-point vector and a
        shower direction to consider that a shower originated from the vertex
    '''
    # Selected the set of particles to use as a basis for vertex prediction
    if use_primaries:
        particles = [p for p in interaction.particles \
            if p.is_primary and (p.semantic_type in include_semantics) and p.size > 0]
    if not use_primaries or not len(particles):
        particles = [p for p in interaction.particles \
            if p.semantic_type in include_semantics and p.size > 0]
    if not len(particles):
        particles = [p for p in interaction.particles if p.size > 0]

    if len(particles) > 0:
        # Collapse particle objects to a set of start, end points and directions
        start_points = np.vstack([p.start_point for p in particles]).astype(np.float32)
        end_points   = np.vstack([p.end_point for p in particles]).astype(np.float32)
        directions   = np.vstack([p.start_dir for p in particles]).astype(np.float32)
        semantics    = np.array([p.semantic_type for p in particles], dtype=np.int32)

        # Reconstruct the vertex for this interaction
        vtx, vtx_mode = get_vertex(start_points, end_points, directions, semantics,
                anchor_vertex, touching_threshold, return_mode=True)
        interaction.vertex = vtx
        interaction.vertex_mode = vtx_mode

        # If requested, update primaries on the basis of the predicted vertex
        if update_primaries:
            for p in interaction.particles:
                if p.semantic_type not in [SHOWR_SHP, TRACK_SHP]:
                    p.is_primary = False
                elif np.linalg.norm(p.start_point - interaction.vertex) < touching_threshold:
                    p.is_primary = True
                elif p.semantic_type == SHOWR_SHP and np.dot(p.start_point, interaction.vertex) < angle_threshold:
                    p.is_primary = True
