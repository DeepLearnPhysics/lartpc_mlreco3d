import numpy as np

from typing import List, Union
from collections import defaultdict, OrderedDict, Counter

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from . import Particle, TruthParticle, Interaction, TruthInteraction


def matrix_counts(particles_x, particles_y):
    """Function for computing the M x N overlap matrix by counts.

    Parameters
    ----------
    particles_x: List[Particle]
        List of N particles to match with <particles_y>
    particles_y: List[Particle]
        List of M particles to match with <particles_x>

    Note the correspondence particles_x -> N and particles_y -> M.

    Returns
    -------
    overlap_matrix: (M, N) np.array of ints
    """
    overlap_matrix = np.zeros((len(particles_y), len(particles_x)), dtype=np.int64)
    for i, py in enumerate(particles_y):
        for j, px in enumerate(particles_x):
            overlap_matrix[i, j] = len(np.intersect1d(py.index,
                                                      px.index))
    return overlap_matrix


def matrix_iou(particles_x, particles_y):
    """Function for computing the M x N overlap matrix by IoU.

    Here IoU refers to Intersection-over-Union metric.

    Parameters
    ----------
    particles_x: List[Particle]
        List of N particles to match with <particles_y>
    particles_y: List[Particle]
        List of M particles to match with <particles_x>

    Note the correspondence particles_x -> N and particles_y -> M.

    Returns
    -------
    overlap_matrix: (M, N) np.float array, with range [0, 1]
    """
    overlap_matrix = np.zeros((len(particles_y), len(particles_x)), dtype=np.float32)
    for i, py in enumerate(particles_y):
        for j, px in enumerate(particles_x):
            cap = np.intersect1d(py.index, px.index)
            cup = np.union1d(py.index, px.index)
            overlap_matrix[i, j] = float(cap.shape[0]) / float(cup.shape[0])
    return overlap_matrix


def matrix_chamfer(particles_x, particles_y, mode='default'):
    """Function for computing the M x N overlap matrix by the Chamfer distance.

    Parameters
    ----------
    particles_x: List[Particle]
        List of N particles to match with <particles_y>
    particles_y: List[Particle]
        List of M particles to match with <particles_x>

    Note the correspondence particles_x -> N and particles_y -> M.

    This function can match two arbitrary points clouds, hence
    there is no need for the two particle lists to share the same
    voxels. 

    In particular, this could be used to match TruthParticle with Particles
    using true nonghost coordinates. In this case, <particles_x> must be the
    list of TruthParticles and <particles_y> the list of Particles. 

    Returns
    -------
    overlap_matrix: (M, N) np.float array, with range [0, 1]
    """
    overlap_matrix = np.zeros((len(particles_y), len(particles_x)), dtype=np.float32)
    for i, py in enumerate(particles_y):
        for j, px in enumerate(particles_x):
            if mode == 'default':
                dist = cdist(px.points, py.points)
            elif mode == 'true_nonghost':
                if type(px) == TruthParticle and type(py) == Particle:
                    dist = cdist(px.truth_points, py.points)
                elif type(px) == Particle and type(py) == TruthParticle:
                    dist = cdist(px.points, py.truth_points)
                elif type(px) == Particle and type(py) == Particle:
                    dist = cdist(px.points, py.points)
                else:
                    dist = cdist(px.truth_points, py.truth_points)
            else:
                raise ValueError('Particle overlap computation mode {} is not implemented!'.format(mode))
            loss_x = np.min(dist, axis=0)
            loss_y = np.min(dist, axis=1)
            loss = loss_x.sum() / loss_x.shape[0] + loss_y.sum() / loss_y.shape[0]
            overlap_matrix[i, j] = loss
    return overlap_matrix


def weighted_matrix_iou(particles_x, particles_y):
    """Function for computing the IoU matrix, where each IoU value is
    weighted by the factor w = (|size_x + size_y| / (|size_x - size_y| + 1).

    Parameters
    ----------
    particles_x: List[Particle]
        List of N particles to match with <particles_y>
    particles_y: List[Particle]
        List of M particles to match with <particles_x>

    Returns
    -------
    overlap_matrix: np.ndarray
        (M, N) array of IoU values
    cost_matrix: np.ndarray
        (M, N) array of weighted IoU values. 
    """
    overlap_matrix = np.zeros((len(particles_y), len(particles_x)), dtype=np.float32)
    cost_matrix = np.zeros_like(overlap_matrix)
    for i, py in enumerate(particles_y):
        for j, px in enumerate(particles_x):
            cap = np.intersect1d(py.index, px.index)
            cup = np.union1d(py.index, px.index)
            n, m = px.index.shape[0], py.index.shape[0]
            overlap_matrix[i, j] = (float(cap.shape[0]) / float(cup.shape[0]))
            w = float(abs(n+m)) / float(1.0 + abs(n-m))
            cost_matrix[i,j] = overlap_matrix[i,j] * w
    return overlap_matrix, cost_matrix


def match_particles_fn(particles_from : Union[List[Particle], List[TruthParticle]],
                       particles_to   : Union[List[Particle], List[TruthParticle]],
                       min_overlap=0, num_classes=5, verbose=False, overlap_mode='iou'):
    '''
    Match each Particle in <pred_particles> to <truth_particles>
    The number of matches will be equal to the length of <pred_particles>.

    Parameters
    ----------
    particles_from: List[Particle] or List[TruthParticle]
        List of particles to loop over during matching procedure.
    particles_to: List[Particle] or List[TruthParticle]
        List of particles to match a given particle from <particles_from>.

    min_overlap: int, float, or List[int]/List[float]
        Minimum required overlap value (float for IoU, int for counts)
        for a valid particle-particle match pair.

        If min_overlap is a list with same length as <num_classes>,
        a minimum overlap value will be applied separately
        for different classes.

        Example
        -------
        match_particles_fn(parts_from, parts_to,
                           min_overlap=[0.9, 0.9, 0.99, 0.99],
                           num_classes=4)
        -> This applies a minimum overlap cut of 0.9 IoU for class labels 0
        and 1, and a cut of 0.99 IoU for class labels 2 and 3.

    num_classes: int
        Total number of semantic classes (or any other label).
        This is used for setting <min_overlap> to differ across different
        semantic labels, for example.

    verbose: bool
        If True, print a message when a given particle has no match.

    overlap_mode: str
        Supported modes:

        'iou': overlap matrix is constructed from computing the
        intersection-over-union metric.

        'counts': overlap matrix is constructed from counting the number
        of shared voxels.


    Returns
    -------
    matches: List[Tuple[Particle, Particle]]
        List of tuples, indicating the matched particles.
        In case of no valid matches, a particle is matched with None
    idx: np.array of ints
        Index of matched particles
    intersections: np.array of floats/ints
        IoU/Count information for each matches.
    '''

    particles_x, particles_y = particles_from, particles_to

    if isinstance(min_overlap, float) or isinstance(min_overlap, int):
        thresholds = {key : min_overlap for key in np.arange(num_classes)}
    else:
        assert len(min_overlap) == num_classes
        thresholds = {key : val for key, val in zip(np.arange(num_classes), min_overlap)}

    if len(particles_y) == 0 or len(particles_x) == 0:
        if verbose:
            print("No particles to match.")
        return [], [0]

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(particles_x, particles_y)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(particles_x, particles_y)
    else:
        raise ValueError("Overlap matrix mode {} is not supported.".format(overlap_mode))
    # print(overlap_matrix)
    idx = overlap_matrix.argmax(axis=0)
    intersections = np.atleast_1d(overlap_matrix.max(axis=0))

    matches = []

    for j, px in enumerate(particles_x):
        select_idx = idx[j]
        if intersections[j] <= thresholds[px.pid]:
            # If no truth could be matched, assign None
            matched_truth = None
        else:
            matched_truth = particles_y[select_idx]
            # px._match.append(matched_truth.id)
            px._match_counts[matched_truth.id] = intersections[j]
            # matched_truth._match.append(px.id)
            matched_truth._match_counts[px.id] = intersections[j]
        matches.append((px, matched_truth))

    # for p in particles_y:
    #     p._match = sorted(list(p._match_counts.keys()), key=lambda x: p._match_counts[x],
    #                               reverse=True)

    return matches, intersections


def match_particles_optimal(particles_from : Union[List[Particle], List[TruthParticle]],
                            particles_to   : Union[List[Particle], List[TruthParticle]],
                            min_overlap=0, 
                            num_classes=5, 
                            verbose=False, 
                            overlap_mode='iou'):
    '''
    Match particles so that the final resulting sum of the overlap matrix
    is optimal. 

    The number of matches will be equal to length of the longer list.
    '''
    
    if len(particles_from) <= len(particles_to):
        particles_x, particles_y = particles_from, particles_to
    else:
        particles_y, particles_x = particles_from, particles_to

    if isinstance(min_overlap, float) or isinstance(min_overlap, int):
        thresholds = {key : min_overlap for key in np.arange(num_classes)}
    else:
        assert len(min_overlap) == num_classes
        thresholds = {key : val for key, val in zip(np.arange(num_classes), min_overlap)}

    if len(particles_y) == 0 or len(particles_x) == 0:
        if verbose:
            print("No particles to match.")
        return [], [0]

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(particles_y, particles_x)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(particles_y, particles_x)
    elif overlap_mode == 'chamfer':
        overlap_matrix = -matrix_chamfer(particles_y, particles_x)
    else:
        raise ValueError("Overlap matrix mode {} is not supported.".format(overlap_mode))

    matches, intersections = [], []

    ix, iy = linear_sum_assignment(overlap_matrix, maximize=True)
    
    mapping = dict(zip(iy, ix)) # iy is the index over the larger dimension

    for j in np.arange(overlap_matrix.shape[1]):
        i = mapping.get(j, None)
        match = (None, None)
        if i is None:
            match = (None, particles_y[j])
        else:
            overlap = overlap_matrix[i, j]
            intersections.append(overlap)
            # particles_y[j]._match.append(particles_x[i].id)
            # particles_x[i]._match.append(particles_y[j].id)
            particles_y[j]._match_counts[particles_x[i].id] = overlap
            particles_x[i]._match_counts[particles_y[j].id] = overlap
            match = (particles_x[i], particles_y[j])

        # Always place TruthParticle at front, for consistentcy with
        # selection scripts
        if (type(match[0]) is Particle) or (type(match[1]) is TruthParticle):
            p1, p2 = match[1], match[0]
            match = (p1, p2)
        matches.append(match)

    intersections = np.array(intersections)

    return matches, intersections


def match_interactions_fn(ints_from : List[Interaction],
                          ints_to : List[Interaction],
                          min_overlap=0, verbose=False, overlap_mode="iou"):
    """
    Same as <match_particles_fn>, but for lists of interactions.
    """
    ints_x, ints_y = ints_from, ints_to

    if len(ints_y) == 0 or len(ints_x) == 0:
        if verbose:
            print("No particles/interactions to match.")
        return [], 0

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(ints_x, ints_y)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(ints_x, ints_y)
    else:
        raise ValueError("Overlap matrix mode {} is not supported.".format(overlap_mode))
    
    idx = overlap_matrix.argmax(axis=0)
    intersections = overlap_matrix.max(axis=0)

    matches = []

    for j, interaction in enumerate(ints_x):
        select_idx = idx[j]
        if intersections[j] <= min_overlap:
            # If no truth could be matched, assign None
            matched_truth = None
        else:
            matched_truth = ints_y[select_idx]
            # interaction._match.append(matched_truth.id)
            interaction._match_counts[matched_truth.id] = intersections[j]
            # matched_truth._match.append(interaction.id)
            matched_truth._match_counts[interaction.id] = intersections[j]
            match = (interaction, matched_truth)
            matches.append(match)

        # if (type(match[0]) is Interaction) or (type(match[1]) is TruthInteraction):
        #     p1, p2 = match[1], match[0]
        #     match = (p1, p2)
        # matches.append(match)

    return matches, intersections


def match_interactions_optimal(ints_from : List[Interaction],
                               ints_to : List[Interaction],
                               min_overlap=0, verbose=False, overlap_mode="iou"):
    
    if len(ints_from) <= len(ints_to):
        ints_x, ints_y = ints_from, ints_to
    else:
        ints_y, ints_x = ints_from, ints_to

    if len(ints_y) == 0 or len(ints_x) == 0:
        if verbose:
            print("No particles/interactions to match.")
        return [], 0

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(ints_y, ints_x)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(ints_y, ints_x)
    elif overlap_mode == 'chamfer':
        overlap_matrix = -matrix_iou(ints_y, ints_x)
    else:
        raise ValueError("Overlap matrix mode {} is not supported.".format(overlap_mode))

    matches, intersections = [], []

    ix, iy = linear_sum_assignment(overlap_matrix, maximize=True)
    mapping = dict(zip(iy, ix)) # iy is the index over the larger dimension

    for j in np.arange(overlap_matrix.shape[1]):
        i = mapping.get(j, None)
        match = (None, None)
        if i is None:
            match = (None, ints_y[j])
            intersections.append(-1)
        else:
            overlap = overlap_matrix[i, j]
            intersections.append(overlap)
            # ints_y[j]._match.append(ints_x[i].id)
            # ints_x[i]._match.append(ints_y[j].id)
            ints_y[j]._match_counts[ints_x[i].id] = overlap
            ints_x[i]._match_counts[ints_y[j].id] = overlap
            match = (ints_x[i], ints_y[j])

        # Always place TruthParticle at front, for consistentcy with
        # selection scripts
        if (type(match[0]) is Interaction) or (type(match[1]) is TruthInteraction):
            p1, p2 = match[1], match[0]
            match = (p1, p2)
        matches.append(match)

    intersections = np.array(intersections)

    return matches, intersections


def group_particles_to_interactions_fn(particles : List[Particle],
                                       get_nu_id=False, 
                                       mode='pred',
                                       verbose=False):
    """
    Function for grouping particles to its parent interactions.

    Parameters
    ----------
    particles: List[Particle]
        List of Particle instances to construct Interaction instances from.
    get_nu_id: bool
        Option to retrieve neutrino_id (unused)
    mode: str
        Supported modes:
        'pred': output list will contain <Interaction> instances
        'truth': output list will contain <TruthInteraction> instances.

        Do not mix predicted interactions with TruthInteractions and
        interactions constructed from using labels with Interactions.
    """
    interactions = defaultdict(list)
    for p in particles:
        interactions[p.interaction_id].append(p)
        
    for int_id, particles in interactions.items():
        if mode == 'pred':
            interactions[int_id] = Interaction.from_particles(particles)
        elif mode == 'truth':
            interactions[int_id] = TruthInteraction.from_particles(particles)
        else:
            raise ValueError(f"Unknown aggregation mode {mode}.")

    # nu_id = -1
    # for int_id, particles in interactions.items():
    #     if get_nu_id:
    #         nu_id = np.unique([p.nu_id for p in particles])
    #         if nu_id.shape[0] > 1:
    #             if verbose:
    #                 print("Interaction {} has non-unique particle "\
    #                     "nu_ids: {}".format(int_id, str(nu_id)))
    #             nu_id = nu_id[0]
    #         else:
    #             nu_id = nu_id[0]

    #     counter = Counter([p.volume_id for p in particles if p.volume_id != -1])
    #     if not bool(counter):
    #         volume_id = -1
    #     else:
    #         volume_id = counter.most_common(1)[0][0]
    #     particles_dict = OrderedDict({p.id : p for p in particles})
    #     if mode == 'pred':
    #         interactions[int_id] = Interaction(int_id, particles_dict.values(), nu_id=nu_id, volume_id=volume_id)
    #     elif mode == 'truth':
    #         interactions[int_id] = TruthInteraction(int_id, particles_dict.values(), nu_id=nu_id, volume_id=volume_id)
    #     else:
    #         raise ValueError
        

    return list(interactions.values())


def check_particle_matches(loaded_particles, clear=False):
    match_dict = OrderedDict({})
    for p in loaded_particles:
        for i, m in enumerate(p.match):
            match_dict[int(m)] = p.match_counts[i]
        if clear:
            p._match = []
            p._match_counts = OrderedDict()

    match_counts = np.array(list(match_dict.values()))
    match = np.array(list(match_dict.keys())).astype(int)
    perm = np.argsort(match_counts)[::-1]
    match_counts = match_counts[perm]
    match = match[perm]

    return match, match_counts


def match_particles_recursive(particles_x, particles_y,
                              min_overlap=0.0):
    """Match particle using the optimal linear assignment method.
    
    Once the initial optimal assignments are found, the remaining 
    unmatched particles will undergo multiple rounds of matching against
    their counterparts until all remaining particles cannot be matched to
    any entities (zero overlap). 

    Parameters
    ----------
    particles_x : List[Union[Particle, TruthParticle]]
        List of Particles/TruthParticles to perform matching
    particles_y : List[Union[TruthParticle, Particle]]
        List of TruthParticles/Particles to perform matching

    Returns
    -------
    matches: List[Tuple(TruthParticle, Particle)]
        List of matched TruthParticle, Particle pairs. 
    intersections: Dict[Tuple(int, int), float]
        Dict of matched particle ids (keys) and match IoU values (values). 
    """
    
    particles_less = [p for p in particles_x if p.size > 0]
    particles_many = [p for p in particles_y if p.size > 0]
    
    if len(particles_x) <= len(particles_y):
        particles_less, particles_many = particles_x, particles_y
    else:
        particles_less, particles_many = particles_y, particles_x
    
    if len(particles_less) == 0 or len(particles_many) == 0:
        return [], {}
    
    overlap_matrix, cost_matrix = weighted_matrix_iou(particles_many, particles_less)
    
    matches = []
    ix, iy = linear_sum_assignment(cost_matrix, maximize=True)
    
    mapping = dict(zip(ix, iy)) # iy is the index over the larger dimension
    intersections = {}
    
    less_ids = {p.id: p for p in particles_less}
    many_ids = {p.id: p for p in particles_many}
    
    for i in np.arange(overlap_matrix.shape[0]):
        j = mapping[i]
        val = overlap_matrix[i,j]
        if val > min_overlap:
            if type(particles_less[i]) is Particle:
                match = (particles_less[i], particles_many[j])
                key = (particles_less[i].id, particles_many[j].id)
            elif type(particles_less[i]) is TruthParticle:
                match = (particles_many[j], particles_less[i])
                key = (particles_many[j].id, particles_less[i].id)
            else:
                msg = "Some entries in your input lists is neither a "\
                    "Particle nor a TruthParticle."
                raise ValueError(msg)
            matches.append(match)
            particles_less[i]._match.append(particles_many[j].id)
            particles_many[j]._match.append(particles_less[i].id)
            particles_less[i]._match_counts[particles_many[j].id] = val
            particles_many[j]._match_counts[particles_less[i].id] = val
            less_ids[particles_less[i].id].matched = True
            many_ids[particles_many[j].id].matched = True
            intersections[key] = val
            
    if len(matches) == 0:
        # All particles in domain have no viable match
        for part_id in less_ids:
            if type(less_ids[part_id]) is TruthParticle:
                matches.append((less_ids[part_id], None))
            elif type(less_ids[part_id]) is Particle:
                matches.append((None, less_ids[part_id]))
            else:
                msg = "Some entries in your input lists is neither a "\
                    "Particle nor a TruthParticle."
                raise ValueError(msg)
        return matches, intersections
            
    unmatched_less = [p for p in particles_less if not less_ids[p.id].matched]
    nested_matches, nested_ints = match_particles_recursive(unmatched_less, particles_many)
    
    matches.extend(nested_matches)
    intersections.update(nested_ints)
    
    unmatched_many = [p for p in particles_many if not many_ids[p.id].matched]
    nested_matches, nested_ints = match_particles_recursive(unmatched_many, particles_less)
    
    matches.extend(nested_matches)
    intersections.update(nested_ints)
            
    return matches, intersections