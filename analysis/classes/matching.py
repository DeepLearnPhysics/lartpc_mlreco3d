import numpy as np
import numba as nb
from numba.typed import List

from typing import List, Union
from collections import defaultdict, OrderedDict, Counter

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from . import Particle, TruthParticle, Interaction, TruthInteraction


class VoxelMatcher:

    def __init__(self, metric='dice', algorithm='argmax'):

        self._metric_name = metric
        self._algorithm_name = algorithm

        self._match_fn = None
        self._value_matrix_fn = None
        self._weight_fn = None

# --------------------------Helper Functions--------------------------

def value_matrix_dict():

    out = {
        'counts': matrix_counts,
        'iou': matrix_iou,
        'weighted_iou': weighted_matrix_iou,
        'weightd_dice_nb': weighted_matrix_dice
    }

    return out


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


def weighted_matrix_iou(particles_x, particles_y, weight=False):
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
            if weight:
                cost_matrix[i,j] = overlap_matrix[i,j] * w
            else:
                cost_matrix[i,j] = overlap_matrix[i,j]
    return overlap_matrix, cost_matrix


def weighted_matrix_dice(particles_x, particles_y):
    index_x = List([p for p in particles_x])
    index_y = List([p for p in particles_y])
    mat = _weighted_matrix_dice(index_x, index_y)
    return mat


@nb.njit(cache=True)
def _weighted_matrix_dice(index_x : List[nb.int64[:]],
                          index_y : List[nb.int64[:]]) -> nb.float32[:,:]:
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i, py in enumerate(index_x):
        for j, px in enumerate(index_y):
            cap = np.intersect1d(py, px)
            cup = len(py) + len(px)
            w = (len(px) + len(py)) / (1 + np.abs(len(px) - len(py)))
            overlap_matrix[i, j] = (2.0 * float(cap.shape[0]) / float(cup)) * w
    return overlap_matrix

def match_particles_fn(particles_x : Union[List[Particle], List[TruthParticle]],
                       particles_y : Union[List[Particle], List[TruthParticle]],
                       value_matrix: np.ndarray,
                       overlap_matrix: np.ndarray,
                       min_overlap=0.0):
    return match_particles_all(particles_x, 
                               particles_y,
                               value_matrix, 
                               overlap_matrix, 
                               min_overlap=min_overlap)


def match_particles_all(particles_x : Union[List[Particle], List[TruthParticle]],
                        particles_y : Union[List[Particle], List[TruthParticle]],
                        value_matrix: np.ndarray,
                        overlap_matrix: np.ndarray,
                        min_overlap=0.0):
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
    assert value_matrix.shape == (len(particles_y), len(particles_x))

    if not len(value_matrix.flatten()):
        return OrderedDict(), []

    #idx = value_matrix.argmax(axis=0)
    #intersections = np.atleast_1d(value_matrix.max(axis=0))

    matches = OrderedDict()
    out_counts = []
    
    for px in particles_x:
        px.match_overlap = OrderedDict()
    #for py in particles_y:
    #    py.match_overlap = OrderedDict()

    # For each particle in x, choose one in y
    for j, px in enumerate(particles_x):
        #select_idx = idx[j]
        match_idxs = np.where(overlap_matrix[:,j] > min_overlap)[0]
        out_counts.append(match_idxs)
        if not len(match_idxs):
            key = (px.id, None)
            matches[key] = (px, None)
            px.matched = False
        else:
            px.matched = True
            for idx in match_idxs:
                matched = particles_y[idx]
                px._match_overlap[matched.id] = overlap_matrix[idx,j]
                # matched._match_overlap[px.id] = intersections[j]
                key = (px.id, matched.id)
                matches[key] = (px, matched)

    out_counts = np.array(out_counts)

    return matches, out_counts


def match_particles_principal(particles_x : Union[List[Particle], List[TruthParticle]],
                       particles_y : Union[List[Particle], List[TruthParticle]],
                       value_matrix: np.ndarray,
                       overlap_matrix: np.ndarray,
                       min_overlap=0.0):
    '''
    Same as <match_particles_fn>, but only keeps principal matches.
    '''
    assert value_matrix.shape == (len(particles_y), len(particles_x))

    if not len(value_matrix.flatten()):
        return OrderedDict(), []

    idx = value_matrix.argmax(axis=0)
    intersections = np.atleast_1d(value_matrix.max(axis=0))

    matches = OrderedDict()
    out_counts = []
    
    for px in particles_x:
        px.match_overlap = OrderedDict()
    #for py in particles_y:
    #    py.match_overlap = OrderedDict()

    # For each particle in x, choose one in y
    for j, px in enumerate(particles_x):
        select_idx = idx[j]
        out_counts.append(overlap_matrix[select_idx, j])
        if intersections[j] <= min_overlap:
            key = (px.id, None)
            matches[key] = (px, None)
            px.matched = False
        else:
            matched = particles_y[select_idx]
            px._match_overlap[matched.id] = intersections[j]
            # matched._match_overlap[px.id] = intersections[j]
            key = (px.id, matched.id)
            matches[key] = (px, matched)
            px.matched = True

    out_counts = np.array(out_counts)

    return matches, out_counts


def match_interactions_fn(ints_x : List[Interaction],
                          ints_y : List[Interaction],
                          value_matrix: np.ndarray,
                          overlap_matrix: np.ndarray,
                          min_overlap=0):
    """
    Same as <match_particles_fn>, but for lists of interactions.
    """
    return match_particles_fn(ints_x, ints_y,
                              value_matrix=value_matrix,
                              overlap_matrix=overlap_matrix,
                              min_overlap=min_overlap)


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
    # Sort the particles by interactions
    interactions = []
    interaction_ids = np.array([p.interaction_id for p in particles])
    for i, int_id in enumerate(np.unique(interaction_ids)):
        # Get particles in interaction int_it
        particle_ids = np.where(interaction_ids == int_id)[0]
        parts = [particles[i] for i in particle_ids]

        # Build interactions
        if mode == 'pred':
            interaction = Interaction.from_particles(parts)
            interaction.id = i
        elif mode == 'truth':
            interaction = TruthInteraction.from_particles(parts)
            interaction.id = i
            interaction.truth_id = int_id
        else:
            raise ValueError(f"Unknown aggregation mode {mode}.")

        # Reset the interaction ID of the constiuent particles
        for j in particle_ids:
            particles[j].interaction_id = i

        # Append
        interactions.append(interaction)

    return interactions


def check_particle_matches(loaded_particles, clear=False):
    match_dict = OrderedDict({})
    for p in loaded_particles:
        for i, m in enumerate(p.match):
            match_dict[int(m)] = p.match_overlap[i]
        if clear:
            p._match = []
            p._match_overlap = OrderedDict()

    match_overlap = np.array(list(match_dict.values()))
    match = np.array(list(match_dict.keys())).astype(int)
    perm = np.argsort(match_overlap)[::-1]
    match_overlap = match_overlap[perm]
    match = match[perm]

    return match, match_overlap

def generate_match_pairs(truth, reco, prefix='matches', only_principal=False):
    out = {
        prefix+'_t2r': [],
        prefix+'_r2t': [],
        prefix+'_t2r_values': [],
        prefix+'_r2t_values': []
    }
    true_dict = {p.id : p for p in truth}
    reco_dict = {p.id : p for p in reco}

    for p in truth:
        if len(p.match) == 0:
            pair = (p, None)
            out[prefix+'_t2r'].append(pair)
            out[prefix+'_t2r_values'].append(-1)
            continue
        if only_principal:
            idxmax = np.argmax(p.match_overlap)
            reco_id = p.match[idxmax]
            pair = (p, reco_dict[reco_id])
            out[prefix+'_t2r'].append(pair)
            out[prefix+'_t2r_values'].append(p.match_overlap[idxmax])
        else:
            for i, reco_id in enumerate(p.match):
                pair = (p, reco_dict[reco_id])
                out[prefix+'_t2r'].append(pair)
                out[prefix+'_t2r_values'].append(p.match_overlap[i])
    for p in reco:
        if len(p.match) == 0:
            pair = (p, None)
            out[prefix+'_r2t'].append(pair)
            out[prefix+'_r2t_values'].append(-1)
            continue
        if only_principal:
            idxmax = np.argmax(p.match_overlap)
            true_id = p.match[idxmax]
            pair = (p, true_dict[true_id])
            out[prefix+'_r2t'].append(pair)
            out[prefix+'_r2t_values'].append(p.match_overlap[idxmax])
        else:
            for i, true_id in enumerate(p.match):
                pair = (p, true_dict[true_id])
                out[prefix+'_r2t'].append(pair)
                out[prefix+'_r2t_values'].append(p.match_overlap[i])
                
    return out
