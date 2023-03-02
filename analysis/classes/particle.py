import numpy as np
import pandas as pd

from typing import Counter, List, Union
from collections import defaultdict, OrderedDict
from functools import partial
import re

from scipy.optimize import linear_sum_assignment

from pprint import pprint

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
            overlap_matrix[i, j] = len(np.intersect1d(py.voxel_indices,
                                                      px.voxel_indices))
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
            cap = np.intersect1d(py.voxel_indices, px.voxel_indices)
            cup = np.union1d(py.voxel_indices, px.voxel_indices)
            overlap_matrix[i, j] = float(cap.shape[0] / cup.shape[0])
    return overlap_matrix


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
        return [], 0

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(particles_x, particles_y)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(particles_x, particles_y)
    else:
        raise ValueError("Overlap matrix mode {} is not supported.".format(overlap_mode))
    # print(overlap_matrix)
    idx = overlap_matrix.argmax(axis=0)
    intersections = overlap_matrix.max(axis=0)

    matches = []

    for j, px in enumerate(particles_x):
        select_idx = idx[j]
        if intersections[j] <= thresholds[px.pid]:
            # If no truth could be matched, assign None
            matched_truth = None
        else:
            matched_truth = particles_y[select_idx]
            px.match.append(matched_truth.id)
            px._match_counts[matched_truth.id] = intersections[j]
            matched_truth.match.append(px.id)
            matched_truth._match_counts[px.id] = intersections[j]
        matches.append((px, matched_truth))

    for p in particles_y:
        p.match = sorted(p.match, key=lambda x: p._match_counts[x],
                                  reverse=True)

    return matches, intersections


def match_particles_optimal(particles_from : Union[List[Particle], List[TruthParticle]],
                            particles_to   : Union[List[Particle], List[TruthParticle]],
                            min_overlap=0, num_classes=5, verbose=False, overlap_mode='iou'):
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
        return [], 0

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(particles_y, particles_x)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(particles_y, particles_x)
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
            particles_y[j].match.append(particles_x[i].id)
            particles_x[i].match.append(particles_y[j].id)
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
            interaction.match.append(matched_truth.id)
            interaction._match_counts[matched_truth.id] = intersections[j]
            matched_truth.match.append(interaction.id)
            matched_truth._match_counts[interaction.id] = intersections[j]
        matches.append((interaction, matched_truth))

    for interaction in ints_y:
        interaction.match = sorted(interaction.match,
                                   key=lambda x: interaction._match_counts[x],
                                   reverse=True)

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
            ints_y[j].match.append(ints_x[i].id)
            ints_x[i].match.append(ints_y[j].id)
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
                                       get_nu_id=False, mode='pred'):
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

    nu_id = -1
    for int_id, particles in interactions.items():
        if get_nu_id:
            nu_id = np.unique([p.nu_id for p in particles])
            if nu_id.shape[0] > 1:
                print("Interaction {} has non-unique particle "\
                    "nu_ids: {}".format(int_id, str(nu_id)))
                nu_id = nu_id[0]
            else:
                nu_id = nu_id[0]
        particles_dict = OrderedDict({p.id : p for p in particles})
        if mode == 'pred':
            interactions[int_id] = Interaction(int_id, particles_dict, nu_id=nu_id)
        elif mode == 'truth':
            interactions[int_id] = TruthInteraction(int_id, particles_dict, nu_id=nu_id)
        else:
            raise ValueError

    return list(interactions.values())
