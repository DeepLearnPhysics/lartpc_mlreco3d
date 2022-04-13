import numpy as np
import pandas as pd

from typing import Counter, List, Union
from collections import defaultdict, Counter
from functools import partial
import re

from pprint import pprint


class Particle:
    '''
    Data Structure for managing Particle-level 
    full chain output information

    Attributes
    ----------
    id: int
        Unique ID of the particle
    points: (N, 3) np.array
        3D coordinates of the voxels that belong to this particle
    size: int
        Total number of voxels that belong to this particle
    depositions: (N, 1) np.array
        Array of energy deposition values for each voxel
    voxel_indices: (N, ) np.array
        Numeric integer indices of voxel positions of this particle
        with respect to the total array of point in a single image.
    semantic_type: int
        Semantic type (shower fragment (0), track (1), 
        michel (2), delta (3), lowE (4)) of this particle. 
    pid: int
        PDG Type (Photon (0), Electron (1), Muon (2), 
        Charged Pion (3), Proton (4)) of this particle. 
    pid_conf: float
        Softmax probability score for the most likely pid prediction
    interaction_id: int
        Integer ID of the particle's parent interaction
    image_id: int
        ID of the image in which this particle resides in
    is_primary: bool
        Indicator whether this particle is a primary from an interaction.
    match: List[int]
        List of TruthParticle IDs for which this particle is matched to

    startpoint: (1,3) np.array 
        (1, 3) array of particle's startpoint, if it could be assigned
    endpoint: (1,3) np.array 
        (1, 3) array of particle's endpoint, if it could be assigned
    '''
    def __init__(self, coords, group_id, semantic_type, interaction_id,
                 pid, image_id=0, voxel_indices=None, depositions=None, **kwargs):
        self.id = group_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions
        self.voxel_indices = voxel_indices
        self.semantic_type = semantic_type
        self.pid = pid
        self.pid_conf = kwargs.get('pid_conf', None)
        self.interaction_id = interaction_id
        self.image_id = image_id
        self.is_primary = kwargs.get('is_primary', False)
        self.match = []
        self._match_counts = {}
#         self.fragments = fragment_ids
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }

        self.pid_keys = {
            -1: 'None',
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }

        self.sum_edep = np.sum(self.depositions)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        fmt = "Particle( Image ID={:<3} | Particle ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8} | Primary: {:<2} | Score = {:.2f}% | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type],
                         self.pid_keys[self.pid],
                         self.is_primary,
                         self.pid_conf * 100,
                         self.interaction_id,
                         self.points.shape[0])
        return msg


class ParticleFragment(Particle):
    '''
    Data structure for managing fragment-level 
    full chain output information

    Attributes
    ----------
    See <Particle> documentation for shared attributes.
    Below are attributes exclusive to ParticleFragment

    id: int
        fragment ID of this particle fragment (different from particle id)
    group_id: int
        Group ID (alias for Particle ID) for which this fragment belongs to.
    is_primary: bool
        If True, then this particle fragment corresponds to 
        a primary ionization trajectory within the group of fragments that
        compose a particle. 
    '''
    def __init__(self, coords, fragment_id, semantic_type, interaction_id,
                 group_id, image_id=0, voxel_indices=None,
                 depositions=None, **kwargs):
        self.id = fragment_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions
        self.voxel_indices = voxel_indices
        self.semantic_type = semantic_type
        self.group_id = group_id
        self.interaction_id = interaction_id
        self.image_id = image_id
        self.is_primary = kwargs.get('is_primary', False)
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        fmt = "ParticleFragment( Image ID={:<3} | Fragment ID={:<3} | Semantic_type: {:<15}"\
            " | Group ID: {:<3} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type],
                         self.group_id,
                         self.is_primary,
                         self.interaction_id,
                         self.points.shape[0])
        return msg


class TruthParticle(Particle):
    '''
    Data structure mirroring <Particle>, reserved for true particles 
    derived from true labels / true MC information.

    Attributes
    ----------
    See <Particle> documentation for shared attributes.
    Below are attributes exclusive to TruthParticle

    asis: larcv.Particle C++ object (Optional)
        Raw larcv.Particle C++ object as retrived from parse_particles_asis.
    match: List[int]
        List of Particle IDs that match to this TruthParticle
    '''
    def __init__(self, *args, particle_asis=None, coords_noghost=None, depositions_noghost=None, **kwargs):
        super(TruthParticle, self).__init__(*args, **kwargs)
        self.asis = particle_asis
        self.match = []
        self._match_counts = {}
        self.coords_noghost = coords_noghost
        self.depositions_noghost = depositions_noghost

    def __repr__(self):
        fmt = "TruthParticle( Image ID={:<3} | Particle ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type],
                         self.pid_keys[self.pid],
                         self.is_primary,
                         self.interaction_id,
                         self.points.shape[0])
        return msg


    def is_contained(self, spatial_size):

        p = self.particle_asis
        check_contained = p.position().x() >= 0 and p.position().x() <= spatial_size \
            and p.position().y() >= 0 and p.position().y() <= spatial_size \
            and p.position().z() >= 0 and p.position().z() <= spatial_size \
            and p.end_position().x() >= 0 and p.end_position().x() <= spatial_size \
            and p.end_position().y() >= 0 and p.end_position().y() <= spatial_size \
            and p.end_position().z() >= 0 and p.end_position().z() <= spatial_size
        return check_contained

    def purity_efficiency(self, other_particle):
        overlap = len(np.intersect1d(self.voxel_indices, other_particle.voxel_indices))
        return {
            "purity": overlap / len(other_particle.voxel_indices),
            "efficiency": overlap / len(self.voxel_indices)
        }

class Interaction:
    """
    Data structure for managing interaction-level
    full chain output information. 

    Attributes
    ----------
    id: int
        Unique ID (Interaction ID) of this interaction.
    particles: List[Particle]
        List of <Particle> objects that belong to this Interaction.
    vertex: (1,3) np.array (Optional)
        3D coordinates of the predicted interaction vertex
    nu_id: int (Optional, TODO)
        Label indicating whether this interaction is a neutrino interaction
        WARNING: The nu_id label is most likely unreliable. Don't use this
        in reconstruction (used for debugging)
    num_particles: int 
        total number of particles in this interaction. 
    """
    def __init__(self, interaction_id, particles, vertex=None, nu_id=-1):
        self.id = interaction_id
        self.particles = particles
        self.match = []
        self._match_counts = {}
        self.check_validity()
        # Voxel indices of an interaction is defined by the union of
        # constituent particle voxel indices
        self.voxel_indices = []
        for p in self.particles:
            self.voxel_indices.append(p.voxel_indices)
            assert p.interaction_id == interaction_id
        self.voxel_indices = np.hstack(self.voxel_indices)
        self.size = self.voxel_indices.shape[0]
        self.num_particles = len(self.particles)

        self.pid_keys = {
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }

        self.get_particles_summary()

        self.vertex = vertex
        if self.vertex is None:
            self.vertex = np.array([-1, -1, -1])

        self.nu_id = nu_id

        self.particle_ids = [p.id for p in self.particles]
        self.particle_counts = Counter({ i : 0 for i in range(len(self.pid_keys))})
        self.particle_counts.update([p.pid for p in self.particles])

        self.primary_particle_counts = Counter({ i : 0 for i in range(len(self.pid_keys))})
        self.primary_particle_counts.update([p.pid for p in self.particles if p.is_primary])

        if sum(self.primary_particle_counts.values()) == 0:
            print("Interaction {} has no primary particles!".format(self.id))
            self.is_valid = False
        else:
            self.is_valid = True

    def check_validity(self):
        for p in self.particles:
            assert isinstance(p, Particle)

    def get_particles_summary(self):
        self.particles_summary = ""
        self.particles = sorted(self.particles, key=lambda x: x.id)
        for p in self.particles:
            pmsg = "    - Particle {}: PID = {}, Size = {}, Match = {} \n".format(
                p.id, self.pid_keys[p.pid], p.points.shape[0], str(p.match))
            self.particles_summary += pmsg


    def __repr__(self):

        self.get_particles_summary()
        msg = "Interaction {}, Valid: {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "--------------------------------------------------------------------\n".format(
            self.id, self.is_valid, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __str__(self):
        return "Interaction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))


class TruthInteraction(Interaction):
    """
    Analogous data structure for Interactions retrieved from true labels. 
    """
    def __init__(self, *args, **kwargs):
        super(TruthInteraction, self).__init__(*args, **kwargs)
        self.match = []
        self._match_counts = {}

    def check_validity(self):
        for p in self.particles:
            assert isinstance(p, TruthParticle)

    def __repr__(self):

        self.get_particles_summary()
        msg = "TruthInteraction {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "-----------------------------------------------\n".format(
            self.id, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __str__(self):
        return "TruthInteraction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))


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
        return [], 0, 0

    if overlap_mode == 'counts':
        overlap_matrix = matrix_counts(particles_x, particles_y)
    elif overlap_mode == 'iou':
        overlap_matrix = matrix_iou(particles_x, particles_y)
    else:
        raise ValueError("Overlap matrix mode {} is not supported.".format(overlap_mode))
    # print(overlap_matrix)
    idx = overlap_matrix.argmax(axis=0)
    intersections = overlap_matrix.max(axis=0)

    # idx[intersections < min_overlap] = -1
    # intersections[intersections < min_overlap_count] = -1

    matches = []
    # print(thresholds)

    for j, px in enumerate(particles_x):
        select_idx = idx[j]
        # print(px.semantic_type, px.id, particles_y[select_idx].semantic_type, particles_y[select_idx].id, intersections[j])
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

    return matches, idx, intersections


def match_interactions_fn(ints_from : List[Interaction],
                          ints_to : List[Interaction],
                          min_overlap=0, verbose=False):
    """
    Same as <match_particles_fn>, but for lists of interactions. 
    """
    ints_x, ints_y = ints_from, ints_to

    if len(ints_y) == 0 or len(ints_x) == 0:
        if verbose:
            print("No particles/interactions to match.")
        return [], 0, 0

    overlap_matrix = matrix_iou(ints_x, ints_y)
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

    return matches, idx, intersections


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
        if mode == 'pred':
            interactions[int_id] = Interaction(int_id, particles, nu_id=nu_id)
        elif mode == 'truth':
            interactions[int_id] = TruthInteraction(int_id, particles, nu_id=nu_id)
        else:
            raise ValueError

    return list(interactions.values())
