import numpy as np
import pandas as pd

from typing import Counter, List, Union
from collections import defaultdict, Counter
from functools import partial
import re

from pprint import pprint


class Particle:
    '''
    Simple Particle Class with managable __repr__ and __str__ functions.
    '''
    def __init__(self, coords, group_id, semantic_type, interaction_id, 
                 pid, batch_id=0, voxel_indices=None, depositions=None, **kwargs):
        self.id = group_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions
        self.voxel_indices = voxel_indices
        self.semantic_type = semantic_type
        self.pid = pid
        self.pid_conf = kwargs.get('pid_conf', None)
        self.interaction_id = interaction_id
        self.batch_id = batch_id
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

        self.startpoint = -np.ones(3)
        self.endpoints = -np.ones((2, 3))
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        fmt = "Particle( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8} | Primary: {:<2} | Score = {:.2f}% | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.batch_id, self.id, 
                         self.semantic_keys[self.semantic_type], 
                         self.pid_keys[self.pid], 
                         self.is_primary,
                         self.pid_conf * 100,
                         self.interaction_id,
                         self.points.shape[0])
        return msg


class ParticleFragment(Particle):
    '''
    Reserved for particle fragments.
    '''
    def __init__(self, coords, fragment_id, semantic_type, interaction_id, 
                 group_id, batch_id=0, depositions=None, alias="Particle", **kwargs):
        self.id = fragment_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions
        self.semantic_type = semantic_type
        self.group_id = group_id
        self.interaction_id = interaction_id
        self.batch_id = batch_id
        self.is_primary = kwargs.get('is_primary', False)
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }

        self.startpoint = -np.ones(3)

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        fmt = "ParticleFragment( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | Group ID: {:<3} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.batch_id, self.id, 
                         self.semantic_keys[self.semantic_type], 
                         self.group_id, 
                         self.is_primary,
                         self.interaction_id,
                         self.points.shape[0])
        return msg


class TruthParticle(Particle):
    '''
    Reserved for true particles derived from true labels / true MC information.
    '''
    def __init__(self, *args, particle_asis=None, **kwargs):
        super(TruthParticle, self).__init__(*args, **kwargs)
        self.asis = particle_asis
        self.match = []
        self._match_counts = {}

    def __repr__(self):
        fmt = "TruthParticle( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.batch_id, self.id, 
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


class Interaction:

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
            self.vertex = [-1, -1, -1]

        self.nu_id = nu_id

        self.particle_ids = [p.id for p in self.particles]
        self.particle_counts = Counter({ i : 0 for i in range(len(self.pid_keys))})
        self.particle_counts.update([p.pid for p in self.particles])

        self.primary_particle_counts = Counter({ i : 0 for i in range(len(self.pid_keys))})
        self.primary_particle_counts.update([p.pid for p in self.particles if p.is_primary])

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
        msg = "Interaction {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "-----------------------------------------------\n".format(
            self.id, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __str__(self):
        return "Interaction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))


class TruthInteraction(Interaction):

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


def match(particles_from : Union[List[Particle], List[TruthParticle]], 
          particles_to   : Union[List[Particle], List[TruthParticle]], 
          primaries=True, min_overlap_count=1, verbose=False):
    '''
    Match each Particle in <pred_particles> to <truth_particles>
    The number of matches will be equal to the length of <pred_particles>. 
    
    '''

    particles_x, particles_y = particles_from, particles_to
    if primaries:
        particles_x, particles_y = [], []
        for px in particles_from:
            px.match = []
            px._match_counts = {}
            if px.is_primary:
                particles_x.append(px)
        for py in particles_to:
            py.match = []
            py._match_counts = {}
            if py.is_primary:
                particles_y.append(py)

    if len(particles_y) == 0 or len(particles_x) == 0:
        if verbose:
            print("No particles/interactions to match.")
        return [], 0, 0

    overlap_matrix = np.zeros((len(particles_y), len(particles_x)), dtype=np.int64)
    for i, py in enumerate(particles_y):
        for j, px in enumerate(particles_x):
            overlap_matrix[i, j] = len(np.intersect1d(py.voxel_indices, 
                                                      px.voxel_indices))

    idx = overlap_matrix.argmax(axis=0)
    intersections = overlap_matrix.max(axis=0)

    idx[intersections < min_overlap_count] = -1
    # intersections[intersections < min_overlap_count] = -1

    matches = []

    for j, px in enumerate(particles_x):
        select_idx = idx[j]
        if select_idx < 0:
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
                          min_overlap_count=1):
    
    f = partial(match, primaries=False, 
                min_overlap_count=min_overlap_count)
    
    return f(ints_from, ints_to)


def group_particles_to_interactions_fn(particles : List[Particle], 
                                       get_nu_id=False, mode='pred'):

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