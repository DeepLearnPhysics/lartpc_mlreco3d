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
                 pid, batch_id=0, depositions=None, **kwargs):
        self.id = group_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions
        self.semantic_type = semantic_type
        self.pid = pid
        self.pid_conf = kwargs.get('pid_conf', None)
        self.interaction_id = interaction_id
        self.batch_id = batch_id
        self.is_primary = kwargs.get('is_primary', False)
#         self.fragments = fragment_ids
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }
    
        self.pid_keys = {
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }

        self.startpoint = -np.ones(3)
        self.endpoints = -np.ones((2, 3))


    def get_info(self):

        names = ['pred_particle_id', 
                 'pred_particle_type', 
                 'pred_particle_is_primary', 
                 'pred_particle_size', 
                 'pred_particle_conf', 
                 'pred_particle_num_ppn_candidates', 
                 'pred_particle_startpoint_x', 
                 'pred_particle_startpoint_y', 
                 'pred_particle_startpoint_z',
                 'pred_particle_endpoint_1_x', 
                 'pred_particle_endpoint_1_y', 
                 'pred_particle_endpoint_1_z',
                 'pred_particle_endpoint_2_x', 
                 'pred_particle_endpoint_2_y', 
                 'pred_particle_endpoint_2_z']

        values = [
            self.id,
            self.pid,
            self.is_primary,
            self.size,
            self.pid_conf,
            self.ppn_candidates.shape[0],
            self.startpoint[0],
            self.startpoint[1],
            self.startpoint[2],
            self.endpoints[0, 0],
            self.endpoints[0, 1],
            self.endpoints[0, 2],
            self.endpoints[1, 0],
            self.endpoints[1, 1],
            self.endpoints[1, 2]
        ]

        return dict(zip(names, values))
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        fmt = "Particle( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8} | Conf = {:.2f}% | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.batch_id, self.id, 
                         self.semantic_keys[self.semantic_type], 
                         self.pid_keys[self.pid], 
                         self.pid_conf * 100,
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

    def __repr__(self):
        fmt = "TruthParticle( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8} | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.batch_id, self.id, 
                         self.semantic_keys[self.semantic_type], 
                         self.pid_keys[self.pid], 
                         self.interaction_id,
                         self.points.shape[0])
        return msg

    def get_info(self):

        names = ['true_particle_id', 
                 'true_particle_type', 
                 'true_particle_is_primary', 
                 'true_particle_size', 
                 'true_particle_startpoint_x', 
                 'true_particle_startpoint_y', 
                 'true_particle_startpoint_z',
                 'true_particle_endpoint_1_x', 
                 'true_particle_endpoint_1_y', 
                 'true_particle_endpoint_1_z',
                 'true_particle_endpoint_2_x', 
                 'true_particle_endpoint_2_y', 
                 'true_particle_endpoint_2_z']

        values = [
            self.id,
            self.pid,
            self.is_primary,
            self.size,
            self.startpoint[0],
            self.startpoint[1],
            self.startpoint[2],
            self.endpoints[0][0],
            self.endpoints[0][1],
            self.endpoints[0][2],
            self.endpoints[1][0],
            self.endpoints[1][1],
            self.endpoints[1][2]
        ]

        return dict(zip(names, values))


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

        self.particles_summary = ""
        for p in self.particles:
            pmsg = "    - Particle {}: PID = {}, Size = {} \n".format(
                p.id, self.pid_keys[p.pid], p.points.shape[0])
            self.particles_summary += pmsg

        self.vertex = vertex
        if self.vertex is None:
            self.vertex = [-1, -1, -1]

        self.nu_id = nu_id

        self.particle_ids = [p.id for p in self.particles]
        self.particle_counts = Counter({ i : 0 for i in range(len(self.pid_keys))})
        self.particle_counts.update([p.pid for p in self.particles])

    def check_validity(self):
        for p in self.particles:
            assert isinstance(p, Particle)

    def get_info(self): 

        names = ['pred_interaction_id', 
                 'pred_interaction_type', 
                 'pred_interaction_size', 
                 'pred_interaction_particle_counts',
                 'pred_interaction_count_photons', 
                 'pred_interaction_count_electrons', 
                 'pred_interaction_count_muons',
                 'pred_interaction_count_pions', 
                 'pred_interaction_count_protons', 
                 'pred_interaction_vtx_x',
                 'pred_interaction_vtx_y', 
                 'pred_interaction_vtx_z']

        values = [
            self.id,
            self.nu_id,
            self.size,
            self.num_particles,
            self.particle_counts[0],
            self.particle_counts[1],
            self.particle_counts[2],
            self.particle_counts[3],
            self.particle_counts[4],
            self.vertex[0],
            self.vertex[1],
            self.vertex[2]
        ]

        return dict(zip(names, values))


    def __repr__(self):

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

    def check_validity(self):
        for p in self.particles:
            assert isinstance(p, TruthParticle)

    def get_info(self): 

        names = ['true_interaction_id', 
                 'true_interaction_type', 
                 'true_interaction_size', 
                 'true_interaction_particle_counts',
                 'true_interaction_count_photons', 
                 'true_interaction_count_electrons', 
                 'true_interaction_count_muons',
                 'true_interaction_count_pions', 
                 'true_interaction_count_protons', 
                 'true_interaction_vtx_x',
                 'true_interaction_vtx_y', 
                 'true_interaction_vtx_z']

        values = [
            self.id,
            self.nu_id,
            self.size,
            self.num_particles,
            self.particle_counts[0],
            self.particle_counts[1],
            self.particle_counts[2],
            self.particle_counts[3],
            self.particle_counts[4],
            self.vertex[0],
            self.vertex[1],
            self.vertex[2]
        ]

        return dict(zip(names, values))

    def __repr__(self):

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
            if px.is_primary:
                particles_x.append(px)
        for py in particles_to:
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
    intersections[intersections < min_overlap_count] = -1

    matches = []

    for j, px in enumerate(particles_x):
        select_idx = idx[j]
        if select_idx < 0:
            # If no truth could be matched, assign None
            matched_truth = None
        else:
            matched_truth = particles_y[select_idx]
        matches.append((px, matched_truth))

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
                raise ValueError("Interaction {} has non-unique particle "\
                    "nu_ids: {}".format(int_id, str(nu_id)))
            else:
                nu_id = nu_id[0]
        if mode == 'pred':
            interactions[int_id] = Interaction(int_id, particles, nu_id=nu_id)
        elif mode == 'truth':
            interactions[int_id] = TruthInteraction(int_id, particles, nu_id=nu_id)
        else:
            raise ValueError
            
    return list(interactions.values())