import numpy as np
import pandas as pd

from typing import Counter, List
from collections import defaultdict, Counter
from functools import partial
class Particle:
    '''
    Simple Particle Class with managable __repr__ and __str__ functions.
    '''
    def __init__(self, coords, group_id, semantic_type, interaction_id, 
                 pid, batch_id=0, depositions=None, **kwargs):
        self.id = group_id
        self.points = coords
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

        self.startpoint = None
        self.endpoints = None
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        fmt = "Particle( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8}, Conf = {:.2f}% | Interaction ID: {:<2} | Size: {:<5} )"
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


class Interaction:

    def __init__(self, interaction_id, particles, vertex=None):
        self.id = interaction_id
        self.particles = particles

        # Voxel indices of an interaction is defined by the union of
        # constituent particle voxel indices
        self.voxel_indices = []
        for p in self.particles:
            self.voxel_indices.append(p.voxel_indices)
            assert p.interaction_id == interaction_id
        self.voxel_indices = np.hstack(self.voxel_indices)

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
            self.vertex = [None, None, None]

        self.particle_ids = [p.id for p in self.particles]

        self.counter = Counter([p.pid for p in self.particles])


    def __repr__(self):

        msg = "Interaction {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "-----------------------------------------------\n".format(
            self.id, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __str__(self):
        return "Interaction(id={}, vertex={}, Particles={})".format(
            self.id, str(self.vertex), str(self.particle_ids))


def match_particles_fn(pred_particles  : List[Particle], 
                       truth_particles : List[TruthParticle], 
                       primaries=True, min_overlap_count=1, relabel=False):

    part_p, part_t = pred_particles, truth_particles
    if primaries:
        part_p, part_t = [], []
        for p in pred_particles:
            if p.is_primary:
                part_p.append(p)
        for tp in truth_particles:
            if tp.is_primary:
                part_t.append(tp)

    overlap_matrix = np.zeros((len(part_t), len(part_p)), dtype=np.int64)
    for i, tp in enumerate(part_t):
        for j, p in enumerate(part_p):
            overlap_matrix[i, j] = len(np.intersect1d(tp.voxel_indices, 
                                                      p.voxel_indices))

    idx = overlap_matrix.argmax(axis=0)
    intersections = overlap_matrix.max(axis=0)

    idx[intersections < min_overlap_count] = -1
    intersections[intersections < min_overlap_count] = -1

    matches = []

    for j, p in enumerate(part_p):
        select_idx = idx[j]
        if select_idx < 0:
            # If no truth could be matched, assign None
            matched_truth = None
        else:
            matched_truth = part_t[select_idx]
        if relabel:  # TODO: This can potentially lead to duplicate labels
            p.id = j
            matched_truth.id = j
        matches.append((p, matched_truth))

    return matches, idx, intersections


def match_interactions_fn(pred_interactions : List[Interaction], 
                          true_interactions : List[Interaction], 
                          min_overlap_count=1, relabel=False):
    
    f = partial(match_particles_fn, primaries=False, 
                min_overlap_count=min_overlap_count, relabel=relabel)
    
    return f(pred_interactions, true_interactions)


def group_particles_to_interactions_fn(particles : List[Particle]):

    interactions = defaultdict(list)
    for p in particles:
        interactions[p.interaction_id].append(p)

    for int_id, particles in interactions.items():
        interactions[int_id] = Interaction(int_id, particles)

    return list(interactions.values())