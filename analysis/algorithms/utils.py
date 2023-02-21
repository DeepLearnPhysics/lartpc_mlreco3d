from collections import OrderedDict
from turtle import up
from analysis.classes.particle import Interaction, Particle, TruthParticle
from analysis.algorithms.calorimetry import *

from scipy.spatial.distance import cdist
import numpy as np
import ROOT


def attach_prefix(update_dict, prefix):
    if prefix is None:
        return update_dict
    out = OrderedDict({})

    for key, val in update_dict.items():
        new_key = "{}_".format(prefix) + str(key)
        out[new_key] = val

    return out


def correct_track_points(particle):
    '''
    Correct track startpoint and endpoint using PPN's
    <classify_endpoints> prediction.

    Warning: only meant for tracks, operation is in-place.
    '''
    assert particle.semantic_type == 1
    num_candidates = particle.ppn_candidates.shape[0]

    x = np.vstack([particle.startpoint, particle.endpoint])

    if num_candidates == 0:
        pass
    elif num_candidates == 1:
        # Get closest candidate and place candidate's label
        # print(x.shape, particle.ppn_candidates[0, :3])
        dist = cdist(x, particle.ppn_candidates[:, :3]).squeeze()
        label = np.argmax(particle.ppn_candidates[0, 5:])
        x1, x2 = np.argmin(dist), np.argmax(dist)
        if label == 0:
            # Closest point x1 is adj to a startpoint
            particle.startpoint = x[x1]
            particle.endpoint = x[x2]
        elif label == 1:
            # Closest point x2 is adj to an endpoint
            particle.endpoint = x[x1]
            particle.startpoint = x[x2]
        else:
            raise ValueError("Track endpoint label should be either 0 or 1, \
                got {}, which should not happen!".format(label))
    else:
        dist = cdist(x, particle.ppn_candidates[:, :3])
        # Classify endpoint scores associated with x
        scores = particle.ppn_candidates[dist.argmin(axis=1)][:, 5:]
        particle.startpoint = x[scores[:, 0].argmax()]
        particle.endpoint = x[scores[:, 1].argmax()]


def load_range_reco(particle_type='muon', kinetic_energy=True):
    """
    Return a function maps the residual range of a track to the kinetic
    energy of the track. The mapping is based on the Bethe-Bloch formula
    and stored per particle type in TGraph objects. The TGraph::Eval
    function is used to perform the interpolation.

    Parameters
    ----------
    particle_type: A string with the particle name.
    kinetic_energy: If true (false), return the kinetic energy (momentum)
    
    Returns
    -------
    The kinetic energy or momentum according to Bethe-Bloch.
    """
    output_var = ('_RRtoT' if kinetic_energy else '_RRtodEdx')
    if particle_type in ['muon', 'pion', 'kaon', 'proton']:
        input_file = ROOT.TFile.Open('RRInput.root', 'read')
        graph = input_file.Get(f'{particle_type}{output_var}')
        return np.vectorize(graph.Eval)
    else:
        print(f'Range-based reconstruction for particle "{particle_type}" not available.')


def make_range_based_momentum_fns():
    f_muon = load_range_reco('muon')
    f_pion = load_range_reco('pion')
    f_proton = load_range_reco('proton')
    return [f_muon, f_pion, f_proton]


def get_interaction_properties(interaction: Interaction, spatial_size, prefix=None):

    update_dict = OrderedDict({
        'interaction_id': -1,
        'interaction_size': -1,
        'count_primary_leptons': -1,
        'count_primary_electrons': -1,
        'count_primary_particles': -1,
        'vertex_x': -1,
        'vertex_y': -1,
        'vertex_z': -1,
        'has_vertex': False,
        'vertex_valid': 'Default Invalid',
        'count_primary_protons': -1
    })

    if interaction is None:
        out = attach_prefix(update_dict, prefix)
        return out
    else:
        count_primary_leptons = {}
        count_primary_particles = {}
        count_primary_protons = {}
        count_primary_electrons = {}

        for p in interaction.particles:
            if p.is_primary:
                count_primary_particles[p.id] = True
                if p.pid == 1:
                    count_primary_electrons[p.id] = True
                if (p.pid == 1 or p.pid == 2):
                    count_primary_leptons[p.id] = True
                elif p.pid == 4:
                    count_primary_protons[p.id] = True

        update_dict['interaction_id'] = interaction.id
        update_dict['interaction_size'] = interaction.size
        update_dict['count_primary_leptons'] = sum(count_primary_leptons.values())
        update_dict['count_primary_particles'] = sum(count_primary_particles.values())
        update_dict['count_primary_protons'] = sum(count_primary_protons.values())
        update_dict['count_primary_electrons'] = sum(count_primary_electrons.values())

        within_volume = np.all(interaction.vertex <= spatial_size) and np.all(interaction.vertex >= 0)

        if within_volume:
            update_dict['has_vertex'] = True
            update_dict['vertex_x'] = interaction.vertex[0]
            update_dict['vertex_y'] = interaction.vertex[1]
            update_dict['vertex_z'] = interaction.vertex[2]
            update_dict['vertex_valid'] = 'Valid'
        else:
            if ((np.abs(np.array(interaction.vertex)) > 1e6).any()):
                update_dict['vertex_valid'] = 'Invalid Magnitude'
            else:
                update_dict['vertex_valid'] = 'Outside Volume'
                update_dict['has_vertex'] = True
                update_dict['vertex_x'] = interaction.vertex[0]
                update_dict['vertex_y'] = interaction.vertex[1]
                update_dict['vertex_z'] = interaction.vertex[2]
        out = attach_prefix(update_dict, prefix)

    return out


def get_particle_properties(particle: Particle, prefix=None, save_feats=False):

    update_dict = OrderedDict({
        'particle_id': -1,
        'particle_interaction_id': -1,
        'particle_type': -1,
        'particle_semantic_type': -1,
        'particle_size': -1,
        'particle_E': -1,
        'particle_is_primary': False,
        'particle_has_startpoint': False,
        'particle_has_endpoint': False,
        'particle_length': -1,
        'particle_dir_x': -1,
        'particle_dir_y': -1,
        'particle_dir_z': -1,
        'particle_startpoint_x': -1,
        'particle_startpoint_y': -1,
        'particle_startpoint_z': -1,
        'particle_endpoint_x': -1,
        'particle_endpoint_y': -1,
        'particle_endpoint_z': -1,
        'particle_startpoint_is_touching': True,
        # 'particle_is_contained': False
    })

    if save_feats:
        node_dict = OrderedDict({'node_feat_{}'.format(i) : -1 for i in range(28)})
        update_dict.update(node_dict)

    if particle is None:
        out = attach_prefix(update_dict, prefix)
        return out
    else:
        update_dict['particle_id'] = particle.id
        update_dict['particle_interaction_id'] = particle.interaction_id
        update_dict['particle_type'] = particle.pid
        update_dict['particle_semantic_type'] = particle.semantic_type
        update_dict['particle_size'] = particle.size
        update_dict['particle_E'] = particle.sum_edep
        update_dict['particle_is_primary'] = particle.is_primary
        # update_dict['particle_is_contained'] = particle.is_contained
        if particle.startpoint is not None:
            update_dict['particle_has_startpoint'] = True
            update_dict['particle_startpoint_x'] = particle.startpoint[0]
            update_dict['particle_startpoint_y'] = particle.startpoint[1]
            update_dict['particle_startpoint_z'] = particle.startpoint[2]
        if particle.endpoint is not None:
            update_dict['particle_has_endpoint'] = True
            update_dict['particle_endpoint_x'] = particle.endpoint[0]
            update_dict['particle_endpoint_y'] = particle.endpoint[1]
            update_dict['particle_endpoint_z'] = particle.endpoint[2]

        if isinstance(particle, TruthParticle):
            dists = np.linalg.norm(particle.points - particle.startpoint.reshape(1, -1), axis=1)
            min_dist = np.min(dists)
            if min_dist > 5.0:
                update_dict['particle_startpoint_is_touching'] = False
        # if particle.semantic_type == 1:
        #     update_dict['particle_length'] = compute_track_length(particle.points)
        #     direction = compute_particle_direction(particle, vertex=vertex)
        #     assert len(direction) == 3
        #     update_dict['particle_dir_x'] = direction[0]
        #     update_dict['particle_dir_y'] = direction[1]
        #     update_dict['particle_dir_z'] = direction[2]
            # if particle.pid == 2:
            #     mcs_E = compute_mcs_muon_energy(particle)
            #     update_dict['particle_mcs_E'] = mcs_E
        # if not isinstance(particle, TruthParticle):
        #     node_dict = OrderedDict({'node_feat_{}'.format(i) : particle.node_features[i] \
        #         for i in range(particle.node_features.shape[0])})

        #     update_dict.update(node_dict)

    out = attach_prefix(update_dict, prefix)

    return out


def get_mparticles_from_minteractions(int_matches):
    '''
    Given list of Tuple[(Truth)Interaction, (Truth)Interaction], 
    return list of particle matches Tuple[TruthParticle, Particle]. 

    If no match, (Truth)Particle is replaced with None.
    '''

    matched_particles, match_counts = [], []

    for m in int_matches:
        ia1, ia2 = m[0], m[1]
        num_parts_1, num_parts_2 = -1, -1
        if m[0] is not None:
            num_parts_1 = len(m[0].particles)
        if m[1] is not None:
            num_parts_2 = len(m[1].particles)
        if num_parts_1 <= num_parts_2:
            ia1, ia2 = m[0], m[1]
        else:
            ia1, ia2 = m[1], m[0]
            
        for p in ia2.particles:
            if len(p.match) == 0:
                if type(p) is Particle:
                    matched_particles.append((None, p))
                    match_counts.append(-1)
                else:
                    matched_particles.append((p, None))
                    match_counts.append(-1)
            for match_id in p.match:
                if type(p) is Particle:
                    matched_particles.append((ia1[match_id], p))
                else:
                    matched_particles.append((p, ia1[match_id]))
                match_counts.append(p._match_counts[match_id])
    return matched_particles, np.array(match_counts)