from collections import OrderedDict
from turtle import up
from analysis.classes.particle import Interaction, Particle, TruthParticle
from analysis.algorithms.calorimetry import compute_track_length, compute_particle_direction

from pprint import pprint
import numpy as np


def attach_prefix(update_dict, prefix):
    if prefix is None:
        return update_dict
    out = OrderedDict({})

    for key, val in update_dict.items():
        new_key = "{}_".format(prefix) + str(key)
        out[new_key] = val

    return out


def count_primary_particles(interaction: Interaction, prefix=None):

    update_dict = OrderedDict({
        'interaction_id': -1,
        'count_primary_leptons': -1,
        'count_primary_particles': -1,
        'vertex_x': -1,
        'vertex_y': -1,
        'vertex_z': -1,
        'has_vertex': False,
        'count_primary_protons': -1
    })

    if interaction is None:
        out = attach_prefix(update_dict, prefix)
        return out
    else:
        count_primary_leptons = {}
        count_primary_particles = {}
        count_primary_protons = {}

        for p in interaction.particles:
            if p.is_primary:
                count_primary_particles[p.id] = True
                if (p.pid == 1 or p.pid == 2):
                    count_primary_leptons[p.id] = True
                elif p.pid == 4:
                    count_primary_protons[p.id] = True

        update_dict['interaction_id'] = interaction.id
        update_dict['count_primary_leptons'] = sum(count_primary_leptons.values())
        update_dict['count_primary_particles'] = sum(count_primary_particles.values())
        update_dict['count_primary_protons'] = sum(count_primary_protons.values())
        if (np.array(interaction.vertex) > 0).all():
            update_dict['has_vertex'] = True
            update_dict['vertex_x'] = interaction.vertex[0]
            update_dict['vertex_y'] = interaction.vertex[1]
            update_dict['vertex_z'] = interaction.vertex[1]

        out = attach_prefix(update_dict, prefix)

    return out


def get_particle_properties(particle: Particle, vertex, prefix=None, save_feats=False):

    update_dict = OrderedDict({
        'particle_id': -1,
        'particle_interaction_id': -1,
        'particle_type': -1,
        'particle_size': -1,
        'particle_E': -1,
        'particle_is_primary': False,
        'particle_has_startpoint': False,
        'particle_has_endpoints': False,
        'particle_length': -1,
        'particle_dir_x': -1,
        'particle_dir_y': -1,
        'particle_dir_z': -1
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
        update_dict['particle_size'] = particle.size
        update_dict['particle_E'] = particle.sum_edep
        update_dict['particle_is_primary'] = particle.is_primary
        if particle.semantic_type == 1:
            update_dict['particle_length'] = compute_track_length(particle.points)
            direction = compute_particle_direction(particle, vertex=vertex)
            assert len(direction) == 3
            update_dict['particle_dir_x'] = direction[0]
            update_dict['particle_dir_y'] = direction[1]
            update_dict['particle_dir_z'] = direction[2]
        # if not isinstance(particle, TruthParticle):
        #     node_dict = OrderedDict({'node_feat_{}'.format(i) : particle.node_features[i] \
        #         for i in range(particle.node_features.shape[0])})

        #     update_dict.update(node_dict)

    out = attach_prefix(update_dict, prefix)

    return out
