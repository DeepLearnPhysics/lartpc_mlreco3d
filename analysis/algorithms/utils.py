from collections import OrderedDict
from turtle import up
from analysis.classes.particle import Interaction, Particle, TruthParticle
from analysis.algorithms.calorimetry import *

from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from analysis.algorithms.point_matching import get_track_endpoints_max_dist
from analysis.algorithms.calorimetry import compute_track_dedx, get_particle_direction

import numpy as np


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


def handle_singleton_ppn_candidate(p, pts, ppn_candidates):
    assert ppn_candidates.shape[0] == 1
    score = ppn_candidates[0][5:]
    label = np.argmax(score)
    dist = cdist(pts, ppn_candidates[:, :3])
    pt_near = pts[dist.argmin(axis=0)]
    pt_far = pts[dist.argmax(axis=0)]
    if label == 0:
        p.startpoint = pt_near.reshape(-1)
        p.endpoint = pt_far.reshape(-1)
    else:
        p.endpoint = pt_near.reshape(-1)
        p.startpoint = pt_far.reshape(-1)



def correct_track_endpoints_ppn(p):
    assert p.semantic_type == 1
    pts = np.vstack([p.startpoint, p.endpoint])

    if p.ppn_candidates.shape[0] == 0:
        p.startpoint = pts[0]
        p.endpoint = pts[1]
    elif p.ppn_candidates.shape[0] == 1:
        # If only one ppn candidate, find track endpoint closer to
        # ppn candidate and give the candidate's label to that track point
        handle_singleton_ppn_candidate(p, pts, p.ppn_candidates)
    else:
        dist1 = cdist(np.atleast_2d(p.ppn_candidates[:, :3]), 
                      np.atleast_2d(pts[0])).reshape(-1)
        dist2 = cdist(np.atleast_2d(p.ppn_candidates[:, :3]), 
                      np.atleast_2d(pts[1])).reshape(-1)
        
        ind1, ind2 = dist1.argmin(), dist2.argmin()
        if ind1 == ind2:
            ppn_candidates = p.ppn_candidates[dist1.argmin()].reshape(1, 7)
            handle_singleton_ppn_candidate(p, pts, ppn_candidates)
        else:
            pt1_score = p.ppn_candidates[ind1][5:]
            pt2_score = p.ppn_candidates[ind2][5:]
            
            labels = np.array([pt1_score.argmax(), pt2_score.argmax()])
            scores = np.array([pt1_score.max(), pt2_score.max()])
            
            if labels[0] == 0 and labels[1] == 1:
                p.startpoint = pts[0]
                p.endpoint = pts[1]
            elif labels[0] == 1 and labels[1] == 0:
                p.startpoint = pts[1]
                p.endpoint = pts[0]
            elif labels[0] == 0 and labels[1] == 0:
                # print("Particle {} has no endpoint".format(p.id))
                # Select point with larger score as startpoint
                ix = np.argmax(scores)
                iy = np.argmin(scores)
                # print(ix, iy, pts, scores)
                p.startpoint = pts[ix]
                p.endpoint = pts[iy]
            elif labels[0] == 1 and labels[1] == 1:
                ix = np.argmax(scores) # point with higher endpoint score
                iy = np.argmin(scores)
                p.startpoint = pts[iy]
                p.endpoint = pts[ix]
            else:
                raise ValueError("Classify endpoints feature dimension must be 2, got something else!")
    if np.linalg.norm(p.startpoint - p.endpoint) < 1e-6:
        p.startpoint = pts[0]
        p.endpoint = pts[1]


def correct_track_endpoints_local_density(p, r=5):
    pca = PCA(n_components=2)
    assert p.semantic_type == 1
    mask_st = np.linalg.norm(p.startpoint - p.points, axis=1) < r
    if np.count_nonzero(mask_st) < 2:
        return
    pca_axis = pca.fit_transform(p.points[mask_st])
    length = pca_axis[:, 0].max() - pca_axis[:, 0].min()
    local_d_start = p.depositions[mask_st].sum() / length
    mask_end = np.linalg.norm(p.endpoint - p.points, axis=1) < r
    if np.count_nonzero(mask_end) < 2:
        return
    pca_axis = pca.fit_transform(p.points[mask_end])
    length = pca_axis[:, 0].max() - pca_axis[:, 0].min()
    local_d_end = p.depositions[mask_end].sum() / length
    # Startpoint must have lowest local density
    if local_d_start > local_d_end:
        p1, p2 = p.startpoint, p.endpoint
        p.startpoint = p2
        p.endpoint = p1


def correct_track_endpoints_linfit(p, bin_size=17):
    if len(p.points) >= 2:
        dedx = compute_track_dedx(p, bin_size=bin_size)
        if len(dedx) > 1:
            x = np.arange(len(dedx))
            params = np.polyfit(x, dedx, 1)
            if params[0] < 0:
                p1, p2 = p.startpoint, p.endpoint
                p.startpoint = p2
                p.endpoint = p1


def correct_track_endpoints_direction(p):
    assert p.semantic_type == 1
    vec = p.endpoint - p.startpoint
    vec = vec / np.linalg.norm(vec)
    direction = get_particle_direction(p, optimize=True)
    direction = direction / np.linalg.norm(direction)
    if np.sum(vec * direction) < 0:
        p1, p2 = p.startpoint, p.endpoint
        p.startpoint = p2
        p.endpoint = p1


def get_track_points(p, correction_mode='ppn', brute_force=False):
    if brute_force:
        pts = np.vstack(get_track_endpoints_max_dist(p))
    else:
        pts = np.vstack([p.startpoint, p.endpoint])
    if correction_mode == 'ppn':
        correct_track_endpoints_ppn(p)
    elif correction_mode == 'local_density':
        correct_track_endpoints_local_density(p)
    elif correction_mode == 'linfit':
        correct_track_endpoints_linfit(p)
    elif correction_mode == 'direction':
        correct_track_endpoints_direction(p)
    else:
        raise ValueError("Track extrema correction mode {} not defined!".format(correction_mode))


def get_interaction_properties(interaction: Interaction, spatial_size, prefix=None):

    update_dict = OrderedDict({
        'interaction_id': -1,
        'interaction_size': -1,
        'count_primary_particles': -1,
        'vertex_x': -1,
        'vertex_y': -1,
        'vertex_z': -1,
        'has_vertex': False,
        'vertex_valid': 'Default Invalid',
        'count_primary_photons': -1,
        'count_primary_electrons': -1,
        'count_primary_muons': -1,
        'count_primary_pions': -1,
        'count_primary_protons': -1,
        'count_pi0': -1
        # 'nu_reco_energy': -1
    })

    if interaction is None:
        out = attach_prefix(update_dict, prefix)
        return out
    else:
        count_primary_muons = {}
        count_primary_particles = {}
        count_primary_protons = {}
        count_primary_electrons = {}
        count_primary_photons = {}
        count_primary_pions = {}

        for p in interaction.particles:
            if p.is_primary:
                count_primary_particles[p.id] = True
                if p.pid == 0:
                    count_primary_photons[p.id] = True
                if p.pid == 1:
                    count_primary_electrons[p.id] = True
                if p.pid == 2:
                    count_primary_muons[p.id] = True
                if p.pid == 3:
                    count_primary_pions[p.id] = True
                if p.pid == 4:
                    count_primary_protons[p.id] = True 

        update_dict['count_pi0'] = len(interaction._pi0_tagged_photons)

        update_dict['interaction_id'] = interaction.id
        update_dict['interaction_size'] = interaction.size
        update_dict['count_primary_muons'] = sum(count_primary_muons.values())
        update_dict['count_primary_photons'] = sum(count_primary_photons.values())
        update_dict['count_primary_pions'] = sum(count_primary_pions.values())
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


def get_particle_properties(particle: Particle, 
                            prefix=None, 
                            save_feats=False,
                            splines=None,
                            compute_energy=False):

    update_dict = OrderedDict({
        'particle_id': -1,
        'particle_interaction_id': -1,
        'particle_type': -1,
        'particle_semantic_type': -1,
        'particle_size': -1,
        'particle_is_primary': False,
        'particle_has_startpoint': False,
        'particle_has_endpoint': False,
        'particle_startpoint_x': -1,
        'particle_startpoint_y': -1,
        'particle_startpoint_z': -1,
        'particle_endpoint_x': -1,
        'particle_endpoint_y': -1,
        'particle_endpoint_z': -1,
        'particle_startpoint_is_touching': True,
        'particle_creation_process': "Default Invalid",
        'particle_num_ppn_candidates': -1,
        # 'particle_is_contained': False
    })

    if compute_energy:
        update_dict.update(OrderedDict({
            'particle_dir_x': -1,
            'particle_dir_y': -1,
            'particle_dir_z': -1,
            'particle_length': -1,
            'particle_reco_energy': -1,
            'particle_sum_edep': -1
        }))

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

        if hasattr(particle, 'ppn_candidates'):
            assert particle.ppn_candidates.shape[1] == 7
            update_dict['particle_num_ppn_candidates'] = len(particle.ppn_candidates)

        if isinstance(particle, TruthParticle):
            if particle.size > 0:
                dists = np.linalg.norm(particle.points - particle.startpoint.reshape(1, -1), axis=1)
                min_dist = np.min(dists)
                if min_dist > 5.0:
                    update_dict['particle_startpoint_is_touching'] = False
            creation_process = particle.particle_asis.creation_process()
            update_dict['particle_creation_process'] = creation_process
            update_dict['particle_px'] = float(particle.particle_asis.px())
            update_dict['particle_py'] = float(particle.particle_asis.py())
            update_dict['particle_pz'] = float(particle.particle_asis.pz())
        if compute_energy and particle.size > 0:
            update_dict['particle_sum_edep'] = particle.sum_edep
            direction = get_particle_direction(particle, optimize=True)
            assert len(direction) == 3
            update_dict['particle_dir_x'] = direction[0]
            update_dict['particle_dir_y'] = direction[1]
            update_dict['particle_dir_z'] = direction[2]
            if particle.semantic_type == 1:
                length =  compute_track_length(particle.points)
                update_dict['particle_length'] = length
                particle.length = length
                if splines is not None and particle.pid == 4:
                    reco_energy = compute_range_based_energy(particle, splines['proton'])
                    update_dict['particle_reco_energy'] = reco_energy
                if splines is not None and particle.pid == 2:
                    reco_energy = compute_range_based_energy(particle, splines['muon'])
                    update_dict['particle_reco_energy'] = reco_energy

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

def closest_distance_two_lines(a0, u0, a1, u1):
    '''
    a0, u0: point (a0) and unit vector (u0) defining line 1
    a1, u1: point (a1) and unit vector (u1) defining line 2
    '''
    cross = np.cross(u0, u1)
    # if the cross product is zero, the lines are parallel
    if np.linalg.norm(cross) == 0:
        # use any point on line A and project it onto line B
        t = np.dot(a1 - a0, u1)
        a = a1 + t * u1 # projected point
        return np.linalg.norm(a0 - a)
    else:
        # use the formula from https://en.wikipedia.org/wiki/Skew_lines#Distance
        t = np.dot(np.cross(a1 - a0, u1), cross) / np.linalg.norm(cross)**2
        # closest point on line A to line B
        p = a0 + t * u0
        # closest point on line B to line A
        q = p - cross * np.dot(p - a1, cross) / np.linalg.norm(cross)**2
        return np.linalg.norm(p - q) # distance between p and q