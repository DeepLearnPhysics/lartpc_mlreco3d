from collections import defaultdict
from itertools import combinations
from analysis.post_processing.reconstruction.utils import closest_distance_two_lines
from mlreco.utils.gnn.cluster import cluster_direction

# TODO: Need to refactor according to post processing conventions

def _tag_neutral_pions_true(particles):
    out = []
    tagged = defaultdict(list)
    for part in particles:
        num_voxels_noghost = part.coords_noghost.shape[0]
        ancestor = part.ancestor_track_id
        if part.pdg_code == 22 \
            and part.creation_process == "Decay" \
            and part.parent_creation_process == "primary" \
            and part.ancestor_pdg_code = 111 \
            and num_voxels_noghost > 0:
            tagged[ancestor].append(part.id)
    for photon_list in tagged.values():
        out.append(tuple(photon_list))
    return out

def _tag_neutral_pions_reco(particles, threshold=5):
    out = []
    photons = [p for p in particles if p.pid == 0]
    for entry in combinations(photons, 2):
        p1, p2 = entry
        v1, v2 = cluster_direction(p1), cluster_direction(p2)
        d = closest_distance_two_lines(p1.startpoint, v1, p2.startpoint, v2)
        if d < threshold:
            out.append((p1.id, p2.id))
    return out

def tag_neutral_pions(particles, mode):
    if mode == 'truth':
        return _tag_neutral_pions_true(particles)
    elif mode == 'pred':
        return _tag_neutral_pions_reco(particles)
    else:
        raise ValueError
