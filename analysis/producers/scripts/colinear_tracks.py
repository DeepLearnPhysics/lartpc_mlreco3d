from analysis.producers.decorator import write_to
from analysis.classes.TruthParticleFragment import TruthParticleFragment
from analysis.classes.ParticleFragment import ParticleFragment
from analysis.producers.logger import ParticleLogger
from collections import OrderedDict
import numpy as np

from scipy.spatial.distance import cosine
from sklearn.metrics import adjusted_rand_score
from analysis.utils import twopoint_iou
import sys

from collections import Counter

@write_to(['particle_pairs_t2r', 'particle_pairs_r2t'])
def select_particle_pairs(data_blob, res, **kwargs):
    """
    Select particle pairs for logging (only for mpv/nu interactions)
    """

    particles_t2r, particles_r2t = [], []

    particle_fieldnames   = kwargs['logger'].get('particles', {})
    pixel_threshold       = kwargs.get('pixel_threshold', -1)
    semantic_label        = kwargs.get('semantic_label', 1)
    only_primaries        = kwargs.get('only_primaries', True)

    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            # 'file_index': data_blob['file_index'][idx]
        }

        # 1. Match Particles
        # pmatches, pcounts = res['matched_particles_t2r'][idx], res['particle_match_overlap_t2r'][idx]
        pmatches = []
        for p in res['matched_particles_t2r'][idx]:
            if p[0].size < pixel_threshold:
                continue
            if p[0].semantic_type != semantic_label:
                continue
            if only_primaries and not p[0].is_primary:
                continue
            if p[0].nu_id == 1:
                pmatches.append(p)

        if len(pmatches) > 0:
            # pindices = [i for i, p in enumerate(res['matched_particles_t2r'][idx]) if p[0].nu_id == 1]
            # pcounts = [pm for i, pm in enumerate(pmatches) if i == pindices[i]]

            # 3. Process particle level information
            particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
            particle_logger.prepare()
            
            for i, p1 in enumerate(pmatches):
                for j, p2 in enumerate(pmatches):
                    if i < j:
                        true_1, true_2 = p1[0], p2[0]
                        reco_1, reco_2 = p1[1], p2[1]
                        
                        reco_pair = [reco_1, reco_2]
                        true_pair = [true_1, true_2]
                        
                        # Colinearity
                        if np.isinf(true_1.truth_start_dir).any() or np.isinf(true_2.truth_start_dir).any():
                            dist = np.inf
                        else:
                            dist = cosine(true_1.truth_start_dir, true_2.truth_start_dir)
                        # Compute overlap
                        mean_iou, max_iou = twopoint_iou(reco_pair,
                                                         true_pair)
                        
                        index_dict['cosine_dist']       = dist
                        index_dict['twopoint_mean_iou'] = mean_iou
                        index_dict['twopoint_max_iou']  = max_iou
                        
                        true_p1_dict = particle_logger.produce(true_1, mode='true', prefix='p1')
                        true_p2_dict = particle_logger.produce(true_2, mode='true', prefix='p2')
                        
                        pred_p1_dict = particle_logger.produce(reco_1, mode='reco', prefix='p1')
                        pred_p2_dict = particle_logger.produce(reco_2, mode='reco', prefix='p2')
                        
                        part_dict = OrderedDict()
                        part_dict.update(index_dict)
                        part_dict.update(true_p1_dict)
                        part_dict.update(true_p2_dict)
                        part_dict.update(pred_p1_dict)
                        part_dict.update(pred_p2_dict)
                        particles_t2r.append(part_dict)
                
        # 1. Match Interactions and log interaction-level information
        pmatches = []
        for p in res['matched_particles_r2t'][idx]:
            if p[1] is None:
                continue
            if p[1].size < pixel_threshold:
                continue
            if p[1].semantic_type != semantic_label:
                continue
            if only_primaries and not p[1].is_primary:
                continue
            if p[1].nu_id == 1:
                pmatches.append(p)
        # pmatches = [p for p in res['matched_particles_r2t'][idx] if (p[1] is not None and p[1].nu_id == 1)]

        if len(pmatches) > 0:
            # pindices = [i for i, p in enumerate(res['matched_particles_r2t'][idx]) if (p[1] is not None and p[1].nu_id == 1)]
            # pcounts = [pm for i, pm in enumerate(pmatches) if i == pindices[i]]

            # 3. Process particle level information
            particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
            particle_logger.prepare()
            
            for i, p1 in enumerate(pmatches):
                for j, p2 in enumerate(pmatches):
                    if i < j:
                        true_1, true_2 = p1[1], p2[1]
                        reco_1, reco_2 = p1[0], p2[0]
                        
                        reco_pair = [reco_1, reco_2]
                        true_pair = [true_1, true_2]
                        
                        # Colinearity
                        if np.isinf(true_1.truth_start_dir).any() or np.isinf(true_2.truth_start_dir).any():
                            dist = np.inf
                        else:
                            dist = cosine(true_1.truth_start_dir, true_2.truth_start_dir)
                        # Compute overlap
                        score = twopoint_iou(reco_pair,
                                             true_pair)
                        
                        index_dict['cosine_dist'] = dist
                        index_dict['twopoint_iou'] = score
                        
                        true_p1_dict = particle_logger.produce(true_1, mode='true', prefix='p1')
                        true_p2_dict = particle_logger.produce(true_2, mode='true', prefix='p2')
                        
                        pred_p1_dict = particle_logger.produce(reco_1, mode='reco', prefix='p1')
                        pred_p2_dict = particle_logger.produce(reco_2, mode='reco', prefix='p2')
                        
                        part_dict = OrderedDict()
                        part_dict.update(index_dict)
                        part_dict.update(true_p1_dict)
                        part_dict.update(true_p2_dict)
                        part_dict.update(pred_p1_dict)
                        part_dict.update(pred_p2_dict)
                        particles_r2t.append(part_dict)

    return [particles_t2r, particles_r2t]