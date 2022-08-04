from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.ui import FullChainEvaluator, FullChainPredictor
from analysis.decorator import evaluate

from pprint import pprint
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def is_attached_at_edge(points1, points2, attached_threshold=5,
                        one_pixel=5, ablation_radius=15, ablation_min_samples=5,
                        return_dbscan_cluster_count=False):
    distances = cdist(points1, points2)
    is_attached = np.min(distances) < attached_threshold
    # check for the edge now
    Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
    min_coords = points2[MIP_min, :]
    ablated_cluster = points2[np.linalg.norm(points2-min_coords, axis=1) > ablation_radius]
    new_cluster_count, old_cluster_count = 0, 1
    if ablated_cluster.shape[0] > 0:
        dbscan = DBSCAN(eps=one_pixel, min_samples=ablation_min_samples)
        old_cluster = dbscan.fit(points2).labels_
        new_cluster = dbscan.fit(ablated_cluster).labels_
        # If only one cluster is left, we were at the edge
        # Account for old cluster count in case track is fragmented
        # and put together by Track GNN
        old_cluster_count = len(np.unique(old_cluster[old_cluster>-1]))
        new_cluster_count =  len(np.unique(new_cluster[new_cluster>-1]))
        is_edge = (old_cluster_count - new_cluster_count) <= 1 and old_cluster_count >= new_cluster_count
    else: # if nothing is left after ablating, this is a really small muon... calling it the edge
        is_edge = True

    if return_dbscan_cluster_count:
        return is_attached and is_edge, new_cluster_count, old_cluster_count

    return is_attached and is_edge


@evaluate(['michels_pred', 'michels_true'], mode='per_batch')
def michel_electrons(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Selection of Michel electrons
    =============================


    Configuration
    =============
    Under `processor_cfg`, you can specify the following parameters:


    Output
    ======
    """
    #
    # ====== Configuration ======
    #
    michels, true_michels = [], []
    deghosting = analysis_cfg['analysis']['deghosting']

    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})
    spatial_size        = processor_cfg['spatial_size']
    # Whether we are running on MC or data
    data           = processor_cfg.get('data', False)

    # Avoid hardcoding labels
    michel_label       = processor_cfg.get('michel_label', 2)
    track_label        = processor_cfg.get('track_label', 1)
    muon_label         = processor_cfg.get('muon_label', 2)
    pion_label         = processor_cfg.get('pion_label', 3)
    shower_label       = processor_cfg.get('shower_label', 0)

    # Thresholds
    attached_threshold = processor_cfg.get('attached_threshold', 5)
    one_pixel          = processor_cfg.get('ablation_eps', 5)
    ablation_radius    = processor_cfg.get('ablation_radius', 15)
    ablation_min_samples = processor_cfg.get('ablation_min_samples', 5)
    shower_threshold   = processor_cfg.get('shower_threshold', 10)
    fiducial_threshold = processor_cfg.get('fiducial_threshold', 30)
    muon_min_voxel_count = processor_cfg.get('muon_min_voxel_count', 30)
    matching_mode      = processor_cfg.get('matching_mode', 'true_to_pred')
    #
    # ====== Initialization ======
    #
    # Initialize analysis differently depending on data/MC setting
    if not data:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg, deghosting=deghosting)
    else:
        predictor = FullChainPredictor(data_blob, res, cfg, processor_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']


    for i, index in enumerate(image_idxs):
        pred_particles = predictor.get_particles(i, only_primaries=False)

        # Match with true particles if available
        if not data:
            true_particles = predictor.get_true_particles(i, only_primaries=False)
            # Match true particles to predicted particles
            true_ids = np.array([p.id for p in true_particles])
            matched_particles = predictor.match_particles(i, mode=matching_mode, min_overlap=5, overlap_mode='counts')

            for tp in true_particles:
                if tp.semantic_type != michel_label: continue
                if not predictor.is_contained(tp.points, threshold=fiducial_threshold): continue
                p = data_blob['particles_asis'][i][tp.id]

                michel_is_attached_at_edge = False
                distance_to_muon = np.infty
                muon_ablation_cluster_count = -1
                muon_ablation_delta = None
                for tp2 in true_particles:
                    if tp2.semantic_type != track_label: continue
                    if tp2.pid != muon_label and tp2.pid != pion_label: continue
                    if tp2.size < muon_min_voxel_count: continue
                    d = cdist(tp.points, tp2.points).min()
                    attached_at_edge, cluster_count, old_cluster_count = is_attached_at_edge(tp.points, tp2.points,
                                                                    attached_threshold=attached_threshold,
                                                                    one_pixel=one_pixel,
                                                                    ablation_radius=ablation_radius,
                                                                    ablation_min_samples=ablation_min_samples,
                                                                    return_dbscan_cluster_count=True)
                    if d < distance_to_muon:
                        distance_to_muon = d
                        muon_ablation_cluster_count = cluster_count
                        muon_ablation_delta = old_cluster_count - cluster_count
                    if not attached_at_edge: continue
                    michel_is_attached_at_edge = True
                    break

                # Determine whether it was matched
                is_matched = False
                for mp in matched_particles: # matching is done true to pred
                    true_idx = 0 if matching_mode == 'true_to_pred' else 1
                    if mp[true_idx] is None or mp[true_idx].id != tp.id: continue
                    is_matched = True
                    break

                # DBSCAN evaluation
                dbscan = DBSCAN(eps=one_pixel, min_samples=ablation_min_samples)
                michel_clusters = dbscan.fit(tp.points).labels_
                print(np.unique(michel_clusters, return_counts=True))
                # Record true Michel
                true_michels.append(OrderedDict({
                    'index': index,
                    'true_num_pix': tp.size,
                    'true_sum_pix': tp.depositions.sum(),
                    'is_attached_at_edge': michel_is_attached_at_edge,
                    'muon_ablation_cluster_count': muon_ablation_cluster_count,
                    'muon_ablation_delta': muon_ablation_delta,
                    'is_matched': is_matched,
                    'energy_init': p.energy_init(),
                    'energy_deposit': p.energy_deposit(),
                    'num_voxels': p.num_voxels(),
                    'distance_to_muon': distance_to_muon,
                    'michel_cluster_count': len(np.unique(michel_clusters[michel_clusters>-1]))
                }))
        # Loop over predicted particles
        for p in pred_particles:
            if p.semantic_type != michel_label: continue
            if not predictor.is_contained(p.points, threshold=fiducial_threshold): continue

            # Check whether it is attached to the edge of a track
            michel_is_attached_at_edge = False
            for p2 in pred_particles:
                if p2.semantic_type != track_label: continue
                if p2.size < muon_min_voxel_count: continue
                if not is_attached_at_edge(p.points, p2.points,
                                        attached_threshold=attached_threshold,
                                        one_pixel=one_pixel,
                                        ablation_radius=ablation_radius,
                                        ablation_min_samples=ablation_min_samples): continue
                michel_is_attached_at_edge = True
                break

            # Require that the Michel is attached at the edge of a track
            if not michel_is_attached_at_edge: continue

            # Look for shower-like particles nearby
            daughter_showers = []
            distance_to_closest_shower = np.infty
            if shower_threshold > -1:
                for p2 in pred_particles:
                    if p2.semantic_type != shower_label: continue
                    distance_to_closest_shower = min(distance_to_closest_shower, cdist(p.points, p2.points).min())
                    if distance_to_closest_shower >= shower_threshold: continue
                    daughter_showers.append(p2)
            daughter_num_pix = np.sum([daugher.size for daugher in daughter_showers])
            daughter_sum_pix = np.sum([daughter.depositions.sum() for daughter in daughter_showers])

            # Record candidate Michel
            update_dict = {
                'index': index,
                'particle_id': p.id,
                'daughter_num_pix': daughter_num_pix,
                'daughter_sum_pix': daughter_sum_pix,
                'distance_to_closest_shower': distance_to_closest_shower,
                'pred_num_pix': p.size,
                'pred_sum_pix': p.depositions.sum(),
                'true_num_pix': -1,
                'true_sum_pix': -1,
                'pred_num_pix_true': -1,
                'pred_sum_pix_true': -1,
                'michel_true_energy_init': -1,
                'michel_true_energy_deposit': -1,
                'cluster_purity': -1,
                'cluster_efficiency': -1,
                'matched': False,
                'true_pdg': -1,
                'true_id': -1,
                'true_semantic_type': -1,
                'true_noghost_num_pix': -1,
                'true_noghost_sum_pix': -1,
                'true_noghost_primary_num_pix': -1,
                'true_noghost_primary_sum_pix': -1
            }

            if not data:
                for mp in matched_particles: # matching is done true2pred
                    true_idx = 0 if matching_mode == 'true_to_pred' else 1
                    pred_idx = 1 - true_idx
                    if mp[pred_idx] is None or mp[pred_idx].id != p.id: continue
                    if mp[true_idx] is None: continue
                    m = mp[true_idx]
                    pe = m.purity_efficiency(p)
                    overlap_indices, mindices, _ = np.intersect1d(m.voxel_indices, p.voxel_indices, return_indices=True)
                    truep = data_blob['particles_asis'][i][m.id]

                    truecluster = data_blob['cluster_label_noghost'][i][:, 6] == m.id
                    trueprimary = (data_blob['cluster_label_noghost'][i][:, 6] == m.id) & (data_blob['cluster_label_noghost'][i][:, -1] == michel_label)
                    update_dict.update({
                        'matched': True,
                        'true_id': m.id,
                        'true_pdg': m.pid,
                        'true_semantic_type': m.semantic_type,
                        'cluster_purity': pe['purity'],
                        'cluster_efficiency': pe['efficiency'],
                        'true_num_pix': m.size,
                        'true_sum_pix': m.depositions.sum(),
                        'pred_num_pix_true': len(overlap_indices),
                        'pred_sum_pix_true': m.depositions[mindices].sum(),
                        'michel_true_energy_init': truep.energy_init(),
                        'michel_true_energy_deposit': truep.energy_deposit(),
                        'michel_true_num_voxels': truep.num_voxels(),
                        # cluster_label includes radiative photon in Michel true particle
                        # so using segmentation labels to find primary ionization as well
                        # voxel sum will be in MeV here.
                        'true_noghost_num_pix': np.count_nonzero(truecluster),
                        'true_noghost_sum_pix': data_blob['cluster_label_noghost'][i][truecluster, 4].sum(),
                        'true_noghost_primary_num_pix': np.count_nonzero(trueprimary),
                        'true_noghost_primary_sum_pix': data_blob['cluster_label_noghost'][i][trueprimary, 4].sum()
                    })

            michels.append(OrderedDict(update_dict))

    return [michels, true_michels]
