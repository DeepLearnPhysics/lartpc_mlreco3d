from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.ui import FullChainEvaluator, FullChainPredictor
from analysis.decorator import evaluate

import numpy as np


@evaluate(['particles', 'interactions'], mode='per_batch')
def statistics(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Collect statistics of predicted particles/interactions.

    To be used for data vs MC comparisons at higher level.
    """
    particles, interactions = [], []

    deghosting          = analysis_cfg['analysis']['deghosting']
    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})

    start_segment_radius = processor_cfg.get('start_segment_radius', 17)
    shower_label          = processor_cfg.get('shower_label', 0)
    Michel_label          = processor_cfg.get('Michel_label', 2)
    track_label          = processor_cfg.get('track_label', 1)
    bin_size             = processor_cfg.get('bin_size', 17) # 5cm

    # Initialize analysis differently depending on data/MC setting
    predictor = FullChainPredictor(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']
    pca = PCA(n_components=2)

    for i, index in enumerate(image_idxs):
        pred_particles = predictor.get_particles(i, only_primaries=False)

        # Loop over predicted particles
        for p in pred_particles:
            direction = None
            if p.semantic_type in [track_label, shower_label] and p.startpoint[0] >= 0.:
                d = np.sqrt(((p.points -p.startpoint)**2).sum(axis=1))
                if (d < start_segment_radius).sum() >= 2:
                    direction = pca.fit(p.points[d < start_segment_radius]).components_[0, :]
            if direction is None:
                if len(p.points) >= 2:
                    direction = pca.fit(p.points).components_[0, :]
                else:
                    direction = [-1, -1, -1]

            length = -1
            if p.semantic_type == track_label:
                length = 0.
                if len(p.points) >= 2:
                    coords_pca = pca.fit_transform(p.points)[:, 0]
                    bins = np.arange(coords_pca.min(), coords_pca.max(), bin_size)
                    # bin_inds takes values in [1, len(bins)]
                    bin_inds = np.digitize(coords_pca, bins)
                    for b_i in np.unique(bin_inds):
                        mask = bin_inds == b_i
                        if np.count_nonzero(mask) < 2: continue
                        # Repeat PCA locally for better measurement of dx
                        pca_axis = pca.fit_transform(p.points[mask])
                        dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
                        length += dx
            update_dict = {
                'index': index,
                'id': p.id,
                'size': p.size,
                'semantic_type': p.semantic_type,
                'pid': p.pid,
                'interaction_id': p.interaction_id,
                'is_primary': p.is_primary,
                'sum_edep': p.depositions.sum(),
                'dir_x': direction[0],
                'dir_y': direction[1],
                'dir_z': direction[2],
                'length': length,
                'start_x': -1,
                'start_y': -1,
                'start_z': -1,
                'end_x': -1,
                'end_y': -1,
                'end_z': -1,
            }
            if p.semantic_type == track_label:
                update_dict.update({
                    'start_x': p.startpoint[0],
                    'start_y': p.startpoint[1],
                    'start_z': p.startpoint[2],
                    'end_x': p.endpoint[0],
                    'end_y': p.endpoint[1],
                    'end_z': p.endpoint[2],
                })
            elif p.semantic_type == shower_label:
                update_dict.update({
                    'start_x': p.startpoint[0],
                    'start_y': p.startpoint[1],
                    'start_z': p.startpoint[2],
                })
            particles.append(OrderedDict(update_dict))

        pred_interactions = predictor.get_interactions(i, drop_nonprimary_particles=False)

        for int in pred_interactions:
            update_dict = {
                'index': index,
                'id': int.id,
                'size': int.size,
                'num_particles': int.num_particles,
            }
            for key in int.particle_counts:
                update_dict[key + '_count'] = int.particle_counts[key]
            for key in int.primary_particle_counts:
                update_dict[key + '_primary_count'] = int.primary_particle_counts[key]
            interactions.append(OrderedDict(update_dict))

    return [particles, interactions]
