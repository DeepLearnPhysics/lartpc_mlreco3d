from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from lartpc_mlreco3d.analysis.algorithms.arxiv.calorimetry import compute_track_length, get_particle_direction
from analysis.classes.predictor import FullChainPredictor
from analysis.classes.evaluator import FullChainEvaluator
from lartpc_mlreco3d.analysis.algorithms.arxiv.decorator import evaluate

import numpy as np


@evaluate(['particles', 'interactions', 'events', 'opflash', 'ohmflash'])
def statistics(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Collect statistics of predicted particles/interactions.

    To be used for data vs MC comparisons at higher level.
    """
    particles, interactions, events, opflashes, ohmflashes = [], [], [], [], []

    deghosting          = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['drop_nonprimary_particles']
    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})

    start_segment_radius = processor_cfg.get('start_segment_radius', 17)
    shower_label          = processor_cfg.get('shower_label', 0)
    Michel_label          = processor_cfg.get('Michel_label', 2)
    track_label          = processor_cfg.get('track_label', 1)
    bin_size             = processor_cfg.get('bin_size', 17) # 5cm

    # Initialize analysis differently depending on data/MC setting
    predictor = FullChainPredictor(data_blob, res, cfg, processor_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']
    pca = PCA(n_components=2)

    for i, index in enumerate(image_idxs):
        index_dict = {
            'Index': index,
            'run': data_blob['run_info'][i][0],
            'subrun': data_blob['run_info'][i][1],
            'event': data_blob['run_info'][i][2]
        }
        edep_before_cryoE = predictor.data_blob['input_data_noghost'][i][:, 4]
        edep_after_cryoE = predictor.data_blob['input_data'][i][:, 4]
        edep_before_cryoW = predictor.data_blob['input_data_noghost'][i+1][:, 4]
        edep_after_cryoW = predictor.data_blob['input_data'][i+1][:, 4]
        update_dict = {
                "edep_count_before_cryoE": edep_before_cryoE.shape[0],
                "edep_count_before_cryoW": edep_before_cryoW.shape[0],
                "edep_count_after_cryoE": edep_after_cryoE.shape[0],
                "edep_count_after_cryoW": edep_after_cryoW.shape[0],
                "opflash_cryoE_count": len(predictor.data_blob['opflash_cryoE'][i]),
                "opflash_cryoW_count": len(predictor.data_blob['opflash_cryoW'][i]),
                "ohmflash_michel_cryoE_count": len(predictor.data_blob['ohmflash_cryoE'][i]),
                "ohmflash_michel_cryoW_count": len(predictor.data_blob['ohmflash_cryoW'][i]),
                "ohmflash_muon_cryoE_count": len(predictor.data_blob['ohmflash_muon_cryoE'][i]),
                "ohmflash_muon_cryoW_count": len(predictor.data_blob['ohmflash_muon_cryoW'][i])
        }
        update_dict.update(index_dict)
        events.append(OrderedDict(update_dict))

        # Loop over opflash
        opflash_cryoE = predictor.data_blob['opflash_cryoE'][i]
        opflash_cryoW = predictor.data_blob['opflash_cryoW'][i]
        opflash_cryo = np.array([0] * len(opflash_cryoE) + [1] * len(opflash_cryoW))
        for opidx, opflash in enumerate(opflash_cryoE + opflash_cryoW):
            update_dict = {
                    "pe_sum": opflash.TotalPE(),
                    "pe_count": np.count_nonzero(opflash.PEPerOpDet()),
                    "time": opflash.time(),
                    "absTime": opflash.absTime(),
                    "time_width": opflash.timeWidth(),
                    "volume": opflash_cryo[opidx],
            }
            update_dict.update(index_dict)
            opflashes.append(OrderedDict(update_dict))

        # Loop over OHMFlash
        ohmflash_michel_cryoE = predictor.data_blob['ohmflash_cryoE'][i]
        ohmflash_michel_cryoW = predictor.data_blob['ohmflash_cryoW'][i]
        ohmflash_muon_cryoE = predictor.data_blob['ohmflash_muon_cryoE'][i]
        ohmflash_muon_cryoW = predictor.data_blob['ohmflash_muon_cryoW'][i]
        is_michel = np.array([True] * (len(ohmflash_michel_cryoE) + len(ohmflash_michel_cryoW)) + [False] * (len(ohmflash_muon_cryoE) + len(ohmflash_muon_cryoW)), dtype=np.bool)
        volume = np.array([0] * len(ohmflash_michel_cryoE) + [1] * len(ohmflash_michel_cryoW) + [0] * len(ohmflash_muon_cryoE) + [1] * len(ohmflash_muon_cryoW), dtype=np.int)
        for opidx, ohmflash in enumerate(ohmflash_michel_cryoE + ohmflash_michel_cryoW + ohmflash_muon_cryoE + ohmflash_muon_cryoW):
            update_dict = {
                    "pe_count": np.count_nonzero(ohmflash.PEPerOpDet()),
                    "time": ohmflash.time(),
                    "absTime": ohmflash.absTime(),
                    "volume": volume[opidx],
                    "is_michel": is_michel[opidx]
            }
            update_dict.update(index_dict)
            ohmflashes.append(OrderedDict(update_dict))

        pred_particles = predictor.get_particles(i, only_primaries=False)

        # Loop over predicted particles
        for p in pred_particles:
            direction = get_particle_direction(p, start_segment_radius=start_segment_radius)

            length = -1
            if p.semantic_type == track_label:
                length = compute_track_length(p.points, bin_size=bin_size)
            update_dict = {
                'index': index,
                'id': p.id,
                'volume': p.volume,
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
            update_dict.update(index_dict)
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

        pred_interactions = predictor.get_interactions(i, drop_nonprimary_particles=False, compute_vertex=False)

        for int in pred_interactions:
            update_dict = {
                'index': index,
                'volume': int.volume,
                'id': int.id,
                'size': int.size,
                'num_particles': int.num_particles,
            }
            update_dict.update(index_dict)
            for key in int.particle_counts:
                update_dict[key + '_count'] = int.particle_counts[key]
            for key in int.primary_particle_counts:
                update_dict[key + '_primary_count'] = int.primary_particle_counts[key]
            interactions.append(OrderedDict(update_dict))

    return [particles, interactions, events, opflashes, ohmflashes]
