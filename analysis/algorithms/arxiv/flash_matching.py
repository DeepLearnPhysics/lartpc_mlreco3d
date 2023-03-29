from collections import OrderedDict
from analysis.classes.evaluator import FullChainEvaluator

from analysis.decorator import evaluate

from pprint import pprint
import time
import numpy as np
import os, sys


def find_true_time(interaction):
    """
    Returns
    =======
    Time in us
    """
    time = None
    for p in interaction.particles:
        if not p.is_primary: continue
        time = 1e-3 * p.asis.ancestor_t() if time is None else min(time, 1e-3 * p.particle_asis.ancestor_t())
    return time


def find_true_x(interaction):
    """
    Returns
    =======
    True vertex x in cm (absolute coordinates)
    """
    x = []
    for p in interaction.particles:
        if not p.is_primary: continue
        #x.append(p.asis.x())
        #x.append(p.asis.ancestor_position().x())
        x.append(p.asis.first_step().x())
    if len(x) == 0:
        return None
    values, counts = np.unique(x, return_counts=True)
    if len(values) > 1:
        print("Warning found > 1 true x in interaction", values, counts)
    return values[np.argmax(counts)]


@evaluate(['interactions', 'flashes', 'matches'], mode='per_batch')
def flash_matching(data_blob, res, data_idx, analysis_cfg, cfg):
    # Setup OpT0finder
    #sys.path.append('/sdf/group/neutrino/ldomine/OpT0Finder/python')
    #import flashmatch
    #from flashmatch import flashmatch, geoalgo

    interactions, flashes, matches = [], [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['drop_nonprimary_particles']
    use_true_tpc_objects = analysis_cfg['analysis'].get('use_true_tpc_objects', False)
    use_depositions_MeV = analysis_cfg['analysis'].get('use_depositions_MeV', False)
    ADC_to_MeV = analysis_cfg['analysis'].get('ADC_to_MeV', 1./350.)
    flashmatch_cfg = analysis_cfg['analysis'].get('flashmatch_cfg', 'flashmatch_112022.cfg')

    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})
    predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg,
            deghosting=deghosting,
            enable_flash_matching=True,
            flash_matching_cfg=os.path.join(os.environ['FMATCH_BASEDIR'], "dat/%s" % flashmatch_cfg),
            opflash_keys=['opflash_cryoE', 'opflash_cryoW'])

    image_idxs = data_blob['index']
    print(data_idx, data_blob['index'], data_blob['run_info'])
    for idx, index in enumerate(image_idxs):
        index_dict = {
            'Index': index,
            'run': data_blob['run_info'][idx][0],
            'subrun': data_blob['run_info'][idx][1],
            'event': data_blob['run_info'][idx][2]
        }
        meta = data_blob['meta'][idx]

        opflash_cryoE = predictor.fm.make_flash([data_blob['opflash_cryoE'][idx]])
        opflash_cryoW = predictor.fm.make_flash([data_blob['opflash_cryoW'][idx]])

        all_times_cryoE, all_times_cryoW = [], []
        for flash in data_blob['opflash_cryoE'][idx]:
            all_times_cryoE.append(flash.time())
        for flash in data_blob['opflash_cryoW'][idx]:
            all_times_cryoW.append(flash.time())
        ordered_flashes_cryoE = np.array(data_blob['opflash_cryoE'][idx])[np.argsort(all_times_cryoE)]
        ordered_flashes_cryoW = np.array(data_blob['opflash_cryoW'][idx])[np.argsort(all_times_cryoW)]

        prev_flash_time, next_flash_time = {}, {}
        for flash_idx, flash in enumerate(ordered_flashes_cryoE):
            if flash_idx > 0:
                prev_flash_time[(0, flash.id())] = ordered_flashes_cryoE[flash_idx-1].time()
            else:
                prev_flash_time[(0, flash.id())] = None
            if flash_idx < len(ordered_flashes_cryoE)-1:
                next_flash_time[(0, flash.id())] = ordered_flashes_cryoE[flash_idx+1].time()
            else:
                next_flash_time[(0, flash.id())] = None
        for flash_idx, flash in enumerate(ordered_flashes_cryoW):
            if flash_idx > 0:
                prev_flash_time[(1, flash.id())] = ordered_flashes_cryoW[flash_idx-1].time()
            else:
                prev_flash_time[(1, flash.id())] = None
            if flash_idx < len(ordered_flashes_cryoW)-1:
                next_flash_time[(1, flash.id())] = ordered_flashes_cryoW[flash_idx+1].time()
            else:
                next_flash_time[(1, flash.id())] = None

        flash_matches_cryoE = predictor.get_flash_matches(idx, use_true_tpc_objects=use_true_tpc_objects, volume=0,
                use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV)
        flash_matches_cryoW = predictor.get_flash_matches(idx, use_true_tpc_objects=use_true_tpc_objects, volume=1,
                use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV)

        matched_interactions = None
        if not use_true_tpc_objects:
            matched_interactions = predictor.match_interactions(idx,
                    mode='pred_to_true', drop_nonprimary_particles=primaries, match_particles=True, compute_vertex=False)

        interaction_ids, flash_ids = [], []
        for interaction, flash, match in flash_matches_cryoE + flash_matches_cryoW:
            interaction_ids.append(interaction.id)
            flash_ids.append(flash.id())

            interaction_dict = OrderedDict(index_dict.copy())

            interaction_dict['interaction_id'] = interaction.id
            interaction_dict['size'] = interaction.size
            interaction_dict['num_particles'] = interaction.num_particles
            interaction_dict['interaction_min_x'] = interaction.points[:, 0].min()
            interaction_dict['interaction_max_x'] = interaction.points[:, 0].max()
            interaction_dict['interaction_min_y'] = interaction.points[:, 1].min()
            interaction_dict['interaction_max_y'] = interaction.points[:, 1].max()
            interaction_dict['interaction_min_z'] = interaction.points[:, 2].min()
            interaction_dict['interaction_max_z'] = interaction.points[:, 2].max()
            interaction_dict['interaction_edep'] = interaction.depositions.sum()
            interaction_dict['fmatched'] = True
            interaction_dict['volume'] = interaction.volume

            if not use_true_tpc_objects: # Using TruthInteraction
                for pred_int, true_int in matched_interactions:
                    if pred_int.id != interaction.id: continue
                    if true_int is None:
                        interaction_dict['matched'] = False
                        interaction_dict['true_time'] = None
                        interaction_dict['true_x'] = None
                    else:
                        interaction_dict['matched'] = True
                        interaction_dict['true_time'] = find_true_time(true_int)
                        interaction_dict['true_x'] = find_true_x(true_int)
            else:
                interaction_dict['true_time'] = find_true_time(interaction)
                interaction_dict['true_x'] = find_true_x(interaction)
                interaction_dict['interaction_edep_MeV'] = interaction.depositions_MeV.sum()

            flash_dict = OrderedDict(index_dict.copy())

            flash_dict['flash_id'] = flash.id()
            flash_dict['time'] = flash.time()
            flash_dict['total_pe'] = flash.TotalPE()
            flash_dict['abstime'] = flash.absTime()
            flash_dict['time_width'] = flash.timeWidth()
            flash_dict['fmatched'] = True
            flash_dict['volume'] = interaction.volume
            flash_dict['prev_flash_time'] = prev_flash_time[(interaction.volume, flash.id())]
            flash_dict['next_flash_time'] = next_flash_time[(interaction.volume, flash.id())]

            interactions.append(interaction_dict)
            flashes.append(flash_dict)
            match_dict = flash_dict.copy()
            match_dict.update(interaction_dict)
            match_dict['fmatch_score'] = match.score
            # Convert from absolute cm to voxel coordinates
            match_dict['fmatch_x'] = (match.tpc_point.x - meta[0]) / meta[6]
            match_dict['hypothesis_total_pe'] = np.sum(match.hypothesis)
            match_dict['tpc_point_x'] = match.tpc_point.x
            match_dict['tpc_point_y'] = match.tpc_point.y
            match_dict['tpc_point_z'] = match.tpc_point.z
            match_dict['tpc_point_error_x'] = match.tpc_point_err.x
            match_dict['tpc_point_error_y'] = match.tpc_point_err.y
            match_dict['tpc_point_error_z'] = match.tpc_point_err.z
            match_dict['touch_match'] = match.touch_match # 0 = NoTouchMatch, 1 = AnodeCrossing, 2 = CathodeCrossing, 3 = AnodeCathodeCrossing
            match_dict['touch_match_score'] = match.touch_score
            match_dict['touch_point_x'] = match.touch_point.x
            match_dict['touch_point_y'] = match.touch_point.y
            match_dict['touch_point_z'] = match.touch_point.z
            match_dict['duration'] = match.duration
            match_dict['num_steps'] = match.num_steps
            match_dict['minimizer_min_x'] = match.minimizer_min_x
            match_dict['minimizer_max_x'] = match.minimizer_max_x

            matches.append(match_dict)

        if use_true_tpc_objects:
            all_interactions = predictor.get_true_interactions(idx, drop_nonprimary_particles=primaries, compute_vertex=False)
        else:
            all_interactions = predictor.get_interactions(idx, drop_nonprimary_particles=primaries, compute_vertex=False)

        for interaction in all_interactions:
            if interaction.id in interaction_ids: continue

            interaction_dict = OrderedDict(index_dict.copy())
            interaction_dict['interaction_id'] = interaction.id
            interaction_dict['size'] = interaction.size
            interaction_dict['num_particles'] = interaction.num_particles
            interaction_dict['interaction_min_x'] = interaction.points[:, 0].min()
            interaction_dict['interaction_max_x'] = interaction.points[:, 0].max()
            interaction_dict['interaction_min_y'] = interaction.points[:, 1].min()
            interaction_dict['interaction_max_y'] = interaction.points[:, 1].max()
            interaction_dict['interaction_min_z'] = interaction.points[:, 2].min()
            interaction_dict['interaction_max_z'] = interaction.points[:, 2].max()
            interaction_dict['interaction_edep'] = interaction.depositions.sum()
            interaction_dict['fmatched'] = False

            if not use_true_tpc_objects: # Using TruthInteraction
                for pred_int, true_int in matched_interactions:
                    if pred_int.id != interaction.id: continue
                    if true_int is None:
                        interaction_dict['matched'] = False
                        interaction_dict['true_time'] = None
                        interaction_dict['true_x'] = None
                    else:
                        interaction_dict['matched'] = True
                        interaction_dict['true_time'] = find_true_time(true_int)
                        interaction_dict['true_x'] = find_true_x(true_int)
            else:
                interaction_dict['true_time'] = find_true_time(interaction)
                interaction_dict['true_x'] = find_true_x(interaction)
                interaction_dict['interaction_edep_MeV'] = interaction.depositions_MeV.sum()
            interactions.append(interaction_dict)

        volume = [0] * len(data_blob['opflash_cryoE'][idx])
        volume += [1] * len(data_blob['opflash_cryoW'][idx])
        for flash_idx, flash in enumerate(data_blob['opflash_cryoE'][idx] + data_blob['opflash_cryoW'][idx]):
            if flash.id() in flash_ids: continue
            flash_dict = OrderedDict(index_dict.copy())

            flash_dict['flash_id'] = flash.id()
            flash_dict['time'] = flash.time()
            flash_dict['total_pe'] = flash.TotalPE()
            flash_dict['abstime'] = flash.absTime()
            flash_dict['time_width'] = flash.timeWidth()
            flash_dict['fmatched'] = False
            flash_dict['volume'] = volume[flash_idx]
            flash_dict['prev_flash_time'] = prev_flash_time[(volume[flash_idx], flash.id())]
            flash_dict['next_flash_time'] = next_flash_time[(volume[flash_idx], flash.id())]
            flashes.append(flash_dict)

    return [interactions, flashes, matches] #[interactions, flashes]
