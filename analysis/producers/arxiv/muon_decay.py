from collections import OrderedDict
from analysis.classes.predictor import FullChainPredictor
from analysis.classes.evaluator import FullChainEvaluator
from lartpc_mlreco3d.analysis.algorithms.arxiv.calorimetry import compute_track_length

from lartpc_mlreco3d.analysis.algorithms.arxiv.decorator import evaluate
from lartpc_mlreco3d.analysis.classes.particle_utils import match_particles_fn, matrix_iou
from analysis.algorithms.arxiv.michel_electrons import get_bounding_box, is_attached_at_edge

from pprint import pprint
import time
import numpy as np
import os, sys
from scipy.spatial.distance import cdist


@evaluate(['michels'])
def muon_decay(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Muon lifetime measurement.

    Use a combination of standard OpFlash and OHMFlash to optically
    tag Michel electrons and extract the time delay between muon and
    Michel electron.
    """

    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})

    pairs = []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['drop_nonprimary_particles']
    use_true_tpc_objects = False #analysis_cfg['analysis'].get('use_true_tpc_objects', False)
    use_depositions_MeV = False #analysis_cfg['analysis'].get('use_depositions_MeV', False)
    ADC_to_MeV = analysis_cfg['analysis'].get('ADC_to_MeV', 1./350.)
    Michel_semantic = analysis_cfg['analysis'].get('Michel_semantic', 2)
    track_semantic = analysis_cfg['analysis'].get('track_semantic', 1)
    flash_michel_window = analysis_cfg['analysis'].get('flash_michel_window', 2) # in us

    # Thresholds
    attached_threshold = processor_cfg.get('attached_threshold', 5)
    one_pixel          = processor_cfg.get('ablation_eps', 5)
    ablation_radius    = processor_cfg.get('ablation_radius', 15)
    ablation_min_samples = processor_cfg.get('ablation_min_samples', 5)
    muon_min_voxel_count = processor_cfg.get('muon_min_voxel_count', 30)
    muon_ohmflash_window = processor_cfg.get('muon_ohmflash_window', 3) # in us
    use_ohmflash_muon = processor_cfg.get('use_ohmflash_muon', False) # Whether to use the OHMFlash muon or the OpFlash muon time
    data = processor_cfg.get('data', True)

    start = time.time()
    if data:
        predictor = FullChainPredictor(data_blob, res, cfg, processor_cfg,
                deghosting=deghosting,
                enable_flash_matching=True,
                flash_matching_cfg=os.path.join(os.environ['FMATCH_BASEDIR'], "dat/flashmatch_112022.cfg"),
                opflash_keys=['opflash_cryoE', 'opflash_cryoW'])
    else:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg,
                deghosting=deghosting,
                enable_flash_matching=True,
                flash_matching_cfg=os.path.join(os.environ['FMATCH_BASEDIR'], "dat/flashmatch_112022.cfg"),
                opflash_keys=['opflash_cryoE', 'opflash_cryoW'])
    print("Muon decay evaluator took %s s" % (time.time() - start))

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

        #opflash_cryoE = predictor.fm.make_flash([data_blob['opflash_cryoE'][idx]])
        #opflash_cryoW = predictor.fm.make_flash([data_blob['opflash_cryoW'][idx]])
        opflash_cryoE = data_blob['opflash_cryoE'][idx]
        opflash_cryoW = data_blob['opflash_cryoW'][idx]
        min_time_cryoE, min_time_cryoW = np.inf, np.inf
        for flash in opflash_cryoE:
            if flash.absTime() < min_time_cryoE:
                min_time_cryoE = flash.absTime()
        for flash in opflash_cryoW:
            if flash.absTime() < min_time_cryoW:
                min_time_cryoW = flash.absTime()
        print('min times ', min_time_cryoE, min_time_cryoW)


        ohmflash_cryoE = data_blob['ohmflash_cryoE'][idx]
        ohmflash_cryoW = data_blob['ohmflash_cryoW'][idx]
        ohmflash_muon_cryoE = data_blob['ohmflash_muon_cryoE'][idx]
        ohmflash_muon_cryoW = data_blob['ohmflash_muon_cryoW'][idx]
        ohmflash_cryoE_times = [f.absTime() for f in ohmflash_cryoE]
        ohmflash_cryoW_times = [f.absTime() for f in ohmflash_cryoW]
        ohmflash_muon_cryoE_times = [f.absTime() for f in ohmflash_muon_cryoE]
        ohmflash_muon_cryoW_times = [f.absTime() for f in ohmflash_muon_cryoW]
        # sort in time the ohm flashes
        perm = np.argsort(ohmflash_cryoE_times)
        ohmflash_cryoE = np.array(ohmflash_cryoE)[perm]
        ohmflash_cryoE_times = np.array(ohmflash_cryoE_times)[perm]
        perm = np.argsort(ohmflash_cryoW_times)
        ohmflash_cryoW = np.array(ohmflash_cryoW)[perm]
        ohmflash_cryoW_times = np.array(ohmflash_cryoW_times)[perm]
        perm = np.argsort(ohmflash_muon_cryoE_times)
        ohmflash_muon_cryoE = np.array(ohmflash_muon_cryoE)[perm]
        ohmflash_muon_cryoE_times = np.array(ohmflash_muon_cryoE_times)[perm]
        perm = np.argsort(ohmflash_muon_cryoW_times)
        ohmflash_muon_cryoW = np.array(ohmflash_muon_cryoW)[perm]
        ohmflash_muon_cryoW_times = np.array(ohmflash_muon_cryoW_times)[perm]

        ohmflash_cryoE_used_index = np.zeros((ohmflash_cryoE.shape[0],), dtype=np.bool)
        ohmflash_cryoW_used_index = np.zeros((ohmflash_cryoW.shape[0],), dtype=np.bool)
        ohmflash_muon_cryoE_used_index = np.zeros((ohmflash_muon_cryoE.shape[0],), dtype=np.bool)
        ohmflash_muon_cryoW_used_index = np.zeros((ohmflash_muon_cryoW.shape[0],), dtype=np.bool)

        start = time.time()
        interactions_cryoE = predictor.get_interactions(idx, drop_nonprimary_particles=False, volume=0, compute_vertex=False)
        interactions_cryoW = predictor.get_interactions(idx, drop_nonprimary_particles=False, volume=1, compute_vertex=False)

        michel_interactions_cryoE, michel_interactions_cryoW = [], []
        for interaction in interactions_cryoE:
            for p in interaction.particles:
                if p.semantic_type != Michel_semantic: continue
                michel_interactions_cryoE.append(interaction.id)
                break
        for interaction in interactions_cryoW:
            for p in interaction.particles:
                if p.semantic_type != Michel_semantic: continue
                michel_interactions_cryoW.append(interaction.id)
                break
        # For now match all interactions
        # michel_interactions_cryoE = [interaction.id for interaction in interactions_cryoE]
        # michel_interactions_cryoW = [interaction.id for interaction in interactions_cryoW]
        print("Michel interactions = ", len(michel_interactions_cryoE), len(michel_interactions_cryoW))
        print('Pre-FM %d s ' % (time.time() - start))

        start = time.time()
        # FIXME run flash matching on all interactions or only the ones we are interested in?
        if len(michel_interactions_cryoE):
            flash_matches_cryoE = predictor.get_flash_matches(idx, use_true_tpc_objects=use_true_tpc_objects, volume=0,
                    use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV, interaction_list=michel_interactions_cryoE)
        else:
            flash_matches_cryoE = []
        if len(michel_interactions_cryoW):
            flash_matches_cryoW = predictor.get_flash_matches(idx, use_true_tpc_objects=use_true_tpc_objects, volume=1,
                    use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV, interaction_list=michel_interactions_cryoW)
        else:
            flash_matches_cryoW = []

        print("Flash matching took %d s" % (time.time() - start))
        if not data: # match particles
            matched_particles = predictor.match_particles(idx, mode='pred_to_true')

        start = time.time()

        for interaction, flash, match in flash_matches_cryoE + flash_matches_cryoW:
            michels, muons = [], []
            for p in interaction.particles:
                if p.semantic_type != Michel_semantic: continue
                michel_is_attached_at_edge = False
                muon = None
                for p2 in interaction.particles:
                    if p2.semantic_type != track_semantic: continue
                    if p2.size < muon_min_voxel_count: continue
                    if not is_attached_at_edge(p.points, p2.points,
                                            attached_threshold=attached_threshold,
                                            one_pixel=one_pixel,
                                            ablation_radius=ablation_radius,
                                            ablation_min_samples=ablation_min_samples): continue
                    michel_is_attached_at_edge = True
                    muon = p2
                    break
                if not michel_is_attached_at_edge: continue
                michels.append(p)
                muons.append(muon)

            if len(michels) == 0: continue

            if len(michels) > 1:
                print("Found > 1 Michels in same interaction")
                # unlikely to happen since they must all meet the condition of
                # touching the muon at the end, but just in case sum them up
                michel = michels[np.argmax([m.size for m in michels])]
                michel.size = np.sum([m.size for m in michels])
                michel.depositions = np.hstack([m.depositions for m in michels])
                michel.points = np.concatenate([m.points for m in michels], axis=0)
                michels = [michel]
                muons = [muons[np.argmax([m.size for m in muons])]]

            print("\nFound a Michel")

            # Record muon endpoint
            if cdist(michels[0].points, [muons[0].startpoint]).min() > cdist(michels[0].points, [muons[0].endpoint]).min():
                endpoint = muons[0].endpoint
            else:
                endpoint = muons[0].startpoint

            # Record distance
            michel_to_muon_distance = cdist(michels[0].points, muons[0].points).min()

            #muon_time = flash.time() + 1500 # FIXME why is absTime always 0.0 here (not in ROOT file) ?
            muon_time = flash.absTime()
            if use_ohmflash_muon:
                ohmflash_muon_times = ohmflash_muon_cryoE_times if interaction.volume == 0 else ohmflash_muon_cryoW_times
                close_ohmflash = np.abs(ohmflash_muon_times - muon_time)
                #print('closest ohmflash muon is at ', ohmflash_muon_times[close_ohmflash.argmin()])
                if np.count_nonzero(close_ohmflash < muon_ohmflash_window) > 0:
                   muon_time = ohmflash_muon_times[close_ohmflash.argmin()]
                   print("correcting muon time from ", flash.absTime(), " to ", muon_time)
            print("muon time is ", flash.absTime(), flash.time())
            # TODO use OHM flash time instead of OpFlash time

            d = OrderedDict(index_dict.copy())
            d['interaction_id'] = interaction.id
            d['interaction_size'] = interaction.size
            d['interaction_id'] = interaction.id
            d['interaction_num_particles'] = interaction.num_particles
            d['interaction_min_x'] = interaction.points[:, 0].min()
            d['interaction_max_x'] = interaction.points[:, 0].max()
            d['interaction_min_y'] = interaction.points[:, 1].min()
            d['interaction_max_y'] = interaction.points[:, 1].max()
            d['interaction_min_z'] = interaction.points[:, 2].min()
            d['interaction_max_z'] = interaction.points[:, 2].max()
            d['interaction_edep'] = interaction.depositions.sum()
            d['muon_time']   = muon_time
            d['muon_size'] = muons[0].size
            d['michel_size'] = michels[0].size
            d['michel_edep'] = michels[0].depositions.sum()
            d['michel_id'] = michels[0].id
            d['michel_time'] = -1
            d['muon_true_time'] = -1
            d['michel_true_time'] = -1
            d['opflash_time'] = flash.absTime()
            d['opflash_pe_sum'] = flash.TotalPE()
            d['volume'] = michels[0].volume
            d['endpoint_x'] = endpoint[0]
            d['endpoint_y'] = endpoint[1]
            d['endpoint_z'] = endpoint[2]
            d['muon_true_pdg'] = -1
            d['muon_true_energy_init'] = -1
            d['muon_true_track_id'] = -1
            d['michel_true_track_id'] = -1
            d['michel_true_pdg'] = -1
            d['michel_true_energy_init'] = -1
            d['michel_true_energy_deposit'] = -1
            d['michel_true_px'] = -1
            d['michel_true_py'] = -1
            d['michel_true_pz'] = -1
            d['michel_true_semantic'] = -1
            d['distance_to_muon'] = michel_to_muon_distance
            d['muon_length'] = compute_track_length(muons[0].points)
            d['min_time_cryoE'] = min_time_cryoE
            d['min_time_cryoW'] = min_time_cryoW

            d.update(get_bounding_box(michels[0].points))

            # Found a Michel in interaction
            # Now looking for OHM flash in same cryostat
            if interaction.volume == 0:
                ohmflash = ohmflash_cryoE[~ohmflash_cryoE_used_index]
                ohmflash_times = ohmflash_cryoE_times[~ohmflash_cryoE_used_index]
            else:
                ohmflash = ohmflash_cryoW[~ohmflash_cryoW_used_index]
                ohmflash_times = ohmflash_cryoW_times[~ohmflash_cryoW_used_index]

            if len(ohmflash) > 0:
                # Finding first OHM flash that immediately follows the muon flash
                #next_flash = ohmflash[np.argmax(ohmflash_times > muon_time)]
                if (ohmflash_times > muon_time).any(): #  OHMFlash located after muon time
                    next_idx = np.where(ohmflash_times > muon_time)[0][0]
                    next_flash = ohmflash[next_idx]
                    michel_time = next_flash.absTime()
                    print("muon and michel time = ", muon_time, michel_time)
                    #if np.abs(michel_time - muon_time) > flash_michel_window: continue

                    # Hurray, we found a pair of muon/Michel flashes!
                    # Time to store
                    d['michel_time'] = michel_time
                    #if interaction.volume == 0:
                    #    ohmflash_cryoE_used_index[np.where(~ohmflash_cryoE_used_index)[0][next_idx]] = True
                    #else:
                    #    ohmflash_cryoW_used_index[np.where(~ohmflash_cryoW_used_index)[0][next_idx]] = True

            if not data:
                # find muon and michel in true particles
                found = 0
                for mp in matched_particles: # matching was done pred 2 true
                    if mp[0] is None or mp[1] is None: continue
                    if mp[0].id == muons[0].id and mp[0].volume == muons[0].volume:
                        d['muon_true_time'] = mp[1].asis.t() * 1e-3 + 1500 #mp[1].asis.t()
                        d['muon_true_energy_init'] = mp[1].asis.energy_init()
                        d['muon_true_pdg'] = mp[1].asis.pdg_code()
                        d['muon_true_track_id'] = mp[1].asis.track_id()
                        found += 1
                    if mp[0].id == michels[0].id and mp[0].volume == michels[0].volume:
                        d['michel_true_time'] = mp[1].asis.t() * 1e-3 + 1500
                        d['michel_true_track_id'] = mp[1].asis.track_id()
                        d['michel_true_pdg'] = mp[1].asis.pdg_code()
                        d['michel_true_energy_init'] = mp[1].asis.energy_init()
                        d['michel_true_energy_deposit'] = mp[1].asis.energy_deposit()
                        d['michel_true_px'] = mp[1].asis.px()
                        d['michel_true_py'] = mp[1].asis.py()
                        d['michel_true_pz'] = mp[1].asis.pz()
                        d['michel_true_semantic'] = mp[1].semantic_type
                        found += 1
                    if found >= 2:
                        break
                print("True times = ", d['muon_true_time'], d['michel_true_time'])
            pairs.append(d)

        print("Loop was %d s" % (time.time() - start))


    return [pairs]
