import flashmatch
from flashmatch import flashmatch
from flashmatch import geoalgo
from matcha.track import Track
from matcha.crthit import CRTHit
from matcha.match_candidate import MatchCandidate
import numpy as np
from analysis.decorator import evaluate
from collections import OrderedDict
from analysis.classes.ui import FullChainEvaluator
import argparse
import yaml
from mlreco.main_funcs import process_config

def main(model_config_path, analysis_config_path):
    model_config    = yaml.load(open(model_config_path   , 'r'), Loader=yaml.Loader)
    analysis_config = yaml.load(open(analysis_config_path, 'r'), Loader=yaml.Loader)

    process_config(model_config, verbose=False)

    process_func = eval(analysis_config['analysis']['name'])
    process_func(model_config, analysis_config)

@evaluate(['fmatch'], mode='per_batch')
def opmatch_evaluator(data, res, data_id, ana_cfg, mod_cfg):
    flash_fields = OrderedDict(ana_cfg['flash_matches']['fields'])
    crt_fields   = OrderedDict(ana_cfg['crt_tpc_matches']['fields'])
    rows = list()
    image_id = data['index']
    predictor = FullChainEvaluator(data, res, mod_cfg, ana_cfg,
                                   deghosting=True,
                                   enable_flash_matching=True,
                                   flash_matching_cfg=ana_cfg['fmatch_cfg'],
                                   opflash_keys=ana_cfg['opflash_keys'])

    for i, index in enumerate(image_id):
        print('----Image', i, '-----')
        print('Starting flash matching...')
        fmatches = predictor.get_flash_matches(i, use_true_tpc_objects=True,
                                               volume=None, ADC_to_MeV=0.00285714285)
        cached_interactions = []
        for (interaction, flash, match) in fmatches:
            entry = OrderedDict(flash_fields)
            #populate_entry(entry, index, data['meta'][i],
            #               data['neutrinos'][i] if 'neutrinos' in data.keys() else list(),
            #               interaction, flash, match)
            #rows.append(entry)
            cached_interactions.append(interaction)

        # Use interactions from flash matching as input to CRT-TPC matching
        print('Starting crt-tpc matching...')
        crt_tpc_matches = predictor.get_crt_tpc_matches(i, use_true_tpc_objects=True,
                                                        volume=None, ADC_to_MeV=0.00285714285,
                                                        interaction_list=cached_interactions)
        for match in crt_tpc_matches:
            entry = OrderedDict(crt_fields)
            populate_crt_tpc_entry(entry, index, data['meta'][i],
                                  data['neutrinos'][i] if 'neutrinos' in data.keys() else list(),
                                  match)
            rows.append(entry)
        print('Done')

        # Loop over CRT hits.
        #crthits = data['crthits'][i]
        #for h in crthits:
        #    entry = OrderedDict(crt_fields)
        #    entry = populate_crthit_entry(entry, index, h)

    return [rows]

def populate_entry(entry, index, meta, neutrinos,
                   interaction, flash, match):
    """
    Populates the flash match entry with the relevant information

    Parameters
    ----------
    entry: A dictionary configured according to the analysis config.
    index: The image index.
    meta: Meta information about real spatial coordinates.
    neutrinos: The list of neutrinos in the image.
    interaction: The interaction matched to the flash.
    flash: The flash matched to the interaction.
    match: The match object containing information from OpT0Finder.

    Returns
    -------
    The populated flash match entry.
    """
    print('In populate_entry')
    entry['index'] = index
    entry['interaction_index'] = interaction.id
    entry['score'] = match.score
    entry['totalpe'] = flash.TotalPE()
    entry['totalpe0'] = sum(flash.PEPerOpDet()[0:90])
    entry['totalpe1'] = sum(flash.PEPerOpDet()[90:180])
    entry['totalpe2'] = sum(flash.PEPerOpDet()[180:270])
    entry['totalpe3'] = sum(flash.PEPerOpDet()[270:360])
    entry['tpc_point_x'] = match.tpc_point.x
    entry['tpc_point_xerr'] = match.tpc_point_err.x
    entry['tpc_point_y'] = match.tpc_point.y
    entry['tpc_point_yerr'] = match.tpc_point_err.y
    entry['tpc_point_z'] = match.tpc_point.z
    entry['tpc_point_zerr'] = match.tpc_point_err.z
    entry['flash_time'] = flash.time()
    entry['flash_time_width'] = flash.timeWidth()
    entry['flash_abs_time'] = flash.absTime()
    entry['flash_frame'] = flash.frame()
    entry['flash_center_x'] = flash.xCenter()
    entry['flash_width_x'] = flash.xWidth()
    entry['flash_center_y'] = flash.yCenter()
    entry['flash_width_y'] = flash.yWidth()
    entry['flash_center_z'] = flash.zCenter()
    entry['flash_width_z'] = flash.zWidth()
    entry['flash_fast_to_total'] = flash.fastToTotal()
    entry['flash_in_beam_frame'] = flash.inBeamFrame()
    entry['flash_on_beam_time'] = flash.onBeamTime()
    vtx = [meta[6+i]*interaction.vertex[i] + meta[i] for i in [0,1,2]]
    entry['vtx_x'], entry['vtx_y'], entry['vtx_z'] = vtx
    if np.any([p.is_primary for p in interaction.particles]):
        entry['itime'] = np.mean([p.particle_asis.t() for p in interaction.particles if p.is_primary])
    nus = [(meta[6+0]*n.x()+meta[0],meta[6+1]*n.y()+meta[1],meta[6+2]*n.z()+meta[2]) for n in neutrinos]
    entry['neutrino'] = np.any(near(nus, vtx))
    return entry

def populate_crt_tpc_entry(entry, index, meta, neutrinos, match):
    """
    Populates the CRT-TPC match information.

    Matched CRT hits and TPC tracks are stored in matcha.MatchCandidate
    instances. 

    Parameters
    ----------
    entry: A dictionary configured according to the analysis config.
    index: The image index.
    meta: Meta information about real spatial coordinates.
    neutrinos: The list of neutrinos in the image.
    match: matcha.MatchCandidate instance containing matched CRT hit
           and TPC track objects

    Returns
    -------
    The populated CRT hit entry.
    """
    entry['index'] = index
    entry['crthit_id'] = match.crthit.id
    entry['crthit_total_pe'] = match.crthit.total_pe
    entry['crthit_t0_sec'] = match.crthit.t0_sec
    entry['crthit_t0_ns'] = match.crthit.t0_ns
    entry['crthit_t1_ns'] = match.crthit.t1_ns
    entry['crthit_position_x'] = match.crthit.position_x
    entry['crthit_position_y'] = match.crthit.position_y
    entry['crthit_position_z'] = match.crthit.position_z
    entry['crthit_error_x'] = match.crthit.error_x
    entry['crthit_error_y'] = match.crthit.error_y
    entry['crthit_error_z'] = match.crthit.error_z
    entry['crthit_plane'] = match.crthit.plane
    entry['crthit_tagger'] = match.crthit.tagger
    entry['track_id'] = match.track.id
    entry['track_image_id'] = match.track.image_id
    entry['track_interaction_id'] = match.track.interaction_id
    entry['track_start_x'] = match.track.start_x
    entry['track_start_y'] = match.track.start_y
    entry['track_start_z'] = match.track.start_z
    entry['track_end_x'] = match.track.end_x
    entry['track_end_y'] = match.track.end_y
    entry['track_end_z'] = match.track.end_z
    entry['match_dca'] = match.distance_of_closest_approach

    return entry

def near(nus, vtx, thr=1):
    return [np.sqrt((n[0] - vtx[0])**2 + (n[1] - vtx[1])**2 + (n[2] - vtx[2])**2) < thr for n in nus]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config'   , type=str, required=True)
    parser.add_argument('--analysis_config', type=str, required=True)
    args = parser.parse_args()

    main(args.model_config, args.analysis_config)









