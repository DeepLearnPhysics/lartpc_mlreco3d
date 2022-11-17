from collections import OrderedDict
from analysis.algorithms.utils import count_primary_particles, get_particle_properties
from analysis.classes.ui import FullChainEvaluator

from analysis.decorator import evaluate
from analysis.classes.particle import match_particles_fn, matrix_iou

from pprint import pprint
import time
import numpy as np

# Setup OpT0finder
import os, sys
sys.path.append('/sdf/group/neutrino/ldomine/OpT0Finder/python')
import flashmatch
from flashmatch import flashmatch, geoalgo


@evaluate(['interactions', 'particles'], mode='per_batch')
def debug_pid(data_blob, res, data_idx, analysis_cfg, cfg):

    interactions, particles = [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['match_primaries']
    enable_flash_matching = analysis_cfg['analysis'].get('enable_flash_matching', False)
    ADC_to_MeV = analysis_cfg['analysis'].get('ADC_to_MeV', 1./350.)

    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})
    if enable_flash_matching:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg,
                deghosting=deghosting,
                enable_flash_matching=True,
                flash_matching_cfg=os.path.join(os.environ['FMATCH_BASEDIR'], "dat/flashmatch_112022.cfg"),
                opflash_keys=['opflash_cryoE', 'opflash_cryoW'])
    else:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg)

    image_idxs = data_blob['index']
    print(data_blob['index'], data_blob['run_info'])
    for idx, index in enumerate(image_idxs):
        index_dict = {
            'Index': index,
            'run': data_blob['run_info'][idx][0],
            'subrun': data_blob['run_info'][idx][1],
            'event': data_blob['run_info'][idx][2]
        }
        if enable_flash_matching:
            flash_matches_cryoE = predictor.get_flash_matches(idx, use_true_tpc_objects=False, volume=0,
                    use_depositions_MeV=False, ADC_to_MeV=ADC_to_MeV)
            flash_matches_cryoW = predictor.get_flash_matches(idx, use_true_tpc_objects=False, volume=1,
                    use_depositions_MeV=False, ADC_to_MeV=ADC_to_MeV)

        # Process Interaction Level Information
        matches, counts = predictor.match_interactions(idx,
            mode='true_to_pred',
            match_particles=True,
            drop_nonprimary_particles=primaries,
            return_counts=True)

        matched_pred_indices = []
        for i, interaction_pair in enumerate(matches):
            true_int, pred_int = interaction_pair[0], interaction_pair[1]
            if pred_int is not None:
                matched_pred_indices.append(pred_int.id)

        pred_interactions = predictor.get_interactions(idx, drop_nonprimary_particles=primaries)
        for int in pred_interactions:
            if int.id not in matched_pred_indices:
                matches.append((None, int))
                if isinstance(counts, list) or isinstance(counts, np.ndarray):
                    counts = np.concatenate([counts, [0]])
                else: counts = np.array([counts, 0])

        for i, interaction_pair in enumerate(matches):
            true_int, pred_int = interaction_pair[0], interaction_pair[1]
            true_int_dict = count_primary_particles(true_int, prefix='true')
            pred_int_dict = count_primary_particles(pred_int, prefix='pred')
            pred_int_dict['true_interaction_matched'] = False
            if true_int is not None and pred_int is not None:
                    pred_int_dict['true_interaction_matched'] = True
            # Store neutrino information
            true_int_dict['true_nu_id'] = -1
            true_int_dict['true_nu_interaction_type'] = -1
            true_int_dict['true_nu_interaction_mode'] = -1
            true_int_dict['true_nu_current_type'] = -1
            true_int_dict['true_nu_energy'] = -1
            if true_int is not None:
                true_int_dict['true_nu_id'] = true_int.nu_id
            if 'neutrino_asis' in data_blob and true_int is not None and true_int.nu_id > 0:
                # assert 'particles_asis' in data_blob
                # particles = data_blob['particles_asis'][i]
                neutrinos = data_blob['neutrino_asis'][idx]
                if len(neutrinos) > 1 or len(neutrinos) == 0: continue
                nu = neutrinos[0]
                # Get larcv::Particle objects for each
                # particle of the true interaction
                # true_particles = np.array(particles)[np.array([p.id for p in true_int.particles])]
                # true_particles_track_ids = [p.track_id() for p in true_particles]
                # for nu in neutrinos:
                #     if nu.mct_index() not in true_particles_track_ids: continue
                true_int_dict['true_nu_interaction_type'] = nu.interaction_type()
                true_int_dict['true_nu_interaction_mode'] = nu.interaction_mode()
                true_int_dict['true_nu_current_type'] = nu.current_type()
                true_int_dict['true_nu_energy'] = nu.energy_init()

            pred_int_dict['interaction_match_counts'] = counts[i]

            if enable_flash_matching:
                volume = true_int.volume if true_int is not None else pred_int.volume
                flash_matches = flash_matches_cryoW if volume == 1 else flash_matches_cryoE
                pred_int_dict['fmatched'] = False
                pred_int_dict['fmatch_time'] = None
                pred_int_dict['fmatch_total_pe'] = None
                pred_int_dict['fmatch_id'] = None
                if pred_int is not None:
                    for interaction, flash, match in flash_matches:
                        if interaction.id != pred_int.id: continue
                        pred_int_dict['fmatched'] = True
                        pred_int_dict['fmatch_time'] = flash.time()
                        pred_int_dict['fmatch_total_pe'] = flash.TotalPE()
                        pred_int_dict['fmatch_id'] = flash.id()
                        break

            interactions_dict = OrderedDict(index_dict.copy())
            interactions_dict.update(true_int_dict)
            interactions_dict.update(pred_int_dict)
            interactions.append(interactions_dict)

            # Process particle level information
            pred_particles, true_particles = [], []
            if pred_int is not None:
                pred_particles = pred_int.particles
            if true_int is not None:
                true_particles = true_int.particles
            matched_particles, _, ious = match_particles_fn(true_particles,
                                                            pred_particles)
            for i, m in enumerate(matched_particles):
                particles_dict = OrderedDict(index_dict.copy())
                true_p, pred_p = m[0], m[1]
                pred_particle_dict = get_particle_properties(pred_p,
                    vertex=pred_int.vertex,
                    prefix='pred')
                true_particle_dict = get_particle_properties(true_p,
                    vertex=true_int.vertex,
                    prefix='true')
                pred_particle_dict['true_particle_is_matched'] = False
                if pred_p is not None and true_p is not None:
                    pred_particle_dict['true_particle_is_matched'] = True

                pred_particle_dict['particle_match_counts'] = ious[i]

                true_particle_dict['true_interaction_id'] = true_int.id
                pred_particle_dict['pred_interaction_id'] = pred_int.id

                true_particle_dict['true_particle_energy_init'] = -1
                true_particle_dict['true_particle_energy_deposit'] = -1
                true_particle_dict['true_particle_children_count'] = -1
                if 'particles_asis' in data_blob:
                    particles_asis = data_blob['particles_asis'][idx]
                    if len(particles_asis) > true_p.id:
                        true_part = particles_asis[true_p.id]
                        true_particle_dict['true_particle_energy_init'] = true_part.energy_init()
                        true_particle_dict['true_particle_energy_deposit'] = true_part.energy_deposit()
                        # If no children other than itself: particle is stopping.
                        children = true_part.children_id()
                        children = [x for x in children if x != true_part.id()]
                        true_particle_dict['true_particle_children_count'] = len(children)

                particles_dict.update(pred_particle_dict)
                particles_dict.update(true_particle_dict)

                particles.append(particles_dict)

    return [interactions, particles]
