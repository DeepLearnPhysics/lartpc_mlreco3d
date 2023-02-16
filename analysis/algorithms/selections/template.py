from collections import OrderedDict
import os, copy, sys

# Flash Matching
sys.path.append('/sdf/group/neutrino/ldomine/OpT0Finder/python')


from analysis.decorator import evaluate
from analysis.classes.evaluator import FullChainEvaluator
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.Interaction import Interaction
from analysis.classes.Particle import Particle
from analysis.classes.TruthParticle import TruthParticle
from analysis.algorithms.utils import get_interaction_properties, get_particle_properties, get_mparticles_from_minteractions

@evaluate(['interactions', 'particles'], mode='per_batch')
def run_inference(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Example of analysis script for nue analysis.
    """
    # List of ordered dictionaries for output logging
    # Interaction and particle level information
    interactions, particles = [], []

    # Analysis tools configuration
    deghosting            = analysis_cfg['analysis']['deghosting']
    primaries             = analysis_cfg['analysis']['match_primaries']
    enable_flash_matching = analysis_cfg['analysis'].get('enable_flash_matching', False)
    ADC_to_MeV            = analysis_cfg['analysis'].get('ADC_to_MeV', 1./350.)
    compute_vertex        = analysis_cfg['analysis']['compute_vertex']
    vertex_mode           = analysis_cfg['analysis']['vertex_mode']
    matching_mode         = analysis_cfg['analysis']['matching_mode']

    # FullChainEvaluator config
    processor_cfg         = analysis_cfg['analysis'].get('processor_cfg', {})

    # Skeleton for csv output
    interaction_dict      = analysis_cfg['analysis'].get('interaction_dict', {})
    particle_dict         = analysis_cfg['analysis'].get('particle_dict', {})

    use_primaries_for_vertex = analysis_cfg['analysis']['use_primaries_for_vertex']

    # Load data into evaluator
    if enable_flash_matching:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg,
                deghosting=deghosting,
                enable_flash_matching=enable_flash_matching,
                flash_matching_cfg="/sdf/group/neutrino/koh0207/logs/nu_selection/flash_matching/config/flashmatch.cfg",
                opflash_keys=['opflash_cryoE', 'opflash_cryoW'])
    else:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']
    spatial_size = predictor.spatial_size

    # Loop over images
    for idx, index in enumerate(image_idxs):
        index_dict = {
            'Index': index,
            # 'run': data_blob['run_info'][idx][0],
            # 'subrun': data_blob['run_info'][idx][1],
            # 'event': data_blob['run_info'][idx][2]
        }
        if enable_flash_matching:
            flash_matches_cryoE = predictor.get_flash_matches(idx, use_true_tpc_objects=False, volume=0,
                    use_depositions_MeV=False, ADC_to_MeV=ADC_to_MeV)
            flash_matches_cryoW = predictor.get_flash_matches(idx, use_true_tpc_objects=False, volume=1,
                    use_depositions_MeV=False, ADC_to_MeV=ADC_to_MeV)

        # 1. Match Interactions and log interaction-level information
        matches, counts = predictor.match_interactions(idx,
            mode='true_to_pred',
            match_particles=True,
            drop_nonprimary_particles=primaries,
            return_counts=True,
            compute_vertex=compute_vertex,
            vertex_mode=vertex_mode,
            overlap_mode=predictor.overlap_mode,
            matching_mode='optimal')

        # 1 a) Check outputs from interaction matching 
        if len(matches) == 0:
            continue

        particle_matches, particle_matches_values = get_mparticles_from_minteractions(matches)

        # 2. Process interaction level information
        for i, interaction_pair in enumerate(matches):
            int_dict = copy.deepcopy(interaction_dict)

            int_dict.update(index_dict)

            int_dict['interaction_match_counts'] = counts[i]
            true_int, pred_int = interaction_pair[0], interaction_pair[1]

            assert (type(true_int) is TruthInteraction) or (true_int is None)
            assert (type(pred_int) is Interaction) or (pred_int is None)

            true_int_dict = get_interaction_properties(true_int, spatial_size, prefix='true')
            pred_int_dict = get_interaction_properties(pred_int, spatial_size, prefix='pred')
            fmatch_dict = {}
            
            if true_int is not None:
                # This means there is a true interaction corresponding to
                # this predicted interaction. Hence:
                pred_int_dict['pred_interaction_has_match'] = True
                true_int_dict['true_nu_id'] = true_int.nu_id
                if 'neutrino_asis' in data_blob and true_int.nu_id > 0:
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
            if pred_int is not None:
                # Similarly:
                pred_int_dict['pred_vertex_candidate_count'] = pred_int.vertex_candidate_count
                true_int_dict['true_interaction_has_match'] = True

            if enable_flash_matching:
                volume = true_int.volume if true_int is not None else pred_int.volume
                flash_matches = flash_matches_cryoW if volume == 1 else flash_matches_cryoE
                if pred_int is not None:
                    for interaction, flash, match in flash_matches:
                        if interaction.id != pred_int.id: continue
                        fmatch_dict['fmatched'] = True
                        fmatch_dict['fmatch_time'] = flash.time()
                        fmatch_dict['fmatch_total_pe'] = flash.TotalPE()
                        fmatch_dict['fmatch_id'] = flash.id()
                        break

            for k1, v1 in true_int_dict.items():
                if k1 in int_dict:
                    int_dict[k1] = v1
                else:
                    raise ValueError("{} not in pre-defined fieldnames.".format(k1))
            for k2, v2 in pred_int_dict.items():
                if k2 in int_dict:
                    int_dict[k2] = v2
                else:
                    raise ValueError("{} not in pre-defined fieldnames.".format(k2))
            if enable_flash_matching:
                for k3, v3 in fmatch_dict.items():
                    if k3 in int_dict:
                        int_dict[k3] = v3
                    else:
                        raise ValueError("{} not in pre-defined fieldnames.".format(k3))
            interactions.append(int_dict)


        # 3. Process particle level information
        for i, mparticles in enumerate(particle_matches):
            true_p, pred_p = mparticles[0], mparticles[1]

            assert (type(true_p) is TruthParticle) or true_p is None
            assert (type(pred_p) is Particle) or pred_p is None

            part_dict = copy.deepcopy(particle_dict)

            part_dict.update(index_dict)
            part_dict['particle_match_value'] = particle_matches_values[i]

            pred_particle_dict = get_particle_properties(pred_p,
                prefix='pred')
            true_particle_dict = get_particle_properties(true_p,
                prefix='true')

            if true_p is not None:
                pred_particle_dict['pred_particle_has_match'] = True
                true_particle_dict['true_particle_interaction_id'] = true_p.interaction_id
                if 'particles_asis' in data_blob:
                    particles_asis = data_blob['particles_asis'][idx]
                    if len(particles_asis) > true_p.id:
                        true_part = particles_asis[true_p.id]
                        true_particle_dict['true_particle_energy_init'] = true_part.energy_init()
                        true_particle_dict['true_particle_energy_deposit'] = true_part.energy_deposit()
                        true_particle_dict['true_particle_creation_process'] = true_part.creation_process()
                        # If no children other than itself: particle is stopping.
                        children = true_part.children_id()
                        children = [x for x in children if x != true_part.id()]
                        true_particle_dict['true_particle_children_count'] = len(children)

            if pred_p is not None:
                true_particle_dict['true_particle_has_match'] = True
                pred_particle_dict['pred_particle_interaction_id'] = pred_p.interaction_id


            for k1, v1 in true_particle_dict.items():
                if k1 in part_dict:
                    part_dict[k1] = v1
                else:
                    raise ValueError("{} not in pre-defined fieldnames.".format(k1))

            for k2, v2 in pred_particle_dict.items():
                if k2 in part_dict:
                    part_dict[k2] = v2
                else:
                    raise ValueError("{} not in pre-defined fieldnames.".format(k2))

        
            particles.append(part_dict)

    return [interactions, particles]