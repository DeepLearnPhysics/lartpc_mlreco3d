import copy
from collections import OrderedDict

from analysis.decorator import evaluate
from analysis.classes.evaluator import FullChainEvaluator
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.Interaction import Interaction
from analysis.algorithms.utils import get_mparticles_from_minteractions
from analysis.algorithms.logger import ParticleLogger, InteractionLogger

@evaluate(['interactions', 'particles'])
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
    compute_energy        = analysis_cfg['analysis'].get('compute_energy', False)
    flash_matching_cfg    = analysis_cfg['analysis'].get('flash_matching_cfg', '')
    tag_pi0               = analysis_cfg['analysis'].get('tag_pi0', False)

    # FullChainEvaluator config
    processor_cfg         = analysis_cfg['analysis'].get('processor_cfg', {})

    # Skeleton for csv output
    # interaction_dict      = analysis_cfg['analysis'].get('interaction_dict', {})

    particle_fieldnames   = analysis_cfg['analysis'].get('particle_fieldnames', {})
    int_fieldnames        = analysis_cfg['analysis'].get('interaction_fieldnames', {})

    use_primaries_for_vertex = analysis_cfg['analysis'].get('use_primaries_for_vertex', True)
    run_reco_vertex = analysis_cfg['analysis'].get('run_reco_vertex', False)
    test_containment = analysis_cfg['analysis'].get('test_containment', False)

    # Load data into evaluator
    if enable_flash_matching:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg,
                enable_flash_matching=enable_flash_matching,
                flash_matching_cfg=flash_matching_cfg,
                opflash_keys=['opflash_cryoE', 'opflash_cryoW'])
    else:
        predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg)

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
            matching_mode=matching_mode,
            tag_pi0=tag_pi0)

        # 1 a) Check outputs from interaction matching 
        if len(matches) == 0:
            continue

        particle_matches, particle_match_counts = get_mparticles_from_minteractions(matches)

        # 2. Process interaction level information
        interaction_logger = InteractionLogger(int_fieldnames)
        interaction_logger.prepare()
        for i, interaction_pair in enumerate(matches):

            int_dict = OrderedDict()
            int_dict.update(index_dict)
            int_dict['interaction_match_counts'] = counts[i]
            true_int, pred_int = interaction_pair[0], interaction_pair[1]

            assert (type(true_int) is TruthInteraction) or (true_int is None)
            assert (type(pred_int) is Interaction) or (pred_int is None)

            true_int_dict = interaction_logger.produce(true_int, mode='true')
            pred_int_dict = interaction_logger.produce(pred_int, mode='reco')
            int_dict.update(true_int_dict)
            int_dict.update(pred_int_dict)
            interactions.append(int_dict)

        # 3. Process particle level information
        particle_logger = ParticleLogger(particle_fieldnames)
        particle_logger.prepare()

        for i, mparticles in enumerate(particle_matches):
            true_p, pred_p = mparticles[0], mparticles[1]

            true_p_dict = particle_logger.produce(true_p, mode='true')
            pred_p_dict = particle_logger.produce(pred_p, mode='reco')

            part_dict = OrderedDict()
            part_dict.update(index_dict)
            part_dict['particle_match_counts'] = particle_match_counts[i]
            part_dict.update(true_p_dict)
            part_dict.update(pred_p_dict)
            particles.append(part_dict)

    return [interactions, particles]
