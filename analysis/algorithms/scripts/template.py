from collections import OrderedDict

from analysis.algorithms.decorator import write_to
from analysis.classes.evaluator import FullChainEvaluator
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.Interaction import Interaction
from analysis.algorithms.logger import ParticleLogger, InteractionLogger


@write_to(['interactions', 'particles'])
def run_inference(data_blob, res, **kwargs):
    """
    Example of analysis script for nue analysis.
    """
    # List of ordered dictionaries for output logging
    # Interaction and particle level information
    interactions, particles = [], []

    # Analysis tools configuration
    primaries             = kwargs['match_primaries']
    enable_flash_matching = kwargs.get('enable_flash_matching', False)
    ADC_to_MeV            = kwargs.get('ADC_to_MeV', 1./350.)
    matching_mode         = kwargs['matching_mode']
    flash_matching_cfg    = kwargs.get('flash_matching_cfg', '')
    boundaries            = kwargs.get('boundaries', [[1376.3], None, None])

    # FullChainEvaluator config
    evaluator_cfg         = kwargs.get('evaluator_cfg', {})
    # Particle and Interaction processor names
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})

    # Load data into evaluator
    if enable_flash_matching:
        predictor = FullChainEvaluator(data_blob, res, 
                predictor_cfg=evaluator_cfg,
                enable_flash_matching=enable_flash_matching,
                flash_matching_cfg=flash_matching_cfg,
                opflash_keys=['opflash_cryoE', 'opflash_cryoW'])
    else:
        predictor = FullChainEvaluator(data_blob, res, 
                                       evaluator_cfg=evaluator_cfg,
                                       boundaries=boundaries)

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
        matches, icounts = predictor.match_interactions(idx,
            mode='true_to_pred',
            match_particles=True,
            drop_nonprimary_particles=primaries,
            return_counts=True,
            overlap_mode=predictor.overlap_mode,
            matching_mode=matching_mode)

        # 1 a) Check outputs from interaction matching 
        if len(matches) == 0:
            continue

        pmatches, pcounts = predictor.match_parts_within_ints(matches)

        # 2. Process interaction level information
        interaction_logger = InteractionLogger(int_fieldnames)
        interaction_logger.prepare()
        
        for i, interaction_pair in enumerate(matches):

            int_dict = OrderedDict()
            int_dict.update(index_dict)
            int_dict['interaction_match_counts'] = icounts[i]
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

        for i, mparticles in enumerate(pmatches):
            true_p, pred_p = mparticles[0], mparticles[1]

            true_p_dict = particle_logger.produce(true_p, mode='true')
            pred_p_dict = particle_logger.produce(pred_p, mode='reco')

            part_dict = OrderedDict()
            part_dict.update(index_dict)
            part_dict['particle_match_counts'] = pcounts[i]
            part_dict.update(true_p_dict)
            part_dict.update(pred_p_dict)
            particles.append(part_dict)

    return [interactions, particles]
