from collections import OrderedDict

from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import ParticleLogger, InteractionLogger

@write_to(['interactions', 'particles'])
def run_inference(data_blob, res, **kwargs):
    """General logging script for particle and interaction level
    information. 

    Parameters
    ----------
    data_blob: dict
        Data dictionary after both model forwarding post-processing
    res: dict
        Result dictionary after both model forwarding and post-processing

    Returns
    -------
    interactions: List[List[dict]]
        List of list of dicts, with length batch_size in the top level
        and length num_interactions (max between true and reco) in the second
        lvel. Each dict corresponds to a row in the generated output file.

    particles: List[List[dict]]
        List of list of dicts, with same structure as <interactions> but with
        per-particle information.

    Information in <interactions> will be saved to $log_dir/interactions.csv
    and <particles> to $log_dir/particles.csv.
    """
    # List of ordered dictionaries for output logging
    # Interaction and particle level information
    interactions, particles = [], []

    # Analysis tools configuration
    matching_mode         = kwargs['matching_mode']
    units                 = kwargs.get('units', 'px')

    # FullChainEvaluator config
    # evaluator_cfg         = kwargs.get('evaluator_cfg', {})
    # Particle and Interaction processor names
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})

    # Load data into evaluator
    # predictor = FullChainEvaluator(data_blob, res, 
    #                                evaluator_cfg=evaluator_cfg)
    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):
      
        # For saving per image information
        index_dict = {
            'Index': index,
            # 'run': data_blob['run_info'][idx][0],
            # 'subrun': data_blob['run_info'][idx][1],
            # 'event': data_blob['run_info'][idx][2]
        }

        # 1. Match Interactions and log interaction-level information
        # if 'matched_interactions' in res:
        matches, icounts = res['matched_interactions'][idx], res['interaction_match_overlap'][idx]
        # else:
        #     print("Running interaction matching...")
        #     matches, icounts = predictor.match_interactions(idx,
        #         matching_mode=matching_mode,
        #         drop_nonprimary_particles=primaries,
        #         return_counts=True)

        # pprint(matches)
        # assert False

        # 1 a) Check outputs from interaction matching 
        if len(matches) == 0:
            continue

        # We access the particle matching information, which is already
        # done by called match_interactions.
        # if 'matched_particles' in res:
        pmatches, pcounts = res['matched_particles'][idx], res['particle_match_overlap'][idx]
        # else:
        #     print("Running particle matching...")
        #     pmatches, pcounts = predictor.match_particles(idx,
        #         matching_mode=matching_mode,
        #         only_primaries=primaries,
        #         return_counts=True)

        # 2. Process interaction level information
        interaction_logger = InteractionLogger(int_fieldnames, meta=meta, units=units)
        interaction_logger.prepare()
        
        # 2-1 Loop over matched interaction pairs
        for i, interaction_pair in enumerate(matches):

            int_dict = OrderedDict()
            int_dict.update(index_dict)
            int_dict['interaction_match_overlap'] = icounts[i]
            
            if matching_mode == 'true_to_pred':
                true_int, pred_int = interaction_pair[0], interaction_pair[1]
            elif matching_mode == 'pred_to_true':
                pred_int, true_int = interaction_pair[0], interaction_pair[1]
            else:
                raise ValueError("Matching mode {} is not supported.".format(matching_mode))

            assert (type(true_int) is TruthInteraction) or (true_int is None)
            assert (type(pred_int) is Interaction) or (pred_int is None)

            true_int_dict = interaction_logger.produce(true_int, mode='true')
            pred_int_dict = interaction_logger.produce(pred_int, mode='reco')
            int_dict.update(true_int_dict)
            int_dict.update(pred_int_dict)
            interactions.append(int_dict)

        # 3. Process particle level information
        particle_logger = ParticleLogger(particle_fieldnames, meta=meta, units=units)
        particle_logger.prepare()

        # Loop over matched particle pairs
        for i, mparticles in enumerate(pmatches):
            if matching_mode == 'true_to_pred':
                true_p, pred_p = mparticles[0], mparticles[1]
            elif matching_mode == 'pred_to_true':
                pred_p, true_p = mparticles[0], mparticles[1]
            else:
                raise ValueError("Matching mode {} is not supported.".format(matching_mode))
            
            assert (type(true_p) is TruthParticle) or (true_p) is None
            assert (type(pred_p) is Particle) or (pred_p) is None

            true_p_dict = particle_logger.produce(true_p, mode='true')
            pred_p_dict = particle_logger.produce(pred_p, mode='reco')

            part_dict = OrderedDict()
            part_dict.update(index_dict)
            part_dict['particle_match_overlap'] = pcounts[i]
            part_dict.update(true_p_dict)
            part_dict.update(pred_p_dict)
            particles.append(part_dict)

    return [interactions, particles]
