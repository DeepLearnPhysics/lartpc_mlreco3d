from collections import OrderedDict

from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import ParticleLogger, InteractionLogger

@write_to(['interactions', 'particles'])
def run_inference(data_blob, res, **kwargs):
    """
    Template for a logging script for particle and interaction objects.

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

    particles: List[List[dict]]
        List of list of dicts, with same structure as <interactions> but with
        per-particle information.

    Information in <interactions> will be saved to $log_dir/interactions.csv
    and <particles> to $log_dir/particles.csv.
    """

    interactions, particles = [], []

    matching_mode         = kwargs['matching_mode']

    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})

    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Index': index,
        }

        # 1. Match Interactions and log interaction-level information
        imatches, icounts = res['matched_interactions'][idx], res['interaction_match_counts'][idx]
        pmatches, pcounts = res['matched_particles'][idx], res['particle_match_counts'][idx]
        # 1 a) Check outputs from interaction matching 
        if len(imatches) > 0:
            # 2. Process interaction level information
            interaction_logger = InteractionLogger(int_fieldnames, meta=meta)
            interaction_logger.prepare()

            for i, interaction_pair in enumerate(imatches):

                int_dict = OrderedDict()
                int_dict.update(index_dict)
                int_dict['interaction_match_counts'] = icounts[i]
                
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

        if len(pmatches) > 0:

            # 3. Process particle level information
            particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
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
                part_dict['particle_match_counts'] = pcounts[i]
                part_dict.update(true_p_dict)
                part_dict.update(pred_p_dict)
                particles.append(part_dict)

    return [interactions, particles]
