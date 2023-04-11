from collections import OrderedDict

from analysis.producers.decorator import write_to
from analysis.classes.evaluator import FullChainEvaluator
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.Interaction import Interaction
from analysis.producers.logger import ParticleLogger, InteractionLogger
from pprint import pprint

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
    primaries             = kwargs['match_primaries']
    matching_mode         = kwargs['matching_mode']
    boundaries            = kwargs.get('boundaries', [[1376.3], None, None])

    # FullChainEvaluator config
    evaluator_cfg         = kwargs.get('evaluator_cfg', {})
    # Particle and Interaction processor names
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})

    # Load data into evaluator
    predictor = FullChainEvaluator(data_blob, res, 
                                   evaluator_cfg=evaluator_cfg)
    image_idxs = data_blob['index']

    for idx, index in enumerate(image_idxs):
      
        # For saving per image information
        index_dict = {
            'Index': index,
            # 'run': data_blob['run_info'][idx][0],
            # 'subrun': data_blob['run_info'][idx][1],
            # 'event': data_blob['run_info'][idx][2]
        }

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

        # We access the particle matching information, which is already
        # done by called match_interactions.
        pmatches = predictor._matched_particles
        pcounts  = predictor._matched_particles_counts

        # 2. Process interaction level information
        interaction_logger = InteractionLogger(int_fieldnames)
        interaction_logger.prepare()
        
        # 2-1 Loop over matched interaction pairs
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

        # Loop over matched particle pairs
        for i, mparticles in enumerate(pmatches):
            true_p, pred_p = mparticles[0], mparticles[1]

            true_p_dict = particle_logger.produce(true_p, mode='true')
            pred_p_dict = particle_logger.produce(pred_p, mode='reco')
            
            pprint(true_p_dict)
            assert False

            part_dict = OrderedDict()
            part_dict.update(index_dict)
            part_dict['particle_match_counts'] = pcounts[i]
            part_dict.update(true_p_dict)
            part_dict.update(pred_p_dict)
            particles.append(part_dict)

    return [interactions, particles]
