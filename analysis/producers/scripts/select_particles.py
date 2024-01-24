from analysis.producers.decorator import write_to
from analysis.classes.Particle import Particle
from analysis.classes.TruthParticle import TruthParticle
from analysis.classes.TruthParticleFragment import TruthParticleFragment
from analysis.classes.ParticleFragment import ParticleFragment
from analysis.producers.logger import ParticleLogger
from collections import OrderedDict
from collections import defaultdict
import numpy as np
from mlreco.utils.gnn.cluster import cluster_direction
from analysis.post_processing.evaluation.match import generate_match_pairs

@write_to(['particles_r2t', 'particles_t2r'])
def select_particles(data_blob, res, **kwargs):
    '''
    Select matched pairs of particles.  

    Matching is done in two modes: (1) reco-to-true and (2) true-to-reco

    To produce two CSVs containing the particle objects and their matched counterparts, run the following command:
    "python3 lartpc_mlreco3d/analysis/run.py anaconfig.cfg"
    '''

    # Output logging
    particles_r2t, particles_t2r = [], []
    fieldnames = kwargs['logger'].get('particles', {})

    image_idxs = data_blob['index']
    # Loop over images
    for idx, index in enumerate(image_idxs):
        index_dict = {'Iteration': kwargs['iteration'], 'Index': index}
        meta = data_blob['meta'][idx]

        # Retrieve data structures and prepare loggers
        reco_particles = res['particles'][idx]
        true_particles = res['truth_particles'][idx]
        particle_logger = ParticleLogger(fieldnames)
        particle_logger.prepare()
        pmatches_r2t = generate_match_pairs(true_particles, reco_particles)['matches_r2t']
        pmatches_t2r = generate_match_pairs(true_particles, reco_particles)['matches_t2r']
        
        # Loop through reco-2-true matches
        for pmatch in pmatches_r2t:
            reco_p = pmatch[0]
            true_p = pmatch[1]
            assert (type(reco_p) is Particle) or (reco_p) is None
            assert (type(true_p) is TruthParticle) or (true_p) is None

            # Store reco info
            reco_p_dict = particle_logger.produce(reco_p, mode='reco')
            
            # Store matched truth info
            true_p_dict = particle_logger.produce(true_p, mode='true')

            particle_dict = OrderedDict(index_dict.copy())
            particle_dict.update(reco_p_dict)
            particle_dict.update(true_p_dict)
            particles_r2t.append(particle_dict)
    
        # Loop through true-2-reco matches
        for pmatch in pmatches_t2r:
            true_p = pmatch[0]
            reco_p = pmatch[1]
            assert (type(reco_p) is Particle) or (reco_p) is None
            assert (type(true_p) is TruthParticle) or (true_p) is None

            # Store ture info
            true_p_dict = particle_logger.produce(true_p, mode='true')

            # Store matched reco info
            reco_p_dict = particle_logger.produce(reco_p, mode='reco')

            particle_dict = OrderedDict(index_dict.copy())
            particle_dict.update(true_p_dict)
            particle_dict.update(reco_p_dict)
            particles_t2r.append(particle_dict)
            

    return [particles_r2t, particles_t2r]


@write_to(['particles_t2r', 'particles_r2t'])
def run_bidirectional_particles(data_blob, res, **kwargs):
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

    particles_t2r, particles_r2t = [], []

    particle_fieldnames   = kwargs['logger'].get('particles', {})

    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            # 'file_index': data_blob['file_index'][idx]
        }

        # 1. Match Particles
        pmatches, pcounts = res['matched_particles_t2r'][idx], res['particle_match_overlap_t2r'][idx]

        if len(pmatches) > 0:

            # 3. Process particle level information
            particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
            particle_logger.prepare()

            # Loop over matched particle pairs
            for i, mparticles in enumerate(pmatches):
                true_p, pred_p = mparticles[0], mparticles[1]
                
                assert (type(true_p) is TruthParticleFragment) or (true_p) is None
                assert (type(pred_p) is ParticleFragment) or (pred_p) is None

                true_p_dict = particle_logger.produce(true_p, mode='true')
                pred_p_dict = particle_logger.produce(pred_p, mode='reco')

                part_dict = OrderedDict()
                part_dict.update(index_dict)
                part_dict['particle_match_overlap'] = pcounts[i]
                part_dict.update(true_p_dict)
                part_dict.update(pred_p_dict)
                particles_t2r.append(part_dict)
                
        # 1. Match Interactions and log interaction-level information
        pmatches, pcounts = res['matched_particles_r2t'][idx], res['particle_match_overlap_r2t'][idx]

        if len(pmatches) > 0:

            # 3. Process particle level information
            particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
            particle_logger.prepare()

            # Loop over matched particle pairs
            for i, mparticles in enumerate(pmatches):
                
                pred_p, true_p = mparticles[0], mparticles[1]
                
                assert (type(true_p) is TruthParticleFragment) or (true_p) is None
                assert (type(pred_p) is ParticleFragment) or (pred_p) is None

                true_p_dict = particle_logger.produce(true_p, mode='true')
                pred_p_dict = particle_logger.produce(pred_p, mode='reco')

                part_dict = OrderedDict()
                part_dict.update(index_dict)
                part_dict['particle_match_overlap'] = pcounts[i]
                part_dict.update(true_p_dict)
                part_dict.update(pred_p_dict)
                particles_r2t.append(part_dict)

    return [particles_t2r, particles_r2t]