from analysis.producers.decorator import write_to
from analysis.classes.Particle import Particle
from analysis.classes.TruthParticle import TruthParticle
from analysis.classes.Interaction import Interaction
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.evaluator import FullChainEvaluator
from analysis.producers.logger import ParticleLogger, InteractionLogger
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
