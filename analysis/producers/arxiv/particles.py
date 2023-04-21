from collections import OrderedDict
import os, copy, sys

# Flash Matching
sys.path.append('/sdf/group/neutrino/ldomine/OpT0Finder/python')


from lartpc_mlreco3d.analysis.algorithms.arxiv.decorator import evaluate
from analysis.classes.evaluator import FullChainEvaluator
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.Interaction import Interaction
from analysis.classes.Particle import Particle
from analysis.classes.TruthParticle import TruthParticle
from analysis.algorithms.utils import get_particle_properties

from lartpc_mlreco3d.analysis.algorithms.arxiv.calorimetry import get_csda_range_spline

@evaluate(['particles'])
def run_inference_particles(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Analysis tools inference script for particle-level information.
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
    particle_dict         = analysis_cfg['analysis'].get('particle_dict', {})

    use_primaries_for_vertex = analysis_cfg['analysis'].get('use_primaries_for_vertex', True)

    splines = {
        'proton': get_csda_range_spline('proton'),
        'muon': get_csda_range_spline('muon')
    }

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

        particle_matches, particle_matches_values = predictor.match_particles(idx,
            only_primaries=primaries,
            mode='true_to_pred',
            volume=None,
            matching_mode=matching_mode,
            return_counts=True
            )

        # 3. Process particle level information
        for i, mparticles in enumerate(particle_matches):
            true_p, pred_p = mparticles[0], mparticles[1]

            assert (type(true_p) is TruthParticle) or true_p is None
            assert (type(pred_p) is Particle) or pred_p is None

            part_dict = copy.deepcopy(particle_dict)

            part_dict.update(index_dict)
            part_dict['particle_match_value'] = particle_matches_values[i]

            pred_particle_dict = get_particle_properties(pred_p,
                prefix='pred', splines=splines)
            true_particle_dict = get_particle_properties(true_p,
                prefix='true', splines=splines)

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

    return [particles]
