from collections import OrderedDict

from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import ParticleLogger, InteractionLogger

@write_to(['pid_metrics'])
def pid_metrics(data_dict, result_dict,
                iteration=-1,
                logger=None,
                matching_mode='pred_to_true',
                primary_only=True,
                mpv_only=True):
    '''
    Script which stores the scores, predictions and labels related
    to particle identification in order to evaluate PID performance.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary after post-processing
    result_dict : dict
        Chain output dictionary after post-processing
    primary_only : bool, default True
        If True, only store particles corresponding to primary particles
    mpv_only : bool, default True
        If True, only store particles corresponding to particles from MPV interactions

    Returns
    -------
    List[dict]
        One dictionary of relevant particle attribute per particle in
        the image being processed.
        
    Information in <pid_metrics> will be saved to $log_dir/pid_metrics.csv.
    '''
    # Initialize index dictionary to be shared by all entries
    index_dict = {
        'iteration': iteration,
        'index': data_dict['index'][0]
    }

    # Get list of matched particle objects
    assert matching_mode in ['true_to_pred', 'pred_to_true'], \
            f'Matching mode is not recognized: {matching_mode}'

    if 'matched_particles' in result_dict:
        matches, counts = (result_dict['matched_particles'][0],
                result_dict['particle_match_overlap'][0])
    else:
        suffix = 't2r' if matching_mode == 'true_to_pred' else 'r2t'
        matches, counts = (result_dict[f'matched_particles_{suffix}'][0],
                result_dict[f'particle_match_overlap_{suffix}'][0])

    # Loop over matches
    metrics = []
    for i, mparticles in enumerate(matches):
        # If there is no principal match, skip
        if mparticles[1] is None:
            continue

        # Fetch particle objects
        if matching_mode == 'true_to_pred':
            true_p, pred_p = mparticles[0], mparticles[1]
            n_classes = len(true_p.pid_scores)
        elif matching_mode == 'pred_to_true':
            pred_p, true_p = mparticles[0], mparticles[1]
            n_classes = len(pred_p.pid_scores)
        
        assert (type(true_p) is TruthParticle) or (true_p) is None
        assert (type(pred_p) is Particle) or (pred_p) is None

        # Apply restrictions, if requested
        if primary_only and not true_p.is_primary:
            continue
        if mpv_only and not true_p.nu_id > -1:
            continue

        # Build dictionary
        metrics_dict = OrderedDict()
        metrics_dict.update(index_dict)
        metrics_dict['true_id'] = true_p.id
        metrics_dict['pred_id'] = pred_p.id
        metrics_dict['is_primary'] = int(true_p.is_primary)
        metrics_dict['nu_id'] = true_p.nu_id
        metrics_dict['match_overlap'] = counts[i]
        metrics_dict['label'] = true_p.pid
        metrics_dict['pred'] = pred_p.pid
        for c in range(n_classes):
            metrics_dict[f'score_{c}'] = pred_p.pid_scores[c]

        # Append
        metrics.append(metrics_dict)

    return [metrics]
