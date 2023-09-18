import numpy as np

from mlreco.utils.globals import TRACK_SHP, PID_MASSES, SHP_TO_PID, SHP_TO_PRIMARY

from analysis.post_processing import post_processing


@post_processing(data_capture=[],
                 result_capture=['particles', 'interactions'])
def enforce_particle_semantics(data_dict, result_dict,
                               enforce_pid=True,
                               enforce_primary=True):
    '''
    Enforce logical connections between semantic predictions and
    particle-level predictions (PID and primary):
    - If a particle has shower shape, it can only have a shower PID
    - If a particle has track shape, it can only have a track PID
    - If a particle has delta/michel shape, it can only be a secondary electron

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    enforce_pid : bool, default True
        Enforce the PID prediction based on the semantic type
    enforce_primary : bool, default True
        Enforce the primary predictionbased on the semantic type
    '''
    # Loop over the particle objects
    for p in result_dict['particles']:
        # Get the particle semantic type
        shape = p.semantic_type

        # Reset the PID scores
        if enforce_pid:
            pid_range = SHP_TO_PID[shape]
            pid_range = pid_range[pid_range < len(p.pid_scores)]

            pid_scores = np.zeros(len(p.pid_scores), dtype=p.pid_scores.dtype)
            pid_scores[pid_range] = p.pid_scores[pid_range]
            pid_scores /= np.sum(pid_scores)
            p.pid_scores = pid_scores

        # Reset the primary scores
        if enforce_primary:
            primary_range = SHP_TO_PRIMARY[shape]

            primary_scores = np.zeros(len(p.primary_scores),
                    dtype=p.primary_scores.dtype)
            primary_scores[primary_range] = p.primary_scores[primary_range]
            primary_scores /= np.sum(primary_scores)
            p.primary_scores = primary_scores

    # Update the interaction information accordingly
    for ia in result_dict['interactions']:
        ia._update_particle_info()

    return {}


@post_processing(data_capture=[],
                 result_capture=['particles', 'interactions'])
def adjust_particle_properties(data_dict, result_dict,
                               em_pid_thresholds={},
                               track_pid_thresholds={},
                               primary_threshold=None):
    '''
    Adjust the particle PID and primary properties according to
    customizable thresholds and priority orderings.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    em_pid_thresholds : dict, optional
        Dictionary which maps an EM PID output to a threshold value, in order
    track_pid_thresholds : dict, optional
        Dictionary which maps a track PID output to a threshold value, in order
    primary_treshold : float, optional
        Primary score above which a paricle is considered a primary
    '''
    # Check that there is something to do, throw otherwise
    if not len(em_pid_thresholds) and not len(track_pid_thresholds) and \
            primary_threshold is None:
        msg = ('Specify one of `em_pid_thresholds`, `track_pid_thresholds`'
               'or `primary_threshold` for this function to do anything.')
        raise ValueError(msg)

    # Loop over the particle objects
    for p in result_dict['particles']:
        # Adjust the particle ID
        pid_thresholds = track_pid_thresholds \
                if p.semantic_type == TRACK_SHP else em_pid_thresholds
        assigned = False
        for k, v in pid_thresholds.items():
            if not assigned and p.pid_scores[k] >= v:
                p.pid = k
                assigned = True
        assert assigned or not len(pid_thresholds), \
                'Must specify a PID threshold for all or no particle type'

        # Adjust the primary ID
        if primary_threshold is not None:
            p.is_primary = p.primary_scores[1] >= primary_threshold

    # Update the interaction information accordingly
    for ia in result_dict['interactions']:
        ia._update_particle_info()

    return {}


@post_processing(data_capture=['input_data'], 
                 result_capture=['particles'])
def reconstruct_momentum(data_dict,
                         result_dict,
                         method='best'):
    '''
    Combines kinetic energy estimates with direction estimates
    to provide the best guess of 3-momentum

    Parameters
    ----------
    data_dict : dict
        Data dictionary (contains one image-worth of data)
    result_dict : dict
        Result dictionary (contains one image-worth of data)
    method : str, defualt 'best'
        Kinematic energy reconstruction method to use
        - 'best': Pick the most appropriate KE source
        - 'csda': Always pick CSDA if available
        - 'calo": Use calorimetry exclusively
        - 'mcs': Pick MCS if available
    '''
    # Check the method is recognized
    if method not in ['best', 'csda', 'calo', 'mcs']:
        raise ValueError(f'Momentum reconstruction method not recognized: {method}')

    # Loop over reconstructed particles
    for p in result_dict['particles']:
        # Check the PID is recognized, otherwise momentum cannot be computed
        if p.pid < 0:
            continue
        if p.pid not in PID_MASSES.keys():
            raise ValueError('PID not recognized, cannot reconstruct momentum')

        # Get the reconstructed kinetic energy
        if p.semantic_type != TRACK_SHP:
            # If a particle is not a track, can only use calorimetric estimate
            ke = p.calo_ke
        else:
            # If a particle is a track, pick csda for contained tracks and
            # pick mcs for uncontained tracks, unless specified otherwise
            if (method == 'csda' or p.is_contained) and p.csda_ke > 0.:
                ke = p.csda_ke
            elif (method == 'mcs' or not p.is_contained) and p.mcs_ke > 0.:
                ke = p.mcs_ke
            else:
                ke = p.calo_ke

        if ke < 0.:
            raise ValueError('Must fill the `*_ke` attributes to fill the momentum')

        # Get the direction
        direction = p.start_dir
        if direction[0] == -np.inf:
            raise ValueError('Must fill the `start_dir` attribute to fill the momentum')

        # Convert the kinetic energy to an absolute momentum value
        mass = PID_MASSES[p.pid]
        momentum = np.sqrt((ke+mass)**2-mass**2)

        # Fill momentum
        p.momentum = momentum * direction
            
    return {}

