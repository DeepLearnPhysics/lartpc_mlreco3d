import numpy as np

from mlreco.utils.globals import TRACK_SHP, PID_MASSES

from analysis.post_processing import post_processing


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

