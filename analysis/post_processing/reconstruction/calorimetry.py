import numpy as np

from analysis.post_processing import post_processing
from mlreco.utils.globals import VALUE_COL


@post_processing(data_capture=['input_data'], 
                 result_capture=['input_rescaled',
                                 'particle_clusts'])
def calorimetric_energy(data_dict,
                        result_dict,
                        conversion_factor=1.):
    '''
    Compute calorimetric energy by summing the charge depositions and
    scaling by the ADC to MeV conversion factor.

    Parameters
    ----------
    data_dict : dict
        Data dictionary (contains one image-worth of data)
    result_dict : dict
        Result dictionary (contains one image-worth of data)
    conversion_factor : float, optional
        ADC to MeV conversion factor (MeV / ADC), by default 1.

    Returns
    -------
    update_dict: dict
        Dictionary to be included into result dictionary, containing the
        computed energy under the key 'particle_calo_energy'.
    '''

    input_data     = data_dict['input_data'] if 'input_rescaled' not in result_dict else result_dict['input_rescaled']
    particles      = result_dict['particle_clusts']

    update_dict = {
        'particle_calo_energy': conversion_factor*np.array([np.sum(input_data[p, VALUE_COL]) for p in particles])
    }
            
    return update_dict
