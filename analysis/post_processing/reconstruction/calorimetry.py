import numpy as np

from mlreco.utils.globals import PID_LABELS

from analysis.post_processing import post_processing


@post_processing(data_capture=['input_data'], 
                 result_capture=['particles'])
def reconstruct_calo_energy(data_dict,
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
    conversion_factor : Union[float, dict], default 1.
        Voxel value to MeV conversion factor. If a single number, the same
        scaling is used for every particle type. If a dictionary is provided,
    '''
    # Loop over reconstructed particles
    for p in result_dict['particles']:
        factor = conversion_factor
        if isinstance(factor, dict):
            if p.pid != -1 and p.pid not in PID_LABELS.keys():
                raise ValueError(f'Particle species not recognized:{p.pid}')
            if p.pid not in conversion_factor.keys():
                raise ValueError(f'Must specify a conversion factor for particle {p.pid}')
            factor = conversion_factor[p.pid]

        p.calo_ke = factor * p.depositions_sum
            
    return {}
