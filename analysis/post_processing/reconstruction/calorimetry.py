import numpy as np

from mlreco.utils.globals import PID_LABELS

from analysis.post_processing import PostProcessor


class CalorimetricEnergyProcessor(PostProcessor):
    '''
    Compute calorimetric energy by summing the charge depositions and
    scaling by the ADC to MeV conversion factor.
    '''
    name = 'reconstruct_calo_energy'
    result_cap = ['particles']

    def __init__(self,
                 conversion_factor=1.):

        '''
        Stores the ADC to MeV conversion factor.
        
        In the future, this will include other position and
        time dependant calibrations.

        Parameters
        ----------
        conversion_factor : Union[float, dict], default 1.
            Voxel value to MeV conversion factor. If a single number, the same
            scaling is used for every particle type. If a dictionary is
            provided, applies different conversions for differnt PID.
        '''
        # Store the conversion factor
        self.factor = conversion_factor

    def process(self, data_dict, result_dict):
        '''
        Reconstruct the calorimetric KE of particles of one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over reconstructed particles
        for p in result_dict['particles']:
            factor = self.factor
            if isinstance(factor, dict):
                if p.pid != -1 and p.pid not in PID_LABELS.keys():
                    raise ValueError(f'Particle species not recognized: ' \
                            '{p.pid}')
                if p.pid not in conversion_factor.keys():
                    raise ValueError(f'Must specify a conversion factor ' \
                            'for particle {p.pid}')
                factor = conversion_factor[p.pid]

            p.calo_ke = factor * p.depositions_sum
                
        return {}, {}
