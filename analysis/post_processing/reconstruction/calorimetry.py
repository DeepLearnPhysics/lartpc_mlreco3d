import numpy as np

from mlreco.utils.globals import TRACK_SHP

from analysis.post_processing import PostProcessor


class CalorimetricEnergyProcessor(PostProcessor):
    '''
    Compute calorimetric energy by summing the charge depositions and
    scaling by the ADC to MeV conversion factor.
    '''
    name = 'reconstruct_calo_energy'
    result_cap = ['particles']

    def __init__(self,
                 ADC_to_MeV,
                 shower_fudge=1.):

        '''
        Stores the ADC to MeV conversion factor.
        
        In the future, this will include other position and
        time dependant calibrations.

        Parameters
        ----------
        ADC_to_MeV : Union[float, str]
            Global ADC to MeV conversion factor (can be an expression)
        shower_fudge : Union[float, str]
            Shower energy fudge factor (accounts for missing cluster energy)
        '''
        # Store the conversion factor
        self.factor = ADC_to_MeV
        if isinstance(self.factor, str):
            self.factor = eval(self.factor)

        # Store the shower fudge factor
        self.shower_fudge = shower_fudge
        if isinstance(self.shower_fudge, str):
            self.shower_fudge = eval(self.shower_fudge)

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
            if p.semantic_type != TRACK_SHP:
                factor *= self.shower_fudge

            p.calo_ke = factor * p.depositions_sum
                
        return {}, {}
