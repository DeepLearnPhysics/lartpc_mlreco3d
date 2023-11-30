import numpy as np


class GainCalibrator:
    '''
    Converts all charge depositions in ADC to a number of electrons. It can
    either use a flat converstion factor or one per TPC in the detector
    '''
    name = 'gain'

    def __init__(self,
                 gain,
                 num_tpcs):
        '''
        Initialize the recombination model and its constants.

        Parameters
        ----------
        gain : Union[list, float]
            Conversion factor from ADC to electrons (unique or per tpc)
        num_tpcs : int
            Number of TPCs in the detector
        '''
        # Initialize the gain values
        assert np.isscalar(gain) or len(gain) == num_tpcs, \
                'Gain must be a single value or given per TPC'
        self.gain = gain

    def process(self, values, tpc_id):
        '''
        Converts deposition values from ADC to 

        Parameters
        ----------
        values : np.ndarray
            (N) array of depositions in ADC in a specific TPC
        tpc_id : int
            ID of the TPC to use

        Returns
        -------
        np.ndarray
            (N) array of depositions in number of electrons
        '''
        # If the gain is specified globally, use it as is
        if np.isscalar(self.gain):
            return values * self.gain

        # If not, use the gain of the TPC at hand
        return values * self.gain[tpc_id]
