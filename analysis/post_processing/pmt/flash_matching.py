import numpy as np
from warnings import warn

from analysis.post_processing import PostProcessor

from .barycenter import BarycenterFlashMatcher
from .likelihood import LikelihoodFlashMatcher

OPFLASH_KEYS = np.array(['opflash', 'opflash_cryoE', 'opflash_cryoW'])


class FlashMatchingProcessor(PostProcessor):
    '''
    Associates TPC interactions with optical flashes.
    '''
    name = 'run_flash_matching'
    data_cap = ['index'] # TODO: should not need
    data_cap_opt = ['opflash', 'opflash_cryoE', 'opflash_cryoW']
    result_cap = ['interactions']

    def __init__(self,
                 method = 'likelihood',
                 opflash_keys = [], # deprecated
                 opflash_map = {},
                 **kwargs):
        '''
        Initialize the flash matching algorithm

        Parameters
        ----------
        method : str, default 'likelihood'
            Flash matching method (one of 'likelihood' or 'barycenter')
        opflash_map : dict
            Maps a flash data product key in the data ditctionary to an
            optical volume in the detector
        **kwargs : dict
            Keyword arguments to pass to specific flash matching algorithms
        '''
        # If there is no map from flash data product to volume ID, assume the
        # flash keys are given in order of optical volumes
        if not len(opflash_map):
            assert len(opflash_keys), 'Must provide `opflash_map`'
            opflash_map = {k:i for i, k in enumerate(opflash_keys)}
            warn('The `opflash_keys` argument is deprecated, ' \
                    'provide opflash_map instead', DeprecationWarning)
        assert len(opflash_map), \
                'Did not specify any optical flash keys, nothing to do'
        self.opflash_map = opflash_map

        # Initialize the flash matching algorithm
        if method == 'barycenter':
            self.matcher = BarycenterFlashMatcher(**kwargs)
        elif method == 'likelihood':
            kwargs['opflash_keys'] = list(opflash_map.keys()) # TODO: Must go
            self.matcher = LikelihoodFlashMatcher(**kwargs, \
                    parent_path=self.parent_path)
        else:
            raise ValueError(f'Flash matching method not recognized: {method}')
        self.method = method # TODO: should not need this


    def process(self, data_dict, result_dict):
        '''
        Find [interaction, flash] pairs

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary

        Notes
        -----
        This post-processor modifies the list of `interaction` objectss
        in-place by adding the following attributes:
            interaction.fmatched: (bool)
                Indicator for whether the given interaction has a flash match
            interaction.fmatch_time: float
                The flash time in microseconds
            interaction.fmatch_total_pE: float
            interaction.fmatch_id: int
        '''
        # Check if the TPC coordinates are in cm
        entry        = data_dict['index'] # TODO: get rid of this absurdity
        interactions = result_dict['interactions']
        if not len(interactions):
            return {}

        # Make sure the interaction coordinates are expressed in cm
        self.check_units(interactions[0])

        # Clear previous flash matching information
        for ia in interactions:
            if ia.fmatched:
                ia.fmatched = False
                ia.flash_id = -1
                ia.flash_time = -np.inf
                ia.flash_total_pE = -1
                ia.flash_hypothesis = -1

        # Loop over flash keys
        for key, volume_id in self.opflash_map.items():
            # Get the list of flashes associated with that key
            opflashes = data_dict[key]

            # Run flash matching
            if self.method == 'likelihood':
                # TODO: get rid of entry and volume ID
                opflashes = data_dict # TODO: Must go
                fmatches = self.matcher.get_matches(int(entry), interactions,
                        opflashes, volume=volume_id)
            else:
                fmatches = self.matcher.get_matches(interactions, opflashes)

            # Store flash information
            for ia, flash, match in fmatches:
                ia.fmatched = True
                ia.flash_id = int(flash.id())
                ia.flash_time = float(flash.time())
                ia.flash_total_pE = float(flash.TotalPE())
                if match is not None:
                    ia.flash_hypothesis = float(np.array(match.hypothesis, 
                        dtype=np.float64).sum())

        return {}, {}
