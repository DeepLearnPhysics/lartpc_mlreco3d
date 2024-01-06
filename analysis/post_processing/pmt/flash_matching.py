import numpy as np
from warnings import warn

from analysis.post_processing import PostProcessor

from .barycenter import BarycenterFlashMatcher
from .likelihood import LikelihoodFlashMatcher


class FlashMatchingProcessor(PostProcessor):
    '''
    Associates TPC interactions with optical flashes.
    '''
    name = 'run_flash_matching'
    data_cap_opt = ['opflash', 'opflash_cryoE', 'opflash_cryoW']
    result_cap = ['interactions']

    def __init__(self,
                 opflash_map,
                 method = 'likelihood',
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
        # If there is no map from flash data product to volume ID, throw
        self.opflash_map = opflash_map

        # Initialize the flash matching algorithm
        if method == 'barycenter':
            self.matcher = BarycenterFlashMatcher(**kwargs)
        elif method == 'likelihood':
            self.matcher = LikelihoodFlashMatcher(**kwargs, \
                    parent_path=self.parent_path)
        else:
            raise ValueError(f'Flash matching method not recognized: {method}')


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
        interactions = result_dict['interactions']
        if not len(interactions):
            return {}, {}

        # Make sure the interaction coordinates are expressed in cm
        self.check_units(interactions[0])

        # Clear previous flash matching information
        for ii in interactions:
            if ii.fmatched:
                ii.fmatched = False
                ii.flash_id = -1
                ii.flash_time = -np.inf
                ii.flash_total_pE = -1.0
                ii.flash_hypothesis = -1.0

        # Loop over flash keys
        for key, volume_id in self.opflash_map.items():
            # Get the list of flashes associated with that key
            opflashes = data_dict[key]

            # Get the list of interactions that share the same volume
            ints = [ii for ii in interactions if ii.volume_id == volume_id]

            # Run flash matching
            fmatches = self.matcher.get_matches(ints, opflashes)

            # Store flash information
            for ii, flash, match in fmatches:
                ii.fmatched = True
                ii.flash_id = int(flash.id())
                ii.flash_time = float(flash.time())
                ii.flash_total_pE = float(flash.TotalPE())
                if hasattr(match, 'hypothesis'):
                    ii.flash_hypothesis = float(np.array(match.hypothesis,
                        dtype=np.float32).sum())

        return {}, {}
