import numpy as np

from scipy.spatial.distance import cdist

from mlreco.utils.globals import COORD_COLS, PPN_SHAPE_COL
from mlreco.utils.ppn import get_ppn_predictions

from analysis.post_processing import PostProcessor


class PPNProcessor(PostProcessor):
    '''
    Run the PPN post-processing function to produce PPN candidate
    points from the raw PPN output.

    If requested, for each particle, match ppn_points that have hausdorff
    distance less than a threshold and inplace update particle.ppn_candidates/

    If `restrict_semantic_type` is `True`, points will be matched
    to particles with the same predicted semantic type only.
    '''
    name = 'get_ppn_candidates'
    data_cap_opt = ['input_data']
    result_cap = ['segmentation', 'ppn_points', 'ppn_coords',
            'ppn_masks', 'ppn_classify_endpoints']
    result_cap_opt = ['input_rescaled', 'particles']

    def __init__(self,
                 assign_to_particles = False,
                 restrict_semantic_type = False,
                 ppn_distance_threshold = 2,
                 **kwargs):
        '''
        Store the `get_ppn_predictions` keyword arguments


        Parameters
        ----------
        assign_to_particles: bool, default False
            If `True`, will assign PPN candidates to particle objects
        restrict_semantic_type : bool, default False
            If `True`, only associate PPN candidates with compatible shape
        ppn_distance_threshold : float, default 2
            Maximum distance required to assign ppn point to particle
        **kwargs : dict, optional
            Keyword arguments to pass to the `get_ppn_predictions` function

        Returns
        -------
        dict
            Update result dictionary containing 'ppn_candidates' key
        '''
        # Store the relevant parameters
        self.assign_to_particles = assign_to_particles
        self.restrict_semantic_type = restrict_semantic_type
        self.ppn_distance_threshold = ppn_distance_threshold
        self.kwargs = kwargs

    def process(self, data_dict, result_dict):
        '''
        Produce PPN candidates for one entry.


        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Pick the input data to be used
        if 'input_rescaled' not in result_dict.keys():
            input_data = data_dict['input_data']
        else:
            input_data = result_dict['input_rescaled']

        result_nest = {}
        for key, val in result_dict.items():
            result_nest[key] = [val]

        # Get the PPN candidates
        ppn_candidates = get_ppn_predictions(input_data,
                result_nest, apply_deghosting=False, **self.kwargs)
        result_dict['ppn_candidates'] = ppn_candidates

        # If requested, assign PPN candidates to particles
        if self.assign_to_particles:
            valid_mask = np.arange(len(ppn_candidates))
            for p in result_dict['particles']:
                if self.restrict_semantic_type:
                    valid_mask = np.where(ppn_candidates[:, PPN_SHAPE_COL] \
                            == p.shape)

                ppn_points = ppn_candidates[valid_mask][:, COORD_COLS]
                dists = np.min(cdist(ppn_points, p.points), axis=1)
                matches = ppn_candidates[valid_mask][dists \
                        < self.ppn_distance_threshold]
                p.ppn_candidates = matches

        return {}, result_dict
