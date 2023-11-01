import time

from analysis.classes import TruthParticle, TruthInteraction


class PostProcessor:
    '''
    Base class of all post-processors, provides:
    - Function that checks that the post-processor is provided
      the necessary information to do its job
    '''
    # Shared public attributes of post-processors
    name = ''
    parent_path = ''
    data_cap = []
    data_cap_opt = []
    result_cap = []
    result_cap_opt = []
    truth_point_mode = 'points'
    units = 'cm'

    # Shared private attributes of post-processors
    _run_modes = ['reco', 'truth', 'both', 'all']

    def __init__(self, run_mode = None, truth_point_mode = None):
        '''
        Initialize default post-processor object properties.

        Parameters
        ----------
        run_mode : str, optional
           If specified, tells whether the post-processor must run on
           reconstructed ('reco'), true ('true) or both objects ('both', 'all')
        truth_point_mode : str, optional
           If specified, tells which attribute of the `TruthParticle` object
           to use to fetch its point coordinates
        '''
        # If run mode is specified, process it
        if run_mode is not None:
            # Check that the run mode is recognized
            assert run_mode in self._run_modes, \
                    f'`run_mode` must be one of {self._run_modes}'

            # Make a list of object keys to process
            req_keys = self.result_cap + self.result_cap_opt
            self.part_keys, self.inter_keys = [], []
            if run_mode != 'truth':
                if 'particles' in req_keys:
                    self.part_keys.append('particles')
                if 'interactions' in req_keys:
                    self.inter_keys.append('interactions')
            if run_mode != 'reco':
                if 'truth_particles' in req_keys:
                    self.part_keys.append('truth_particles')
                if 'truth_interactions' in req_keys:
                    self.inter_keys.append('truth_interactions')

            self.all_keys = self.part_keys + self.inter_keys

        # If a truth point mode is specified, store it
        if truth_point_mode is not None:
            self.truth_point_mode = truth_point_mode

    def run(self, data_dict, result_dict, image_id):
        '''
        Runs a post processor on one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        image_id : int
            Entry number in the input/output dictionaries

        Returns
        -------
        data_update : dict
            Update to the input dictionary
        result_update : dict
            Update to the result dictionary
        time : float
            Post-processor execution time
        '''
        # Fetch the necessary information
        data_single, result_single = {}, {}
        for data_key in self.data_cap:
            if data_key in data_dict.keys():
                data_single[data_key] = data_dict[data_key][image_id]
            else:
                msg = f'Unable to find {data_key} in data dictionary while '\
                        f'running post-processor {self.name}.'
                raise KeyError(msg)

        for result_key in self.result_cap:
            if result_key in result_dict.keys():
                result_single[result_key] = result_dict[result_key][image_id]
            else:
                msg = f'Unable to find {result_key} in result dictionary while '\
                        f'running post-processor {self.name}.'
                raise KeyError(msg)

        # Fetch the optional information, if available
        for data_key in self.data_cap_opt:
            if data_key in data_dict.keys():
                data_single[data_key] = data_dict[data_key][image_id]

        for result_key in self.result_cap_opt:
            if result_key in result_dict.keys():
                result_single[result_key] = result_dict[result_key][image_id]

        # Run the post-processor
        start = time.time()
        data_update, result_update = self.process(data_single, result_single)
        end = time.time()
        process_time = end-start

        return data_update, result_update, process_time

    def process(self, data_dict, result_dict):
        '''
        Function which needs to be defined for each post-processor

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        raise NotImplementedError('Must define the `process` function')

    def get_points(self, obj):
        '''
        Get the point coordinates of a Particle/Interaction or
        TruthParticle/TruthInteraction object. The TruthParticle object points
        are obtained using the `truth_point_mode`.

        Parameters
        ----------
        obj : Union[Particle, TruthParticle, Interactio, TruthInteraction]
            Particle or interaction object

        Results
        -------
        np.ndarray
           (N, 3) Point coordinates
        '''
        if not isinstance(obj, (TruthParticle, TruthInteraction)):
            return obj.points
        else:
            return getattr(obj, self.truth_point_mode)

    def check_units(self, obj):
        '''
        Check that the point coordinates of a Particle/Interaction or
        TruthParticle/TruthInteraction object is what is expected.

        Parameters
        ----------
        obj : Union[Particle, TruthParticle, Interactio, TruthInteraction]
            Particle or interaction object

        Results
        -------
        np.ndarray
           (N, 3) Point coordinates
        '''
        if obj.units != self.units:
            raise ValueError(f'Coordinates must be expressed in ' \
                    f'{self.units}, currently in {obj.units} instead.')
