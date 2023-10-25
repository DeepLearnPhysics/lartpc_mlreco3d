import time

from analysis.classes import TruthParticle, TruthInteraction


class PostProcessor:
    '''
    Base class of all post-processors, provides:
    - Function that checks that the post-processor is provided
      the necessary information to do its job
    '''
    name = ''
    parent_path = ''
    data_cap = []
    data_cap_opt = []
    result_cap = []
    result_cap_opt = []
    truth_point_mode = 'points'
    units = 'cm'

    def __init__(self):
        '''
        Initialize the post-processor object. This is a dummy method
        which exists to give the option not to define an `__init__`
        function in the children post-processor classes.
        '''
        pass

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
