from mlreco.utils.globals import TRACK_SHP
from mlreco.utils.tracking import check_track_orientation
from mlreco.utils.ppn import check_track_orientation_ppn

from analysis.post_processing import PostProcessor


class ParticleExtremaProcessor(PostProcessor):
    '''
    Assigns track start point and end point.
    '''
    name = 'assign_particle_extrema'
    result_cap = ['particles']
    result_cap_opt = ['ppn_candidates']

    def __init__(self,
                 method='local',
                 **kwargs):
        '''
        Parameters
        ----------
        method : algorithm to correct track startpoint/endpoint misplacement.
            The following modes are available:
            - local: computes local energy deposition density only at
            the extrema and chooses the higher one as the endpoint.
            - gradient: computes local energy deposition density throughout
            the track, computes the overall slope (linear fit) of the energy
            density variation to estimate the direction.
            - ppn: uses ppn candidate predictions (classify_endpoints) to
            assign start and endpoints.
        kwargs : dict
            Extra arguments to pass to the `check_track_orientation` or the
            `check_track_orientation_ppn' functions
        '''
        # Store the orientation method and its arguments
        self.method = method
        self.kwargs = kwargs

    def process(self, data_dict, result_dict):
        '''
        Orient all particles in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        for p in result_dict['particles']:
            if p.semantic_type == TRACK_SHP:
                # Check if the end points need to be flipped
                if self.method in ['local', 'gradient']:
                    flip = not check_track_orientation(p.points,
                            p.depositions, p.start_point, p.end_point,
                            self.method, **self.kwargs)
                elif self.method == 'ppn':
                    assert 'ppn_candidates' in result_dict, \
                            'Must run the get_ppn_predictions post-processor '\
                            'before using PPN predictions to assign  extrema'
                    flip = not check_track_orientation_ppn(p.start_point,
                            p.end_point, result_dict['ppn_candidates'])
                else:
                    raise ValueError('Point assignment method not ' \
                            f'recognized: {self.method}')

                # If needed, flip en end points
                if flip:
                    start_point, end_point = p.end_point, p.start_point
                    p.start_point = start_point
                    p.end_point   = end_point

        return {}, {}
