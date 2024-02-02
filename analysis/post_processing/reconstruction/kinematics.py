import numpy as np

from mlreco.utils.globals import TRACK_SHP, PID_MASSES, \
        SHP_TO_PID, SHP_TO_PRIMARY

from analysis.post_processing import PostProcessor


class ParticleSemanticsProcessor(PostProcessor):
    '''
    Enforce logical connections between semantic predictions and
    particle-level predictions (PID and primary):
    - If a particle has shower shape, it can only have a shower PID
    - If a particle has track shape, it can only have a track PID
    - If a particle has delta/michel shape, it can only be a secondary electron
    '''
    name = 'enforce_particle_semantics'
    result_cap = ['particles', 'interactions']

    def __init__(self,
                 enforce_pid=True,
                 enforce_primary=True):
        '''
        Store information about which particle properties should
        or should not be updated.

        Parameters
        ----------
        enforce_pid : bool, default True
            Enforce the PID prediction based on the semantic type
        enforce_primary : bool, default True
            Enforce the primary predictionbased on the semantic type
        '''
        # Store parameters
        self.enforce_pid = enforce_pid
        self.enforce_primary = enforce_primary

    def process(self, data_dict, result_dict):
        '''
        Update PID and primary predictions of one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over the particle objects
        for p in result_dict['particles']:
            # Get the particle semantic type
            shape = p.semantic_type

            # Reset the PID scores
            if self.enforce_pid:
                pid_range = SHP_TO_PID[shape]
                pid_range = pid_range[pid_range < len(p.pid_scores)]

                pid_scores = np.zeros(len(p.pid_scores),
                        dtype=p.pid_scores.dtype)
                pid_scores[pid_range] = p.pid_scores[pid_range]
                pid_scores /= np.sum(pid_scores)
                p.pid_scores = pid_scores

            # Reset the primary scores
            if self.enforce_primary:
                primary_range = SHP_TO_PRIMARY[shape]

                primary_scores = np.zeros(len(p.primary_scores),
                        dtype=p.primary_scores.dtype)
                primary_scores[primary_range] = p.primary_scores[primary_range]
                primary_scores /= np.sum(primary_scores)
                p.primary_scores = primary_scores

        # Update the interaction information accordingly
        for ia in result_dict['interactions']:
            ia._update_particle_info()

        return {}, {}


class ParticlePropertiesProcessor(PostProcessor):
    '''
    Adjust the particle PID and primary properties according to
    customizable thresholds and priority orderings.
    '''
    name = 'adjust_particle_properties'
    result_cap = ['particles', 'interactions']

    def __init__(self,
                 em_pid_thresholds={},
                 track_pid_thresholds={},
                 primary_threshold=None):
        '''
        Store the new thresholds to be used to update the PID
        and primary information of particles.

        Parameters
        ----------
        em_pid_thresholds : dict, optional
            Dictionary which maps an EM PID output to a threshold value,
            in order
        track_pid_thresholds : dict, optional
            Dictionary which maps a track PID output to a threshold value,
            in order
        primary_treshold : float, optional
            Primary score above which a particle is considered a primary
        '''
        # Check that there is something to do, throw otherwise
        if not len(em_pid_thresholds) and not len(track_pid_thresholds) and \
                primary_threshold is None:
            msg = ('Specify one of `em_pid_thresholds`, `track_pid_thresholds`'
                   ' or `primary_threshold` for this function to do anything.')
            raise ValueError(msg)

        # Store the thresholds
        self.em_pid_thresholds = em_pid_thresholds
        self.track_pid_thresholds = track_pid_thresholds
        self.primary_threshold = primary_threshold

    def process(self, data_dict, result_dict):
        '''
        Update PID and primary predictions of one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over the particle objects
        for p in result_dict['particles']:
            # Adjust the particle ID
            pid_thresholds = self.track_pid_thresholds \
                    if p.semantic_type == TRACK_SHP else self.em_pid_thresholds
            if len(pid_thresholds):
                assigned = False
                scores = np.copy(p.pid_scores)
                for k, v in pid_thresholds.items():
                    if scores[k] >= v:
                        p.pid = k
                        assigned = True
                        break
                    else:
                        scores *= 1./(1 - scores[k])

                assert assigned, 'Must specify a ' \
                        'PID threshold for all or no particle type'

            # Adjust the primary ID
            if self.primary_threshold is not None:
                p.is_primary = p.primary_scores[1] >= self.primary_threshold

        # Update the interaction information accordingly
        for ia in result_dict['interactions']:
            ia._update_particle_info()

        return {}, {}


class InteractionTopologyProcessor(PostProcessor):
    '''
    Adjust the topology of interactions by applying thresholds
    on the minimum kinetic energy of particles.
    '''
    name = 'adjust_interaction_topology'
    result_cap = ['interactions']
    result_cap_opt = ['truth_interactions']

    def __init__(self,
                 ke_thresholds,
                 reco_ke_mode='ke',
                 truth_ke_mode='energy_deposit',
                 run_mode='both'):
        '''
        Store the new thresholds to be used to update the PID
        and primary information of particles.

        Parameters
        ----------
        ke_thresholds : Union[float, dict]
            If a scalr, it specifies a blanket KE cut to apply to all
            particles. If it is a dictionary, it maps an PID to a KE threshold.
            If a 'default' key is provided, it is used for all particles,
            unless a number is provided for a specific PID.
        reco_ke_mode : str, default 'ke'
            Which `Particle` attribute to use to apply the KE thresholds
        truth_ke_mode : str, default 'energy_deposit'
            Which `TruthParticle` attribute to use to apply the KE thresholds
        '''
        # Initialize the run mode
        super().__init__(run_mode)

        # Store the attributes that should be used to evaluate the KE
        self.reco_ke_mode = reco_ke_mode
        self.truth_ke_mode = truth_ke_mode

        # Check that there is something to do, throw otherwise
        if not len(ke_thresholds):
            msg = 'Specify `ke_thresholds` for this function to do anything.'
            raise ValueError(msg)

        # Store the thresholds in a dictionary
        if np.isscalar(ke_thresholds):
            ke_thresholds = {'default': float(ke_thresholds)}

        self.ke_thresholds = {}
        for pid in PID_MASSES.keys():
            if pid in ke_thresholds:
                self.ke_thresholds[pid] = ke_thresholds[pid]
            elif 'default' in ke_thresholds:
                self.ke_thresholds[pid] = ke_thresholds['default']
            else:
                self.ke_thresholds[pid] = 0.

    def process(self, data_dict, result_dict):
        '''
        Update PID and primary predictions of one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over the interaction types
        for k in self.inter_keys:
            # Check which attribute should be used for KE
            ke_attr = self.reco_ke_mode \
                    if 'truth' not in k else self.truth_ke_mode

            # Loop over interactions
            for ii in result_dict[k]:
                # Loop over particles, select the ones that pass a threshold
                for p in ii.particles:
                    ke = getattr(p, ke_attr)
                    if ke_attr == 'energy_init' and p.pid > 0:
                        ke -= PID_MASSES[p.pid]
                    if p.pid > -1 and ke < self.ke_thresholds[p.pid]:
                        p.is_valid = False
                    else:
                        p.is_valid = True

                # Update the interaction particle counts
                ii._update_particle_info()

        return {}, {}
