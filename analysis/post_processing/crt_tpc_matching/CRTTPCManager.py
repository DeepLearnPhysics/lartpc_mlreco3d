import numpy as np

class CRTTPCMatcherInterface:
    """
    Adapter class between full chain outputs and matcha (Python package
    for matching tracks to CRT hits)
    """
    def __init__(self, config, 
                 boundaries=None, crthit_keys=[], **kwargs):

        self.config = config
        self.crthit_keys = crthit_keys

        self.detector_specs = kwargs.get('detector_specs', None)
        self.ADC_to_MeV = kwargs.get('ADC_to_MeV', 1.)
        self.use_depositions_MeV = kwargs.get('use_depositions_MeV', False)
        self.boundaries = kwargs.get('boundaries', None)

        self.crt_tpc_matches = {}
        if self.boundaries is not None:
            self.volume_bounds = VolumeBoundaries(self.boundaries)
            self._num_volumes = self.volume_bounds.num_volumes()
        else:
            self.volume_bounds = None
            self._num_volumes = 1

    def initialize_crt_tpc_manager(self, meta):
        self.crt_tpc_manager = CRTTPCManager(self.config, self.fm_config, 
                                             meta=meta,
                                             detector_specs=self.detector_specs)

class CRTTPCManager:
    """
    Class that manages TPC track and CRT hit objects. Similar to the FlashManager
    class, but does not inherit from it. Interfaces with matcha to perform CRT-TPC 
    matching; see https://github.com/andrewmogan/matcha

    Attributes
    ==========

    Methods
    =======
    """
    def __init__(self, cfg, meta=None):
        """
        Constructor

        Parameters
        ==========
        cfg: dict
            The full chain config.
        meta: np.ndarray, optional, default is None
            Used to shift coordinates of interactions to "real" detector
            coordinates for determining track containment. The structure is 
            [0:3] = image min in detector (x,y,z), 
            [3:6] = image max in detector (x,y,z), and 
            [6:9] = scale in (x,y,z)

        Methods
        =======
        TODO INSERT
        """

        # Setup meta
        self.cfg = cfg

        self.min_x, self.min_y, self.min_z = None, None, None
        self.size_voxel_x, self.size_voxel_y, self.size_voxel_z = None, None, None
        if meta is not None:
            self.min_x = meta[0]
            self.min_y = meta[1]
            self.min_z = meta[2]
            self.max_x = meta[0]
            self.max_y = meta[1]
            self.max_z = meta[2]
            self.size_voxel_x = meta[6]
            self.size_voxel_y = meta[7]
            self.size_voxel_z = meta[8]

        self.crt_tpc_matches = None
        self.tpc_v, self.crt_v, self.trk_v = None, None, None

    def make_crthit(self, larcv_crthits, minimum_pe=50):
        """
        Parameters
        ==========
        larcv_crthits: list of list of larcv::CRTHit

        Returns
        =======
        list of matcha:CRTHit
        """
        from matcha.crthit import CRTHit

        crthits = []
        for branch in larcv_crthits:
            crthits.append(branch)

        crt_v = []
        # Convert larcv::CRTHit to matcha::CRTHit
        for idx, larcv_crthit in enumerate(crthits):
            if larcv_crthit.peshit() < minimum_pe: continue
            crthit_id  = larcv_crthit.id()
            t0_sec     = larcv_crthit.ts0_s()   # seconds-only part of CRTHit timestamp
            t0_ns      = larcv_crthit.ts0_ns()  # nanoseconds part of timestamp
            t1_ns      = larcv_crthit.ts1_ns()  # crthit timing, a candidate T0
            position_x = larcv_crthit.x_pos()
            position_y = larcv_crthit.y_pos()
            position_z = larcv_crthit.z_pos()
            error_x    = larcv_crthit.x_err()
            error_y    = larcv_crthit.y_err()
            error_z    = larcv_crthit.z_err()
            total_pe   = larcv_crthit.peshit()
            plane      = larcv_crthit.plane()
            tagger     = larcv_crthit.tagger()
            this_crthit = CRTHit(
                id=crthit_id, t0_sec=t0_sec, t0_ns=t0_ns, t1_ns=t1_ns,
                position_x=position_x, position_y=position_y, position_z=position_z,
                error_x=error_x, error_y=error_y, error_z=error_z,
                total_pe=total_pe, plane=plane, tagger=tagger
            )

            crt_v.append(this_crthit)

        self.crt_v = crt_v
        return crt_v

    def make_tpctrack(self, muon_candidates):
        from matcha.track import Track
        '''
        Fill matcha::Track() from muons candidates selected from interaction list

        Parameters
        ----------
        interactions: list of Interaction() objects

        Returns
        -------
        list of matcha::Track()
        '''
        trk_v = []

        for idx, particle in enumerate(muon_candidates):
            print('-----TRACK', particle.id, '-------')
            particle.points     = self.points_to_cm(particle.points)
            particle.startpoint = self.points_to_cm(np.array(particle.startpoint).reshape(1, 3))
            particle.endpoint   = self.points_to_cm(np.array(particle.endpoint).reshape(1, 3))
            track_id = particle.id
            image_id = particle.image_id
            interaction_id = particle.interaction_id
            points  = particle.points
            depositions = particle.depositions
            start_x = particle.startpoint[0][0]
            start_y = particle.startpoint[0][1]
            start_z = particle.startpoint[0][2]
            end_x   = particle.endpoint[0][0]
            end_y   = particle.endpoint[0][1]
            end_z   = particle.endpoint[0][2]
            this_track = Track(
                id=track_id, image_id=image_id, interaction_id=interaction_id, 
                points=points, depositions=depositions
                #start_x=start_x, start_y=start_y, start_z=start_z, 
                #end_x=end_x, end_y=end_y, end_z=end_z, 
            )
            trk_v.append(this_track)

        self.trk_v = trk_v
        return trk_v

    def run_crt_tpc_matching(self, tracks, crthits):
        """
        Call matcha's match-making function

        Parameters
        ----------
        tracks: list of matcha.Track instances
        crthits: list of matcha.CRTHit instances

        Returns
        -------
        list of matcha.MatchCandidate instances containing matched Track and
        CRTHit objects
        """
        from matcha import match_maker

        distance_threshold = 50
        dca_method = 'simple'
        direction_method = 'pca'
        pca_radius = 10
        min_points_in_radius = 10
        trigger_timestamp = None # Only necessary if isdata=True
        isdata = False
        save_to_file = True
        file_path = '.'
        crt_tpc_matches = match_maker.get_track_crthit_matches(
            tracks, crthits, 
            approach_distance_threshold=distance_threshold, 
            direction_method=direction_method, dca_method=dca_method, 
            pca_radius=pca_radius, min_points_in_radius=min_points_in_radius,
            trigger_timestamp=trigger_timestamp, isdata=isdata,
            save_to_file=save_to_file, file_path=file_path
        )

        return crt_tpc_matches

    def points_to_cm(self, points):
        """
        Convert particle points from voxel units to cm

        Parameters
        ----------
        points: np.ndarray
            Shape (N, 3). Coordinates in voxel units.

        Returns
        -------
        np.ndarray
            Shape (N, 3). Coordinates in cm.
        """
        if points.shape[1] != 3:
            raise ValueError("points should have shape (N,3) but has shape {}".format(points.shape))
        points_in_cm = np.zeros(shape=(len(points), 3))
        for ip, point in enumerate(points):
            points_in_cm[ip][0] = point[0] * self.size_voxel_x + self.min_x
            points_in_cm[ip][1] = point[1] * self.size_voxel_y + self.min_y
            points_in_cm[ip][2] = point[2] * self.size_voxel_z + self.min_z

        return points_in_cm




