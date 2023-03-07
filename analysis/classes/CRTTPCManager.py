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

    def make_crthit(self, larcv_crthits):
        """
        Parameters
        ==========
        larcv_crthits: list of list of larcv::CRTHit

        Returns
        =======
        list of matcha:CRTHit
        """
        from matcha import crthit

        crthits = []
        for branch in larcv_crthits:
            crthits.append(branch)

        crt_v = []
        # Convert larcv::CRTHit to matcha::CRTHit
        for idx, larcv_crthit in enumerate(crthits):
            this_crthit = crthit.CRTHit(larcv_crthit.id())
            this_crthit.total_pe   = larcv_crthit.peshit()
            this_crthit.t0_sec     = larcv_crthit.ts0_s()   # seconds-only part of CRTHit timestamp
            this_crthit.t0_ns      = larcv_crthit.ts0_ns()  # nanoseconds part of timestamp
            this_crthit.t1         = larcv_crthit.ts1_ns()  # crthit timing, a candidate T0
            this_crthit.x_position = larcv_crthit.x_pos()
            this_crthit.x_error    = larcv_crthit.x_err()
            this_crthit.y_position = larcv_crthit.y_pos()
            this_crthit.y_error    = larcv_crthit.y_err()
            this_crthit.z_position = larcv_crthit.z_pos()
            this_crthit.z_error    = larcv_crthit.z_err()
            this_crthit.plane      = larcv_crthit.plane()
            this_crthit.tagger     = larcv_crthit.tagger()

            crt_v.append(crthit)

        self.crt_v = crt_v
        return crt_v

    def make_tpctrack(self, muon_candidates):
        from matcha import track
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
            particle.points = points_to_cm(particle.points)
            this_track = track.Track(id=particle.id)
            this_track.image_id = particle.image_id
            this_track.interaction_id = particle.interaction_id
            this_track.start_x = particle.startpoint[0]
            this_track.start_y = particle.startpoint[1]
            this_track.start_z = particle.startpoint[2]
            this_track.end_x   = particle.endpoint[0]
            this_track.end_y   = particle.endpoint[1]
            this_track.end_z   = particle.endpoint[2]
            print('[MAKE_TPCTRACK] Track info:\n')
            print('\t Track start: ({}, {}, {})'.format(this_track.start_x, this_track.start_y, this_track.start_z))
            print('\t Track end: ({}, {}, {})'.format(this_track.end_x, this_track.end_y, this_track.end_z))
            this_track.points  = particle.points
            this_track.depositions = particle.depositions
            trk_v.append(this_track)

        self.trk_v = trk_v
        return trk_v

    def run_crt_tpc_matching(self, tracks, crthits):
        from matcha import matcher
        print('[CRTTPCManager] Run crt-tpc matching')

        ### Run CRT-TPC matching ###
        # Initialize matcha::Match() class

        crt_tpc_matches = []

        # For each track, loop CRTHits and find DCA
        #for track in tracks:
            #for crthit in crthits:
                #matcher = matcher.MatchMaker(track, crthit)
                #matcher.calculate_distance_of_closest_approach(track, crthit)


        return crt_tpc_matches 
        # TODO Why did I have it return an attribute?
        #return self.crt_tpc_matches 

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
        for ip, point in enumerate(points):
            absolute_x = point[0] * self.size_voxel_x + self.min_x
            absolute_y = point[1] * self.size_voxel_y + self.min_y
            absolute_z = point[2] * self.size_voxel_z + self.min_z




