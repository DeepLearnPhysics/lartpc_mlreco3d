import numpy as np

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
        from matcha.crthit import CRTHit

        crthits = []
        for branch in larcv_crthits:
            crthits.append(branch)

        crt_v = []
        # Convert larcv::CRTHit to matcha::CRTHit
        for idx, larcv_crthit in enumerate(crthits):
            this_crthit = CRTHit(larcv_crthit.id())
            this_crthit.total_pe   = larcv_crthit.peshit()
            this_crthit.t0_sec     = larcv_crthit.ts0_s()   # seconds-only part of CRTHit timestamp
            this_crthit.t0_ns      = larcv_crthit.ts0_ns()  # nanoseconds part of timestamp
            this_crthit.t1         = larcv_crthit.ts1_ns()  # crthit timing, a candidate T0
            this_crthit.position_x = larcv_crthit.x_pos()
            this_crthit.position_y = larcv_crthit.y_pos()
            this_crthit.position_z = larcv_crthit.z_pos()
            this_crthit.error_x    = larcv_crthit.x_err()
            this_crthit.error_y    = larcv_crthit.y_err()
            this_crthit.error_z    = larcv_crthit.z_err()
            this_crthit.plane      = larcv_crthit.plane()
            this_crthit.tagger     = larcv_crthit.tagger()
            print('[MAKECRTHIT]', this_crthit)

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
            this_track = Track(id=particle.id)
            this_track.image_id = particle.image_id
            this_track.interaction_id = particle.interaction_id
            this_track.start_x = particle.startpoint[0][0]
            this_track.start_y = particle.startpoint[0][1]
            this_track.start_z = particle.startpoint[0][2]
            this_track.end_x   = particle.endpoint[0][0]
            this_track.end_y   = particle.endpoint[0][1]
            this_track.end_z   = particle.endpoint[0][2]
            this_track.points  = particle.points
            this_track.depositions = particle.depositions
            start, end = this_track.get_endpoints()
            trk_v.append(this_track)
            print('[MAKETPCTRACK]', this_track)
            print('[MAKETPCTRACK] alternative start/end:\n')
            print('\t', start, end)

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
        crt_tpc_matches = match_maker.get_track_crthit_matches(
            tracks, crthits, approach_distance_threshold=50, save_to_file=True
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




