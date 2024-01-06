import os, sys
import numpy as np
import time

from mlreco.utils.geometry import Geometry


class LikelihoodFlashMatcher:
    '''
    Interface class between full chain outputs and OpT0Finder

    See https://github.com/drinkingkazu/OpT0Finder for more details about it.
    '''
    def __init__(self,
                 fmatch_config,
                 parent_path = '',
                 reflash_merging_window = None,
                 detector = None,
                 boundary_file = None,
                 ADC_to_MeV = 1.0,
                 use_depositions_MeV = False):
        '''
        Initialize the likelihood-based flash matching algorithm

        Parameters
        ----------
        fmatch_config : str
            Flash matching configuration file path
        parent_path : str, optional
            Path to the parent configuration file (allows for relative paths)
        reflash_merging_window : float, optional
            Maximum time between successive flashes to be considered a reflash
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        ADC_to_MeV : float, default 1.0
            Conversion factor between ADC and MeV
        use_depositions_MeV, default `False`
            If `True`, uses true energy depositions
        '''
        # Initialize the flash manager (OpT0Finder wrapper)
        self.initialize_backend(fmatch_config, parent_path)

        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file)

        # Get the external parameters
        self.reflash_merging_window = reflash_merging_window
        self.use_depositions_MeV = use_depositions_MeV
        self.ADC_to_MeV = ADC_to_MeV
        if isinstance(self.ADC_to_MeV, str):
            self.ADC_to_MeV = eval(self.ADC_to_MeV)

        # Initialize flash matching attributes
        self.matches = None
        self.qcluster_v = None
        self.flash_v = None

    def initialize_backend(self,
                           fmatch_config,
                           parent_path):
        '''
        Initialize OpT0Finder (backend).

        Expects that the environment variable `FMATCH_BASEDIR` is set.
        You can either set it by hand (to the path where one can find
        OpT0Finder) or you can source `OpT0Finder/configure.sh` if you
        are running code from a command line.

        Parameters
        ----------
        fmatch_config: str
            Path to config for OpT0Finder
        parent_path : str, optional
            Path to the parent configuration file (allows for relative paths)
        '''
        # Add OpT0finder python interface to the python path
        basedir = os.getenv('FMATCH_BASEDIR')
        if basedir is None:
            msg = 'You need to source OpT0Finder configure.sh '\
                'first, or set the FMATCH_BASEDIR environment variable.'
            raise Exception(msg)
        sys.path.append(os.path.join(basedir, 'python'))

        # Add the OpT0Finder library to the dynamic link loader
        lib_path = os.path.join(basedir, 'build/lib')
        os.environ['LD_LIBRARY_PATH'] = '%s:%s' \
                % (lib_path, os.environ['LD_LIBRARY_PATH'])

        # Add the OpT0Finder data directory if it is not yet set
        if 'FMATCH_DATADIR' not in os.environ:
            os.environ['FMATCH_DATADIR'] = os.path.join(basedir, 'dat')

        # Load up the detector specifications
        from flashmatch import flashmatch
        flashmatch.DetectorSpecs.GetME(
                os.path.join(basedir, 'dat/detector_specs.cfg'))

        # Fetch and initialize the OpT0Finder configuration
        if not os.path.isfile(fmatch_config):
            fmatch_config = os.path.join(parent_path, fmatch_config)
            if not os.path.isfile(fmatch_config):
                raise FileNotFoundError('Cannot find flash-matcher config')

        cfg = flashmatch.CreatePSetFromFile(fmatch_config)

        # Initialize The OpT0Finder flash match manager
        self.mgr = flashmatch.FlashMatchManager()
        self.mgr.Configure(cfg)

        # Get the light path algorithm to produce QCluster_t objects
        self.light_path = \
                flashmatch.CustomAlgoFactory.get().create('LightPath',
                        'ToyMCLightPath')
        self.light_path.Configure(cfg.get['flashmatch::PSet']('LightPath'))

    def get_matches(self,
                    interactions,
                    opflashes):
        '''
        Find TPC interactions compatible with optical flashes.

        Parameters
        ----------
        interactions : List[Union[Interaction, TruthInteraction]]
            List of TPC interactions
        opflashes : List[larcv.Flash]
            List of optical flashes

        Returns
        -------
        list of tuple (Interaction, larcv::Flash, flashmatch::FlashMatch_t)
        '''
        # If there's no interactions or no flashes, nothing to do
        if not len(interactions) or not len(opflashes):
            return []

        # Get the volume ID in which the interactions live
        volume_ids = np.empty(len(interactions), dtype=np.int64)
        for i, ii in enumerate(interactions):
            if len(ii.sources):
                modules, tpcs = self.geo.get_contributors(ii.sources)
                assert len(np.unique(modules)) == 1, 'Cannot match ' \
                        'interactions that originate from > 1 optical volumes'
                volume_ids[i] = modules[0]
            else:
                volume_ids[i] = ii.volume_id

        assert len(np.unique(volume_ids)) == 1, \
                'Should only provide interactions from a single optical volume'
        self.volume_id = volume_ids[0]

        # Build a list of QCluster_t (OpT0Finder interaction representation)
        self.qcluster_v = self.make_qcluster_list(interactions)

        # Build a list of Flash_t (OpT0Finder optical flash representation)
        self.flash_v, opflashes = self.make_flash_list(opflashes)

        # Running flash matching and caching the results
        self.matches = self.run_flash_matching()

        return [(interactions[m.tpc_id], opflashes[m.flash_id], m) \
                for m in self.matches]

    def make_qcluster_list(self, interactions):
        '''
        Converts a list of lartpc_mlreco3d interaction into a list of
        OpT0Finder QCluster_t objects

        Parameters
        ----------
        interactions : List[Union[Interaction, TruthInteraction]]
            List of TPC interactions

        Returns
        -------
        List[QCluster_t]
           List of OpT0Finder flashmatch::QCluster_t objects
        '''
        # Loop over the interacions
        from flashmatch import flashmatch
        qcluster_v = []
        for ii in interactions:
            # Produce a mask to remove negative value points (can happen)
            valid_mask = np.where(ii.depositions > 0.)[0]

            # If the interaction has less than 2 points, skip
            if len(valid_mask) < 2:
                continue

            # Initialize qcluster
            qcluster = flashmatch.QCluster_t()
            qcluster.idx = int(ii.id)
            qcluster.time = 0

            # Get the point coordinates
            points = self.geo.translate(ii.points[valid_mask],
                    self.volume_id, 0)

            # Get the depositions
            if not self.use_depositions_MeV:
                depositions = ii.depositions[valid_mask]
            else:
                depositions = ii.depositions_MeV[valid_mask]

            # Fill the trajectory
            pytraj = np.hstack([points, depositions[:, None]])
            traj = flashmatch.as_geoalgo_trajectory(pytraj)
            qcluster += self.light_path.MakeQCluster(traj, self.ADC_to_MeV)

            # Append
            qcluster_v.append(qcluster)

        return qcluster_v

    def make_flash_list(self, opflashes):
        '''
        Parameters
        ----------
        opflashes : List[larcv.Flash]
            List of optical flashes


        Returns
        -------
        List[Flash_t]
            List of flashmatch::Flash_t objects
        '''
        # If requested, merge flashes that are compatible in time
        if self.reflash_merging_window is not None:
            times = [f.time() for f in opflashes]
            perm = np.argsort(times)
            new_opflashes = [opflashes[perm[0]]]
            for i in range(1, len(perm)):
                if opflashes[perm[i]].time() - opflashes[perm[i-1]].time() \
                        < self.reflash_merging_window:
                    # If compatible, simply add up the PEs
                    pe_v = np.array(opflashes[perm[i-1]].PEPerOpDet()) \
                            + np.array(opflashes[perm[i]].PEPerOpDet())
                    new_opflashes[-1].PEPerOpDet(pe_v)
                else:
                    new_opflashes.append(opflashes[perm[i]])

            opflashes = new_opflashes

        # Loop over the optical flashes
        from flashmatch import flashmatch
        flash_v = []
        for idx, f in enumerate(opflashes):
            # Initialize the Flash_t object
            flash = flashmatch.Flash_t()
            flash.idx = f.id()  # Assign a unique index
            flash.time = f.time()  # Flash timing, a candidate T0

            # Assign the flash position and error on this position
            flash.x, flash.y, flash.z = 0, 0, 0
            flash.x_err, flash.y_err, flash.z_err = 0, 0, 0

            # Assign the individual PMT optical hit PEs
            offset = 0 if len(f.PEPerOpDet()) == 180 else 180
            for i in range(180):
                flash.pe_v.push_back(f.PEPerOpDet()[i + offset])
                flash.pe_err_v.push_back(0.)

            # Append
            flash_v.append(flash)

        return flash_v, opflashes

    def run_flash_matching(self):
        '''
        Drive the OpT0Finder flash matching

        Returns
        -------
        List[flashmatch::FlashMatch_t]
            List of matches
        '''
        # Make sure the interaction and flash objects were set
        assert self.qcluster_v is not None and self.flash_v is not None, \
                'Must make_qcluster_list and make_flash_list first'

        # Register all objects in the manager
        self.mgr.Reset()
        for x in self.qcluster_v:
            self.mgr.Add(x)
        for x in self.flash_v:
            self.mgr.Add(x)

        # Run the matching
        all_matches = self.mgr.Match()

        return all_matches

    def get_qcluster(self, idx, array=False):
        '''
        Fetch a given flashmatch::QCluster_t object

        Parameters
        ----------
        idx : int
            ID of the interaction to fetch
        array : bool, default `False`
            If `True`, The QCluster is returned as an np.ndarray

        Returns
        -------
        Union[flashmatch::QCluster_t, np.ndarray]
            QCluster object
        '''
        if self.qcluster_v is None:
            raise Exception('self.qcluster_v is None')

        for qcluster in self.qcluster_v:
            if qcluster.idx != idx: continue
            if array: return flashmatch.as_ndarray(qcluster)
            else: return qcluster

        raise Exception(f'TPC object {idx} does not exist in self.qcluster_v')

    def get_flash(self, idx, array=False):
        '''
        Fetch a given flashmatch::Flash object

        Parameters
        ----------
        idx : int
            ID of the flash to fetch
        array : bool, default `False`
            If `True`, The flash is returned as an np.ndarray

        Returns
        -------
        Union[flashmatch::Flash, np.ndarray]
            Flash object
        '''
        if self.flash_v is None:
            raise Exception('self.flash_v is None')

        for flash in self.flash_v:
            if flash.idx != idx: continue
            if array: return flashmatch.as_ndarray(flash)
            else: return flash

        raise Exception('Flash {idx} does not exist in self.flash_v')


    def get_match(self, idx):
        '''
        Fetch a match for a given TPC interaction ID

        Parameters
        ----------
        idx : int
            Index of TPC object for which we want to retrieve a match

        Returns
        -------
        flashmatch::FlashMatch_t
            Flash match associated with interaction idx
        '''
        if self.matches is None:
            raise Exception('Need to run flash matching first')

        for m in self.matches:
            if self.qcluster_v[m.tpc_id].idx != idx: continue
            return m

        return None

    def get_matched_flash(self, idx):
        '''
        Fetch a matched flash for a given TPC interaction ID

        Parameters
        ----------
        idx : int
            Index of TPC object for which we want to retrieve a match

        Returns
        -------
        flashmatch::Flash_t
            Optical flash that matches interaction idx
        '''
        # Get a match, if any
        m = self.get_match(idx)
        if m is None: return None

        # Get the flash that corresponds to the match
        flash_id = m.flash_id
        if flash_id is None: return None
        if flash_id > len(self.flash_v):
            raise Exception('Flash {flash_id} does not exist in self.flash_v')

        return self.flash_v[flash_id]

    def get_t0(self, idx):
        '''
        Fetch a matched flash time for a given TPC interaction ID

        Parameters
        ----------
        idx : int
            Index of TPC object for which we want to retrieve a match

        Returns
        -------
        float
            Time in us with respect to simulation time reference
        '''
        # Get the matched flash, if any
        flash = self.get_matched_flash(idx)

        return None if flash is None else flash.time
