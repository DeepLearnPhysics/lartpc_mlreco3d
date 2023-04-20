import os, sys
import numpy as np
import time

from mlreco.utils.volumes import VolumeBoundaries

def modified_box_model(x, constant_calib):
    W_ion = 23.6 * 1e-6 # MeV/electron, work function of argon
    E = 0.5 # kV/cm, drift electric field

    beta = 0.212 # kV/cm g/cm^2 /MeV
    alpha = 0.93
    rho = 1.39295 # g.cm^-3
    return (np.exp(x/constant_calib * beta \
                   * W_ion / (rho * E)) - alpha) / (beta / (rho * E)) # MeV/cm

class FlashMatcherInterface:
    """
    Adapter class between full chain outputs and FlashManager/OpT0Finder
    """
    def __init__(self, config, fm_config, 
                 boundaries=None, opflash_keys=[], **kwargs):

        self.config = config
        self.fm_config = fm_config
        self.opflash_keys = opflash_keys

        self.reflash_merging_window = kwargs.get('reflash_merging_window', None)
        self.detector_specs = kwargs.get('detector_specs', None)
        self.ADC_to_MeV = kwargs.get('ADC_to_MeV', 1.)
        self.use_depositions_MeV = kwargs.get('use_depositions_MeV', False)
        self.boundaries = kwargs.get('boundaries', None)

        self.flash_matches = {}
        if self.boundaries is not None:
            self.vb = VolumeBoundaries(self.boundaries)
            self._num_volumes = self.vb.num_volumes()
        else:
            self.vb = None
            self._num_volumes = 1

    def initialize_flash_manager(self, meta):
        self.fm = FlashManager(self.config, self.fm_config, 
                               meta=meta,
                               reflash_merging_window=self.reflash_merging_window, 
                               detector_specs=self.detector_specs)
        
    def get_flash_matches(self, 
                          entry, 
                          interactions,
                          opflashes,
                          use_true_tpc_objects=False,
                          volume=None,
                          use_depositions_MeV=False,
                          ADC_to_MeV=1.,
                          restrict_interactions=[]):
        """
        If flash matches has not yet been computed for this volume, then it will
        be run as part of this function. Otherwise, flash matching results are
        cached in `self.flash_matches` per volume.

        If `restrict_interactions` is specified, no caching is done.

        Parameters
        ==========
        entry: int
        use_true_tpc_objects: bool, default is False
            Whether to use true or predicted interactions.
        volume: int, default is None
        use_depositions_MeV: bool, default is False
            If using true interactions, whether to use true MeV depositions or reconstructed charge.
        ADC_to_MEV: double, default is 1.
            If using reconstructed interactions, this defines the conversion in OpT0Finder.
            OpT0Finder computes the hypothesis flash using light yield and deposited charge in MeV.
        restrict_interactions: list, default is []
           If specified, the interactions to match will be whittle down to this subset of interactions.
           Provide list of interaction ids.

        Returns
        =======
        list of tuple (Interaction, larcv::Flash, flashmatch::FlashMatch_t)
        """
        # No caching done if matching a subset of interactions
        if (entry, volume, use_true_tpc_objects) not in self.flash_matches or len(restrict_interactions):
            out = self._run_flash_matching(entry, 
                                           interactions,
                                           opflashes,
                                           use_true_tpc_objects=use_true_tpc_objects, 
                                           volume=volume,
                                           use_depositions_MeV=use_depositions_MeV, 
                                           ADC_to_MeV=ADC_to_MeV, 
                                           restrict_interactions=restrict_interactions)

        if len(restrict_interactions) == 0:
            tpc_v, pmt_v, matches = self.flash_matches[(entry, volume, use_true_tpc_objects)]
        else: # it wasn't cached, we just computed it
            tpc_v, pmt_v, matches = out
        return [(tpc_v[m.tpc_id], pmt_v[m.flash_id], m) for m in matches]
    

    def _run_flash_matching(self, entry, interactions,
            opflashes, 
            use_true_tpc_objects=False,
            volume=None,
            use_depositions_MeV=False,
            ADC_to_MeV=1.,
            restrict_interactions=[]):
        """
        Parameters
        ==========
        entry: int
        use_true_tpc_objects: bool, default is False
            Whether to use true or predicted interactions.
        volume: int, default is None
        """
        if use_true_tpc_objects:
            if not hasattr(self, 'get_true_interactions'):
                raise Exception('This Predictor does not know about truth info.')

            tpc_v = [ia for ia in interactions if volume is None or ia.volume == volume]
        else:
            tpc_v = [ia for ia in interactions if volume is None or ia.volume == volume]

        if len(restrict_interactions) > 0: # by default, use all interactions
            tpc_v_select = []
            for interaction in tpc_v:
                if interaction.id in restrict_interactions:
                    tpc_v_select.append(interaction)
            tpc_v = tpc_v_select

        # If we are not running flash matching over the entire volume at once,
        # then we need to shift the coordinates that will be used for flash matching
        # back to the reference of the first volume.
        if volume is not None:
            for tpc_object in tpc_v:
                tpc_object.points = self._untranslate(tpc_object.points, volume)
        input_tpc_v = self.fm.make_qcluster(tpc_v, use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV)
        if volume is not None:
            for tpc_object in tpc_v:
                tpc_object.points = self._translate(tpc_object.points, volume)

        # Now making Flash_t objects
        selected_opflash_keys = self.opflash_keys
        if volume is not None:
            assert isinstance(volume, int)
            selected_opflash_keys = [self.opflash_keys[volume]]
        pmt_v = []
        for key in selected_opflash_keys:
            pmt_v.extend(opflashes[key][entry])
        input_pmt_v = self.fm.make_flash([opflashes[key][entry] for key in selected_opflash_keys])

        # input_pmt_v might be a filtered version of pmt_v,
        # and we want to store larcv::Flash objects not
        # flashmatch::Flash_t objects in self.flash_matches
        from larcv import larcv
        new_pmt_v = []
        for flash in input_pmt_v:
            new_flash = larcv.Flash()
            new_flash.time(flash.time)
            new_flash.absTime(flash.time_true) # Hijacking this field
            new_flash.timeWidth(flash.time_width)
            new_flash.xCenter(flash.x)
            new_flash.yCenter(flash.y)
            new_flash.zCenter(flash.z)
            new_flash.xWidth(flash.x_err)
            new_flash.yWidth(flash.y_err)
            new_flash.zWidth(flash.z_err)
            new_flash.PEPerOpDet(flash.pe_v)
            new_flash.id(flash.idx)
            new_pmt_v.append(new_flash)

        # Running flash matching and caching the results
        start = time.time()
        matches = self.fm.run_flash_matching()
        print('Actual flash matching took %d s' % (time.time() - start))
        if len(restrict_interactions) == 0:
            self.flash_matches[(entry, volume, use_true_tpc_objects)] = (tpc_v, new_pmt_v, matches)
        return tpc_v, new_pmt_v, matches
    
    def _translate(self, voxels, volume):
        """
        Go from 1-volume-only back to full volume coordinates

        Parameters
        ==========
        voxels: np.ndarray
            Shape (N, 3)
        volume: int

        Returns
        =======
        np.ndarray
            Shape (N, 3)
        """
        if self.vb is None or volume is None:
            return voxels
        else:
            return self.vb.translate(voxels, volume)

    def _untranslate(self, voxels, volume):
        """
        Go from full volume to 1-volume-only coordinates

        Parameters
        ==========
        voxels: np.ndarray
            Shape (N, 3)
        volume: int

        Returns
        =======
        np.ndarray
            Shape (N, 3)
        """
        if self.vb is None or volume is None:
            return voxels
        else:
            return self.vb.untranslate(voxels, volume)
        
    

class FlashManager:
    """
    Meant as an interface to OpT0finder, likelihood-based flash matching.

    See https://github.com/drinkingkazu/OpT0Finder for more details about it.
    """
    def __init__(self, cfg, cfg_fmatch, 
                 meta=None, 
                 detector_specs=None, 
                 reflash_merging_window=None):
        """
        Expects that the environment variable `FMATCH_BASEDIR` is set.
        You can either set it by hand (to the path where one can find
        OpT0Finder) or you can source `OpT0Finder/configure.sh` if you
        are running code from a command line.

        Parameters
        ==========
        cfg: dict
            The full chain config.
        cfg_fmatch: str
            Path to config for OpT0Finder.
        meta: np.ndarray, optional, default is None
            Used to shift coordinates of interactions to "real" detector
            coordinates for QCluster_t.
        detector_specs: str, optional
            Path to `detector_specs.cfg` file which defines some geometry
            information about the detector PMT system. By default will look
            into `OpT0Finder/dat/detector_specs.cfg`.
        """

        # Setup OpT0finder
        basedir = os.getenv('FMATCH_BASEDIR')
        if basedir is None:
            msg = "You need to source OpT0Finder configure.sh "\
                "first, or set the FMATCH_BASEDIR environment variable."
            raise Exception(msg)

        sys.path.append(os.path.join(basedir, 'python'))
        os.environ['LD_LIBRARY_PATH'] = "%s:%s" % (os.path.join(basedir, 'build/lib'), os.environ['LD_LIBRARY_PATH'])
        #os.environ['ROOT_INCLUDE_PATH'] = os.path.join(basedir, 'build/include')
        if 'FMATCH_DATADIR' not in os.environ: # needed for loading detector specs
            os.environ['FMATCH_DATADIR'] = os.path.join(basedir, 'dat')
        import ROOT

        import flashmatch
        from flashmatch import flashmatch

        # Setup meta
        self.cfg = cfg

        self.min_x, self.min_y, self.min_z = None, None, None
        self.size_voxel_x, self.size_voxel_y, self.size_voxel_z = None, None, None
        # print(f"META = {meta}")
        if meta is not None:
            self.min_x = meta[0]
            self.min_y = meta[1]
            self.min_z = meta[2]
            self.size_voxel_x = meta[6]
            self.size_voxel_y = meta[7]
            self.size_voxel_z = meta[8]

        # Setup flash matching
        print('Setting up OpT0Finder for flash matching...')
        self.mgr = flashmatch.FlashMatchManager()
        cfg = flashmatch.CreatePSetFromFile(cfg_fmatch)
        if detector_specs is None:
            self.det = flashmatch.DetectorSpecs.GetME(
                os.path.join(basedir, 'dat/detector_specs.cfg'))
        else:
            assert isinstance(detector_specs, str)
            if not os.path.exists(detector_specs):
                raise Exception("Detector specs file not found")

            self.det = flashmatch.DetectorSpecs.GetME(detector_specs)
        self.mgr.Configure(cfg)
        print('...done.')

        self.all_matches = None
        self.pmt_v, self.tpc_v = None, None

        self.reflash_merging_window = reflash_merging_window

    def get_flash(self, flash_id, array=False):
        from flashmatch import flashmatch

        if self.pmt_v is None:
            raise Exception("self.pmt_v is None")

        for flash in self.pmt_v:
            if flash.idx != flash_id: continue
            if array: return flashmatch.as_ndarray(flash)
            else: return flash

        raise Exception("Flash %d does not exist in self.pmt_v" % flash_id)

    def get_qcluster(self, tpc_id, array=False):
        from flashmatch import flashmatch

        if self.tpc_v is None:
            raise Exception("self.tpc_v is None")

        for tpc in self.tpc_v:
            if tpc.idx != tpc_id: continue
            if array: return flashmatch.as_ndarray(tpc)
            else: return tpc

        raise Exception("TPC object %d does not exist in self.tpc_v" % tpc_id)

    def make_qcluster(self, interactions, 
                      use_depositions_MeV=False, ADC_to_MeV=1.):
        """
        Make flashmatch::QCluster_t objects from list of interactions.

        Note that coordinates of `interactions` are in voxel coordinates,
        but inside this function we shift back to real detector coordinates
        using meta information. flashmatch::QCluster_t objects are in
        real cm coordinates.

        Parameters
        ==========
        interactions: list of Interaction/TruthInteraction
            (Predicted or true) interaction objects.

        Returns
        =======
        list of flashmatch::QCluster_t
        """
        from flashmatch import flashmatch

        if self.min_x is None:
            raise Exception('min_x is None')

        tpc_v = []
        for p in interactions:
            qcluster = flashmatch.QCluster_t()
            qcluster.idx = int(p.id) # Assign a unique index
            qcluster.time = 0  # assumed time w.r.t. trigger for reconstruction
            for i in range(p.size):
                # Create a geoalgo::QPoint_t
                qpoint = flashmatch.QPoint_t(
                    p.points[i, 0] * self.size_voxel_x + self.min_x,
                    p.points[i, 1] * self.size_voxel_y + self.min_y,
                    p.points[i, 2] * self.size_voxel_z + self.min_z,
                    p.depositions[i]*ADC_to_MeV*self.det.LightYield() if not use_depositions_MeV else p.depositions_MeV[i]*self.det.LightYield())
                # Add it to geoalgo::QCluster_t
                qcluster.push_back(qpoint)
            tpc_v.append(qcluster)

        self.tpc_v = tpc_v
        print('Made list of %d QCluster_t' % len(tpc_v))
        return tpc_v

    def make_flash(self, larcv_flashes):
        """
        Parameters
        ==========
        larcv_flashes: list of list of larcv::Flash

        Returns
        =======
        list of flashmatch::Flash_t
        """
        from flashmatch import flashmatch

        flashes = []
        for branch in larcv_flashes:
            flashes.extend(branch)

        pmt_v, times = [], []
        for idx, f in enumerate(flashes):
            # f is an object of type larcv::Flash
            flash = flashmatch.Flash_t()
            flash.idx = f.id()  # Assign a unique index
            flash.time = f.time()  # Flash timing, a candidate T0
            flash.time_true = f.absTime() # Hijacking this field to store absolute time
            times.append(flash.time)

            # Assign the flash position and error on this position
            flash.x, flash.y, flash.z = 0, 0, 0
            flash.x_err, flash.y_err, flash.z_err = 0, 0, 0

            # PE distribution over the 360 photodetectors
            offset = 0 if len(f.PEPerOpDet()) == 180 else 180
            for i in range(180):
                flash.pe_v.push_back(f.PEPerOpDet()[i + offset])
                flash.pe_err_v.push_back(0.)
            pmt_v.append(flash)
        if self.reflash_merging_window is not None and len(pmt_v) > 0:
            # then proceed to merging close flashes
            perm = np.argsort(times)
            pmt_v = np.array(pmt_v)[perm]
            final_pmt_v = [pmt_v[0]]
            for idx, flash in enumerate(pmt_v[1:]):
                if flash.time - final_pmt_v[-1].time < self.reflash_merging_window:
                    new_flash = self.merge_flashes(flash, final_pmt_v[-1])
                    final_pmt_v[-1] = new_flash
                else:
                    final_pmt_v.append(flash)
            print("Merged", len(final_pmt_v), len(pmt_v))
            pmt_v = final_pmt_v

        self.pmt_v = pmt_v
        return pmt_v

    def merge_flashes(self, a, b):
        """
        Util to merge 2 flashmatch::Flash_t objects on the fly.

        Final time is minimum of both times. Final PE count per
        photodetectors is the sum between the 2 flashes.

        Parameters
        ==========
        a: flashmatch::Flash_t
        b: flashmatch::Flash_t

        Returns
        =======
        flashmatch::Flash_t
        """
        from flashmatch import flashmatch
        flash = flashmatch.Flash_t()
        flash.idx = min(a.idx, b.idx)
        flash.time = min(a.time, b.time)
        flash.time_true = min(a.time_true, b.time_true)
        flash.x, flash.y, flash.z = min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)
        flash.x_err = min(a.x_err, b.x_err)
        flash.y_err = min(a.y_err, b.y_err)
        flash.z_err = min(a.z_err, b.z_err)
        for i in range(180):
            flash.pe_v.push_back(a.pe_v[i] + b.pe_v[i])
            flash.pe_err_v.push_back(a.pe_err_v[i] + b.pe_err_v[i])
        return flash

    def run_flash_matching(self, flashes=None, interactions=None, **kwargs):
        if self.tpc_v is None:
            if interactions is None:
                msg = "You need to specify `interactions`, "\
                    "or to run make_qcluster."
                raise Exception(msg)
        if interactions is not None:
            self.make_qcluster(interactions, **kwargs)


        if self.pmt_v is None:
            if flashes is None:
                msg = "PMT objects need to be defined. "\
                    "Either specify `flashes`, or run make_flash."
                raise Exception(msg)
        if flashes is not None:
            self.make_flash(flashes)

        assert self.tpc_v is not None and self.pmt_v is not None 

        self.mgr.Reset()

        # First register all objects in manager
        for x in self.tpc_v:
            self.mgr.Add(x)
        for x in self.pmt_v:
            self.mgr.Add(x)

        # Run the matching
        self.all_matches = self.mgr.Match()
        return self.all_matches

    def get_match(self, idx, matches=None):
        """
        Parameters
        ==========
        idx: int
            Index of TPC object for which we want to retrieve a match.
        matches: list of flashmatch::FlashMatch_t, optional, default is None

        Returns
        =======
        flashmatch::FlashMatch_t
        """
        if matches is None:
            if self.all_matches is None:
                raise Exception("Need to run flash matching first with run_flash_matching.")
            matches = self.all_matches

        for m in self.all_matches:
            if self.tpc_v[m.tpc_id].idx != idx: continue
            return m

        return None

    def get_matched_flash(self, idx, matches=None):
        """
        Parameters
        ==========
        idx: int
            Index of TPC object for which we want to retrieve a match.
        matches: list of flashmatch::FlashMatch_t, optional, default is None

        Returns
        =======
        flashmatch::Flash_t
        """
        m = self.get_match(idx, matches=matches)
        if m is None: return None

        flash_id = m.flash_id
        if flash_id is None: return None

        if flash_id > len(self.pmt_v):
            raise Exception("Could not find flash id %d in self.pmt_v" % flash_id)

        return self.pmt_v[flash_id]


    def get_t0(self, idx, matches=None):
        """
        Parameters
        ==========
        idx: int
            Index of TPC object for which we want to retrieve a match.
        matches: list of flashmatch::FlashMatch_t, optional, default is None

        Returns
        =======
        float
            Time in us with respect to simulation time reference.
        """
        flash = self.get_matched_flash(idx, matches=matches)
        return None if flash is None else flash.time
