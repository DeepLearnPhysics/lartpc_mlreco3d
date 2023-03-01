from typing import Callable, Tuple, List
import numpy as np
import pandas as pd
import os, sys
import time

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.metrics import unique_label
from collections import defaultdict

from scipy.special import softmax
from analysis.classes import Particle, ParticleFragment, TruthParticleFragment, \
        TruthParticle, Interaction, TruthInteraction, FlashManager
from analysis.classes.particle import matrix_counts, matrix_iou, \
        match_particles_fn, match_interactions_fn, group_particles_to_interactions_fn
from analysis.algorithms.point_matching import *

from mlreco.utils.groups import type_labels as TYPE_LABELS
from mlreco.utils.vertex import get_vertex, predict_vertex
from mlreco.utils.deghosting import deghost_labels_and_predictions, compute_rescaled_charge

from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.iotools.collates import VolumeBoundaries


class FullChainPredictor:
    '''
    User Interface for full chain inference.

    Usage Example:

        model = Trainer._net.module
        entry = 0   # batch id
        predictor = FullChainPredictor(model, data_blob, res, cfg)
        pred_seg = predictor._fit_predict_semantics(entry)

    Instructions
    -----------------------------------------------------------------------

    1) To avoid confusion between different quantities, the label namings under
    iotools.schema must be set as follows:

        schema:
            input_data:
                - parse_sparse3d_scn
                - sparse3d_pcluster

    2) By default, unwrapper must be turned ON under trainval:

        trainval:
            unwrapper: unwrap_3d_mink

    3) Some outputs needs to be listed under trainval.concat_result.
    The predictor will run through a checklist to ensure this condition

    4) Does not support deghosting at the moment. (TODO)
    '''
    def __init__(self, data_blob, result, cfg, predictor_cfg={}, deghosting=False,
            enable_flash_matching=False, flash_matching_cfg="", opflash_keys=[]):
        self.module_config = cfg['model']['modules']
        self.cfg = cfg

        # Handle deghosting before anything and save deghosting specific
        # quantities separately from data_blob and result

        self.deghosting = self.module_config['chain']['enable_ghost']
        self.data_blob = data_blob
        self.result = result

        if self.deghosting:
            deghost_labels_and_predictions(self.data_blob, self.result)

        self.num_images = len(data_blob['input_data'])
        self.index = self.data_blob['index']

        # self.spatial_size             = predictor_cfg.get('spatial_size', 768)
        # For matching particles and interactions
        self.min_overlap_count        = predictor_cfg.get('min_overlap_count', 0)
        # Idem, can be 'count' or 'iou'
        self.overlap_mode             = predictor_cfg.get('overlap_mode', 'iou')
        # Minimum voxel count for a true non-ghost particle to be considered
        self.min_particle_voxel_count = predictor_cfg.get('min_particle_voxel_count', 20)
        # We want to count how well we identify interactions with some PDGs
        # as primary particles
        self.primary_pdgs             = np.unique(predictor_cfg.get('primary_pdgs', []))
        # Following 2 parameters are vertex heuristic parameters
        self.attaching_threshold      = predictor_cfg.get('attaching_threshold', 2)
        self.inter_threshold          = predictor_cfg.get('inter_threshold', 10)

        self.batch_mask = self.data_blob['input_data']

        # This is used to apply fiducial volume cuts.
        # Min/max boundaries in each dimension haev to be specified.
        self.volume_boundaries = predictor_cfg.get('volume_boundaries', None)
        if self.volume_boundaries is None:
            # Using ICARUS Cryo 0 as a default
            pass
        else:
            self.volume_boundaries = np.array(self.volume_boundaries, dtype=np.float64)
            if 'meta' not in self.data_blob:
                raise Exception("Cannot use volume boundaries because meta is missing from iotools config.")
            else: # convert to voxel units
                meta = self.data_blob['meta'][0]
                min_x, min_y, min_z = meta[0:3]
                size_voxel_x, size_voxel_y, size_voxel_z = meta[6:9]

                self.volume_boundaries[0, :] = (self.volume_boundaries[0, :] - min_x) / size_voxel_x
                self.volume_boundaries[1, :] = (self.volume_boundaries[1, :] - min_y) / size_voxel_y
                self.volume_boundaries[2, :] = (self.volume_boundaries[2, :] - min_z) / size_voxel_z

        # Determine whether we need to account for several distinct volumes
        # split over "virtual" batch ids
        # Note this is different from "self.volume_boundaries" above
        # FIXME rename one or the other to be clearer
        boundaries = cfg['iotool'].get('collate', {}).get('boundaries', None)
        if boundaries is not None:
            self.vb = VolumeBoundaries(boundaries)
            self._num_volumes = self.vb.num_volumes()
        else:
            self.vb = None
            self._num_volumes = 1

        # Prepare flash matching if requested
        self.enable_flash_matching = enable_flash_matching
        self.fm = None
        if enable_flash_matching:
            reflash_merging_window = predictor_cfg.get('reflash_merging_window', None)

            if 'meta' not in self.data_blob:
                raise Exception('Meta unspecified in data_blob. Please add it to your I/O schema.')
            #if 'FMATCH_BASEDIR' not in os.environ:
            #    raise Exception('FMATCH_BASEDIR undefined. Please source `OpT0Finder/configure.sh` or define it manually.')
            assert os.path.exists(flash_matching_cfg)
            assert len(opflash_keys) == self._num_volumes

            self.fm = FlashManager(cfg, flash_matching_cfg, meta=self.data_blob['meta'][0], reflash_merging_window=reflash_merging_window)
            self.opflash_keys = opflash_keys

            self.flash_matches = {} # key is (entry, volume, use_true_tpc_objects), value is tuple (tpc_v, pmt_v, list of matches)
            # type is (list of Interaction/TruthInteraction, list of larcv::Flash, list of flashmatch::FlashMatch_t)

            self.crt_tpc_matches = {} # key is (entry, volume, use_true_tpc_objects), value is tuple (tpc_v, crt_v, list of matches)
            # type is (list of Interaction/TruthInteraction, list of larcv::CRTHit, list of matcha::Match)


    def __repr__(self):
        msg = "FullChainEvaluator(num_images={})".format(int(self.num_images/self._num_volumes))
        return msg

    def get_flash_matches(self, entry,
            use_true_tpc_objects=False,
            volume=None,
            use_depositions_MeV=False,
            ADC_to_MeV=1.,
            interaction_list=[]):
        """
        If flash matches has not yet been computed for this volume, then it will
        be run as part of this function. Otherwise, flash matching results are
        cached in `self.flash_matches` per volume.

        If `interaction_list` is specified, no caching is done.

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
        interaction_list: list, default is []
           If specified, the interactions to match will be whittle down to this subset of interactions.
           Provide list of interaction ids.

        Returns
        =======
        list of tuple (Interaction, larcv::Flash, flashmatch::FlashMatch_t)
        """

        # No caching done if matching a subset of interactions
        print('Interaction list:\n')
        print(interaction_list)
        if (entry, volume, use_true_tpc_objects) not in self.flash_matches or len(interaction_list):
            print('No caching done')
            out = self._run_flash_matching(entry, use_true_tpc_objects=use_true_tpc_objects, volume=volume,
                    use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV, interaction_list=interaction_list)

        if len(interaction_list) == 0:
            print('len(interaction_list) is 0')
            tpc_v, pmt_v, matches = self.flash_matches[(entry, volume, use_true_tpc_objects)]
        else: # it wasn't cached, we just computed it
            print('Else')
            tpc_v, pmt_v, matches = out
        print('[UI:GetFlashMatches] len(tpc_v):', len(tpc_v))
        print('[UI:GetFlashMatches] len(match):', len(matches))
        print('mathces:\n', *matches)
        return [(tpc_v[m.tpc_id], pmt_v[m.flash_id], m) for m in matches]

    def _run_flash_matching(self, entry,
            use_true_tpc_objects=False,
            volume=None,
            use_depositions_MeV=False,
            ADC_to_MeV=1.,
            interaction_list=[]):
        """
        Parameters
        ==========
        entry: int
        use_true_tpc_objects: bool, default is False
            Whether to use true or predicted interactions.
        volume: int, default is None

        Returns
        =======
        tpc_v: list of interactions
        new_pmt_v: list of filtered larcv::Flash objects
        matches: list of OpT0Finder::flashmatch::FlashMatch_t
        """
        print('*** Running Flash Matching ***')
        if use_true_tpc_objects:
            if not hasattr(self, 'get_true_interactions'):
                raise Exception('This Predictor does not know about truth info.')

            tpc_v = self.get_true_interactions(entry, drop_nonprimary_particles=False, volume=volume, compute_vertex=False)
        else:
            tpc_v = self.get_interactions(entry, drop_nonprimary_particles=False, volume=volume, compute_vertex=False)

        print('len(interaction_list):', len(interaction_list))
        if len(interaction_list) > 0: # by default, use all interactions
            tpc_v_select = []
            for interaction in tpc_v:
                if interaction.id in interaction_list:
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
            print('flash key', key)
            pmt_v.extend(self.data_blob[key][entry])
        print('self.data_blob[key][entry]:', [self.data_blob[key][entry] for key in selected_opflash_keys])
        input_pmt_v = self.fm.make_flash([self.data_blob[key][entry] for key in selected_opflash_keys])

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
        print('[UI] Starting self.fm.run_flash_matching()...')
        matches = self.fm.run_flash_matching()
        print('Actual flash matching took %d s' % (time.time() - start))
        
        if len(interaction_list) == 0:
            self.flash_matches[(entry, volume, use_true_tpc_objects)] = (tpc_v, new_pmt_v, matches)
        return tpc_v, new_pmt_v, matches

    def get_crt_tpc_matches(self, entry,
            use_true_tpc_objects=False,
            volume=None,
            use_depositions_MeV=False,
            ADC_to_MeV=1.,
            interaction_list=[]):

        # No caching done if matching a subset of interactions
        if (entry, volume, use_true_tpc_objects) not in self.crt_tpc_matches or len(interaction_list):
            print('[CRTTPC] No caching done')
            out = self._run_crt_tpc_matching(entry, use_true_tpc_objects=use_true_tpc_objects, volume=volume,
                    use_depositions_MeV=use_depositions_MeV, ADC_to_MeV=ADC_to_MeV, interaction_list=interaction_list)

        if len(interaction_list) == 0:
            print('[CRTTPC] len(interaction_list) is 0')
            tpc_v, crt_v, matches = self.crt_tpc_matches[(entry, volume, use_true_tpc_objects)]
        else: # it wasn't cached, we just computed it
            print('[CRTTPC] Not chached')
            tpc_v, crt_v, matches = out
        return [(tpc_v[m.tpc_id], crt_v[m.flash_id], m) for m in matches]

    def _run_crt_tpc_matching(self, entry,
            use_true_tpc_objects=False,
            volume=None,
            use_depositions_MeV=False,
            ADC_to_MeV=1.,
            interaction_list=[]):
        
        # TODO How much of this can we skip, assuming interaction_list is calculated in
        # _run_flash_matching()?
        print('*** Running CRT-TPC Matching ***')
        if use_true_tpc_objects:
            if not hasattr(self, 'get_true_interactions'):
                raise Exception('This Predictor does not know about truth info.')

            tpc_v = self.get_true_interactions(entry, drop_nonprimary_particles=False, volume=volume, compute_vertex=False)
        else:
            tpc_v = self.get_interactions(entry, drop_nonprimary_particles=False, volume=volume, compute_vertex=False)

        if len(interaction_list) > 0: # by default, use all interactions
            tpc_v_select = []
            for interaction in tpc_v:
                if interaction.id in interaction_list:
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

        # From interactions, get non-contained muon candidates for CRT-TPC matching
        # Candidate selection criteria:
        #   - pid >= 2: avoids electrons and photons, but includes pions and protons since mu/pi separation isn't perfect
        #   - Uncontained track: cosmic tracks shouldn't be contained
        muon_candidates = [particle for interaction in interaction_list
                           for particle in interaction.particles 
                           if particle.pid >= 2 and not self.is_contained(particle.points)
        ]

        trk_v = self.fm.make_tpctrack(muon_candidates)
        crt_v = self.fm.make_crthit(self.data_blob['crthits'][entry])

        # TODO Placeholder
        matches = []

        return trk_v, crt_v, matches

    def _fit_predict_ppn(self, entry):
        '''
        Method for predicting ppn predictions.

        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - df (pd.DataFrame): pandas dataframe of ppn points, with
            x, y, z, coordinates, Score, Type, and sample index.
        '''
        # Deghosting is already applied during initialization
        ppn = uresnet_ppn_type_point_selector(self.data_blob['input_data'][entry],
                                              self.result,
                                              entry=entry, apply_deghosting=not self.deghosting)
        ppn_voxels = ppn[:, 1:4]
        ppn_score = ppn[:, 5]
        ppn_type = ppn[:, 12]

        ppn_candidates = []
        for i, pred_point in enumerate(ppn_voxels):
            pred_point_type, pred_point_score = ppn_type[i], ppn_score[i]
            x, y, z = ppn_voxels[i][0], ppn_voxels[i][1], ppn_voxels[i][2]
            ppn_candidates.append(np.array([x, y, z, pred_point_score, pred_point_type]))

        if len(ppn_candidates):
            ppn_candidates = np.vstack(ppn_candidates)
        else:
            enable_classify_endpoints = 'classify_endpoints' in self.result
            ppn_candidates = np.empty((0, 13 if not enable_classify_endpoints else 15), dtype=np.float32)
        return ppn_candidates


    def _fit_predict_semantics(self, entry):
        '''
        Method for predicting semantic segmentation labels.

        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted segmentation labels.
        '''
        segmentation = self.result['segmentation'][entry]
        out = np.argmax(segmentation, axis=1).astype(int)
        return out


    def _fit_predict_gspice_fragments(self, entry):
        '''
        Method for predicting fragment labels (dense clustering)
        using graph spice.

        Inputs:

            - entry: Batch number to retrieve example.

        Returns:

            - pred: 1D numpy integer array of predicted fragment labels.
            The labels only range over classes which were designated to be
            processed in GraphSPICE.

            - G: networkx graph representing the current entry

            - subgraph: same graph in torch_geometric.Data format.
        '''
        import warnings
        warnings.filterwarnings('ignore')

        graph = self.result['graph'][0]
        graph_info = self.result['graph_info'][0]
        index_mapping = { key : val for key, val in zip(
           range(0, len(graph_info.Index.unique())), self.index)}

        min_points = self.module_config['graph_spice'].get('min_points', 1)
        invert = self.module_config['graph_spice_loss'].get('invert', True)

        graph_info['Index'] = graph_info['Index'].map(index_mapping)
        constructor_cfg = self.cluster_graph_constructor.constructor_cfg
        gs_manager = ClusterGraphConstructor(constructor_cfg,
                                             graph_batch=graph,
                                             graph_info=graph_info,
                                             batch_col=0,
                                             training=False)
        pred, G, subgraph = gs_manager.fit_predict_one(entry,
                                                       invert=invert,
                                                       min_points=min_points)

        return pred, G, subgraph

    @staticmethod
    def randomize_labels(labels):
        '''
        Simple method to randomize label order (useful for plotting)
        '''
        labels, _ = unique_label(labels)

        N = np.unique(labels).shape[0]
        perm = np.random.permutation(N)

        new_labels = -np.ones(labels.shape[0]).astype(int)
        for i, c in enumerate(perm):
            mask = labels == i
            new_labels[mask] = c
        return new_labels


    def is_contained(self, points, threshold=30):
        """
        Parameters
        ----------
        points: np.ndarray
            Shape (N, 3). Coordinates in voxel units.
        threshold: float or np.ndarray
            Distance (in voxels) from boundaries beyond which
            an object is contained. Can be an array if different
            threshold must be applied in x, y and z (shape (3,)).

        Returns
        -------
        bool
        """
        if not isinstance(threshold, np.ndarray):
            threshold = threshold * np.ones((3,))
        else:
            assert threshold.shape[0] == 3
            assert len(threshold.shape) == 1

        if self.volume_boundaries is None:
            raise Exception("Please define volume boundaries before using containment method.")

        x_contained = (self.volume_boundaries[0, 0] + threshold[0] <= points[:, 0]) & (points[:, 0] <= self.volume_boundaries[0, 1] - threshold[0])
        y_contained = (self.volume_boundaries[1, 0] + threshold[1] <= points[:, 1]) & (points[:, 1] <= self.volume_boundaries[1, 1] - threshold[1])
        z_contained = (self.volume_boundaries[2, 0] + threshold[2] <= points[:, 2]) & (points[:, 2] <= self.volume_boundaries[2, 1] - threshold[2])

        return (x_contained & y_contained & z_contained).all()


    def _fit_predict_fragments(self, entry):
        '''
        Method for obtaining voxel-level fragment labels for full image,
        including labels from both GraphSPICE and DBSCAN.

        "Voxel-level" means that the label tensor has the same length
        as the full point cloud of the current image (specified by entry #)

        If a voxel is not assigned to any fragment (ex. low E depositions),
        we assign -1 as its fragment label.


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - new_labels: 1D numpy integer array of predicted fragment labels.
        '''
        fragments = self.result['fragments'][entry]

        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_frag_labels = -np.ones(num_voxels).astype(int)

        for i, mask in enumerate(fragments):
            pred_frag_labels[mask] = i

        new_labels = pred_frag_labels

        return new_labels


    def _fit_predict_groups(self, entry):
        '''
        Method for obtaining voxel-level group labels.

        If a voxel does not belong to any particle (ex. low E depositions),
        we assign -1 as its group (particle) label.


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted group labels.
        '''
        particles = self.result['particles'][entry]
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_group_labels = -np.ones(num_voxels).astype(int)

        for i, mask in enumerate(particles):
            pred_group_labels[mask] = i

        new_labels = pred_group_labels

        return new_labels


    def _fit_predict_interaction_labels(self, entry):
        '''
        Method for obtaining voxel-level interaction labels for full image.

        If a voxel does not belong to any interaction (ex. low E depositions),
        we assign -1 as its interaction (particle) label.


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - new_labels: 1D numpy integer array of predicted interaction labels.
        '''
        inter_group_pred = self.result['inter_group_pred'][entry]
        particles = self.result['particles'][entry]
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_inter_labels = -np.ones(num_voxels).astype(int)

        for i, mask in enumerate(particles):
            pred_inter_labels[mask] = inter_group_pred[i]

        new_labels = pred_inter_labels

        return new_labels


    def _fit_predict_pids(self, entry):
        '''
        Method for obtaining voxel-level particle type
        (photon, electron, muon, ...) labels for full image.

        If a voxel does not belong to any particle (ex. low E depositions),
        we assign -1 as its particle type label.


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted particle type labels.
        '''
        particles = self.result['particles'][entry]
        type_logits = self.result['node_pred_type'][entry]
        pids = np.argmax(type_logits, axis=1)
        num_voxels = self.data_blob['input_data'][entry].shape[0]

        pred_pids = -np.ones(num_voxels).astype(int)

        for i, mask in enumerate(particles):
            pred_pids[mask] = pids[i]

        return pred_pids


    def _fit_predict_vertex_info(self, entry, inter_idx):
        '''
        Method for obtaining interaction vertex information given
        entry number and interaction ID number.

        Inputs:
            - entry: Batch number to retrieve example.

            - inter_idx: Interaction ID number.

        If the interaction specified by <inter_idx> does not exist
        in the sample numbered by <entry>, function will raise a
        ValueError.

        Returns:
            - vertex_info: (x,y,z) coordinate of predicted vertex
        '''
        vertex_info = predict_vertex(inter_idx, entry,
                                     self.data_blob['input_data'],
                                     self.result,
                                     attaching_threshold=self.attaching_threshold,
                                     inter_threshold=self.inter_threshold,
                                     apply_deghosting=False)

        return vertex_info

    def _get_entries(self, entry, volume):
        """
        Make a list of actual entries in the batch ids. This accounts for potential
        virtual batch ids in case we used volume boundaries to process several volumes
        separately.

        Parameters
        ==========
        entry: int
            Which entry of the original dataset you want to access.
        volume: int or None
            Which volume you want to access. None means all of them.

        Returns
        =======
        list
            List of integers = actual batch ids in the tensors (potentially virtual batch ids).
        """
        entries = [entry] # default behavior
        if self.vb is not None: # in case we defined virtual batch ids (volume boundaries)
            entries = self.vb.virtual_batch_ids(entry) # these are ALL the virtual batch ids corresponding to this entry
            if volume is not None: # maybe we wanted to select a specific volume
                entries = [entries[volume]]
        return entries

    def _check_volume(self, volume):
        """
        Basic sanity check that the volume given makes sense given the config.

        Parameters
        ==========
        volume: int or None

        Returns
        =======
        Nothing
        """
        if volume is not None and self.vb is None:
            raise Exception("You need to specify volume boundaries in your I/O config (collate section).")
        if volume is not None:
            assert isinstance(volume, (int, np.int64, np.int32)) and volume >= 0

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

    def get_fragments(self, entry, only_primaries=False,
                      min_particle_voxel_count=-1,
                      attaching_threshold=2,
                      semantic_type=None, verbose=False,
                      true_id=False, volume=None) -> List[Particle]:
        '''
        Method for retriving fragment list for given batch index.

        The output fragments will have its ppn candidates attached as
        attributes in the form of pandas dataframes (same as _fit_predict_ppn)

        Method also performs startpoint prediction for shower fragments.

        Parameters
        ==========
        entry: int
            Batch number to retrieve example.
        only_primaries: bool, default False
        min_particle_voxel_count: int, default -1
        attaching_threshold: float, default 2
            threshold distance to attach ppn point to particle.
        semantic_type: int, default None
            If True, only ppn candiates with the
            same predicted semantic type will be matched to its corresponding
            particle.
        verbose: bool, default False
        true_id: bool, default False
        volume: int, default None

        Returns
        =======
        list
            List of <Particle> instances (see Particle class definition).
        '''
        self._check_volume(volume)

        if min_particle_voxel_count < 0:
            min_particle_voxel_count = self.min_particle_voxel_count

        entries = self._get_entries(entry, volume)

        out_fragment_list = []
        for entry in entries:
            volume = entry % self._num_volumes

            point_cloud = self.data_blob['input_data'][entry][:, 1:4]
            depositions = self.result['input_rescaled'][entry][:, 4]
            fragments = self.result['fragments'][entry]
            fragments_seg = self.result['fragments_seg'][entry]

            shower_mask = np.isin(fragments_seg, self.module_config['grappa_shower']['base']['node_type'])
            shower_frag_primary = np.argmax(self.result['shower_node_pred'][entry], axis=1)

            if 'shower_node_features' in self.result:
                shower_node_features = self.result['shower_node_features'][entry]
            if 'track_node_features' in self.result:
                track_node_features = self.result['track_node_features'][entry]

            assert len(fragments_seg) == len(fragments)

            temp = []

            if ('inter_group_pred' in self.result) and ('particles' in self.result) and len(fragments) > 0:

                group_labels = self._fit_predict_groups(entry)
                inter_labels = self._fit_predict_interaction_labels(entry)
                group_ids = get_cluster_label(group_labels.reshape(-1, 1), fragments, column=0)
                inter_ids = get_cluster_label(inter_labels.reshape(-1, 1), fragments, column=0)

            else:
                group_ids = np.ones(len(fragments)).astype(int) * -1
                inter_ids = np.ones(len(fragments)).astype(int) * -1

            if true_id:
                true_fragment_labels = self.data_blob['cluster_label'][entry][:, 5]


            for i, p in enumerate(fragments):
                voxels = point_cloud[p]
                seg_label = fragments_seg[i]
                part = ParticleFragment(self._translate(voxels, volume),
                                i, seg_label,
                                interaction_id=inter_ids[i],
                                group_id=group_ids[i],
                                image_id=entry,
                                voxel_indices=p,
                                depositions=depositions[p],
                                is_primary=False,
                                pid_conf=-1,
                                alias='Fragment',
                                volume=volume)
                temp.append(part)
                if true_id:
                    fid = true_fragment_labels[p]
                    fids, counts = np.unique(fid.astype(int), return_counts=True)
                    part.true_ids = fids
                    part.true_counts = counts

            # Label shower fragments as primaries and attach startpoint
            shower_counter = 0
            for p in np.array(temp)[shower_mask]:
                is_primary = shower_frag_primary[shower_counter]
                p.is_primary = bool(is_primary)
                p.startpoint = shower_node_features[shower_counter][19:22]
                # p.group_id = int(shower_group_pred[shower_counter])
                shower_counter += 1
            assert shower_counter == shower_frag_primary.shape[0]

            # Attach endpoint to track fragments
            track_counter = 0
            for p in temp:
                if p.semantic_type == 1:
                    # p.group_id = int(track_group_pred[track_counter])
                    p.startpoint = track_node_features[track_counter][19:22]
                    p.endpoint = track_node_features[track_counter][22:25]
                    track_counter += 1
            # assert track_counter == track_group_pred.shape[0]

            # Apply fragment voxel cut
            out = []
            for p in temp:
                if p.points.shape[0] < min_particle_voxel_count:
                    continue
                out.append(p)

            # Check primaries and assign ppn points
            if only_primaries:
                out = [p for p in out if p.is_primary]

            if semantic_type is not None:
                out = [p for p in out if p.semantic_type == semantic_type]

            if len(out) == 0:
                return out

            ppn_results = self._fit_predict_ppn(entry)
            match_points_to_particles(ppn_results, out,
                ppn_distance_threshold=attaching_threshold)

            out_fragment_list.extend(out)

        return out_fragment_list


    def get_particles(self, entry, only_primaries=True,
                      min_particle_voxel_count=-1,
                      attaching_threshold=2,
                      volume=None) -> List[Particle]:
        '''
        Method for retriving particle list for given batch index.

        The output particles will have its ppn candidates attached as
        attributes in the form of pandas dataframes (same as _fit_predict_ppn)

        Method also performs endpoint prediction for tracks and startpoint
        prediction for showers.

        1) If a track has no or only one ppn candidate, the endpoints
        will be calculated by selecting two voxels that have the largest
        separation distance. Otherwise, the two ppn candidates with the
        largest separation from the particle coordinate centroid will be
        selected.

        2) If a shower has no ppn candidates, <get_shower_startpoint>
        will raise an Exception. Otherwise it selects the ppn candidate
        with the closest Hausdorff distance to the particle point cloud
        (smallest point-to-set distance)

        Note
        ====
        Particle id is unique only within volume.

        Parameters
        ==========
        entry: int
            Batch number to retrieve example.
        only_primaries: bool, default True
            If set to True, only retrieve predicted primaries.
        min_particle_voxel_count: int, default -1
        attaching_threshold: int, default 2
        volume: int, default None

        Returns
        =======
        list
            List of <Particle> instances (see Particle class definition).
        '''
        self._check_volume(volume)

        if min_particle_voxel_count < 0:
            min_particle_voxel_count = self.min_particle_voxel_count

        entries = self._get_entries(entry, volume)

        out_particle_list = []
        for entry in entries:
            volume = entry % self._num_volumes

            point_cloud      = self.data_blob['input_data'][entry][:, 1:4]
            depositions      = self.result['input_rescaled'][entry][:, 4]
            particles        = self.result['particles'][entry]
            # inter_group_pred = self.result['inter_group_pred'][entry]
            #print(point_cloud.shape, depositions.shape, len(particles))
            particles_seg    = self.result['particles_seg'][entry]

            type_logits = self.result['node_pred_type'][entry]
            input_node_features = [None] * type_logits.shape[0]
            if 'particle_node_features' in self.result:
                input_node_features = self.result['particle_node_features'][entry]
            pids = np.argmax(type_logits, axis=1)

            out = []
            if point_cloud.shape[0] == 0:
                return out
            assert len(particles_seg) == len(particles)
            assert len(pids) == len(particles)
            assert len(input_node_features) == len(particles)
            assert point_cloud.shape[0] == depositions.shape[0]

            node_pred_vtx = self.result['node_pred_vtx'][entry]

            assert node_pred_vtx.shape[0] == len(particles)

            if ('inter_group_pred' in self.result) and ('particles' in self.result) and len(particles) > 0:

                assert len(self.result['inter_group_pred'][entry]) == len(particles)
                inter_labels = self._fit_predict_interaction_labels(entry)
                inter_ids = get_cluster_label(inter_labels.reshape(-1, 1), particles, column=0)

            else:
                inter_ids = np.ones(len(particles)).astype(int) * -1

            for i, p in enumerate(particles):
                voxels = point_cloud[p]
                if voxels.shape[0] < min_particle_voxel_count:
                    continue
                seg_label = particles_seg[i]
                pid = pids[i]
                if seg_label == 2 or seg_label == 3:
                    pid = 1
                interaction_id = inter_ids[i]
                is_primary = bool(np.argmax(node_pred_vtx[i][3:]))
                part = Particle(self._translate(voxels, volume),
                                i,
                                seg_label, interaction_id,
                                pid,
                                batch_id=entry,
                                voxel_indices=p,
                                depositions=depositions[p],
                                is_primary=is_primary,
                                pid_conf=softmax(type_logits[i])[pids[i]],
                                volume=volume)

                part._node_features = input_node_features[i]
                out.append(part)

            if only_primaries:
                out = [p for p in out if p.is_primary]

            if len(out) == 0:
                return out

            ppn_results = self._fit_predict_ppn(entry)

            # Get ppn candidates for particle
            match_points_to_particles(ppn_results, out,
                ppn_distance_threshold=attaching_threshold)

            # Attach startpoint and endpoint
            # as done in full chain geometric encoder
            for p in out:
                if p.size < min_particle_voxel_count:
                    continue
                if p.semantic_type == 0:
                    pt = p._node_features[19:22]
                    # Check startpoint is replicated
                    assert(np.sum(
                        np.abs(pt - p._node_features[22:25])) < 1e-12)
                    p.startpoint = pt
                elif p.semantic_type == 1:
                    startpoint, endpoint = p._node_features[19:22], p._node_features[22:25]
                    p.startpoint = startpoint
                    p.endpoint = endpoint
                else:
                    continue
            out_particle_list.extend(out)

        return out_particle_list


    def get_interactions(self, entry, drop_nonprimary_particles=True, volume=None,
                        compute_vertex=True) -> List[Interaction]:
        '''
        Method for retriving interaction list for given batch index.

        The output particles will have its constituent particles attached as
        attributes as List[Particle].

        Method also performs vertex prediction for each interaction.

        Note
        ----
        Interaction ids are only unique within a volume.

        Parameters
        ----------
        entry: int
            Batch number to retrieve example.
        drop_nonprimary_particles: bool (optional)
            If True, all non-primary particles will not be included in
            the output interactions' .particle attribute.
        volume: int
        compute_vertex: bool, default True

        Returns:
            - out: List of <Interaction> instances (see particle.Interaction).
        '''
        self._check_volume(volume)

        entries = self._get_entries(entry, volume)

        out_interaction_list = []
        for e in entries:
            volume = e % self._num_volumes if self.vb is not None else volume
            particles = self.get_particles(entry, only_primaries=drop_nonprimary_particles, volume=volume)
            out = group_particles_to_interactions_fn(particles)
            for ia in out:
                if compute_vertex:
                    ia.vertex = self._fit_predict_vertex_info(e, ia.id)
                ia.volume = volume
            out_interaction_list.extend(out)

        return out_interaction_list


    def fit_predict_labels(self, entry, volume=None):
        '''
        Predict all labels of a given batch index <entry>.

        We define <labels> to be 1d tensors that annotate voxels.
        '''
        self._check_volume(volume)
        entries = self._get_entries(entry, volume)

        all_pred = {
            'segment': [],
            'fragment': [],
            'group': [],
            'interaction': [],
            'pdg': []
        }
        for entry in entries:
            pred_seg = self._fit_predict_semantics(entry)
            pred_fragments = self._fit_predict_fragments(entry)
            pred_groups = self._fit_predict_groups(entry)
            pred_interaction_labels = self._fit_predict_interaction_labels(entry)
            pred_pids = self._fit_predict_pids(entry)

            pred = {
                'segment': pred_seg,
                'fragment': pred_fragments,
                'group': pred_groups,
                'interaction': pred_interaction_labels,
                'pdg': pred_pids
            }

            for key in pred:
                if len(all_pred[key]) == 0:
                    all_pred[key] = pred[key]
                else:
                    all_pred[key] = np.concatenate([all_pred[key], pred[key]], axis=0)

        self._pred = all_pred

        return all_pred


    def fit_predict(self, **kwargs):
        '''
        Predict all samples in a given batch contained in <data_blob>.

        After calling fit_predict, the prediction information can be accessed
        as follows:

            - self._labels[entry]: labels dict (see fit_predict_labels) for
            batch id <entry>.

            - self._particles[entry]: list of particles for batch id <entry>.

            - self._interactions[entry]: list of interactions for batch id <entry>.
        '''
        labels = []
        list_particles, list_interactions = [], []

        for entry in range(int(self.num_images / self._num_volumes)):

            pred_dict = self.fit_predict_labels(entry)
            labels.append(pred_dict)
            particles = self.get_particles(entry, **kwargs)
            interactions = self.get_interactions(entry)
            list_particles.append(particles)
            list_interactions.append(interactions)

        self._particles = list_particles
        self._interactions = list_interactions
        self._labels = labels

        return labels


class FullChainEvaluator(FullChainPredictor):
    '''
    Helper class for full chain prediction and evaluation.

    Usage:

        model = Trainer._net.module
        entry = 0   # batch id
        predictor = FullChainEvaluator(model, data_blob, res, cfg)
        pred_seg = predictor.get_true_label(entry, mode='segmentation')

    To avoid confusion between different quantities, the label namings under
    iotools.schema must be set as follows:

        schema:
            input_data:
                - parse_sparse3d_scn
                - sparse3d_pcluster
            segment_label:
                - parse_sparse3d_scn
                - sparse3d_pcluster_semantics
            cluster_label:
                - parse_cluster3d_clean_full
                #- parse_cluster3d_full
                - cluster3d_pcluster
                - particle_pcluster
                #- particle_mpv
                - sparse3d_pcluster_semantics
            particles_label:
                - parse_particle_points_with_tagging
                - sparse3d_pcluster
                - particle_corrected
            kinematics_label:
                - parse_cluster3d_kinematics_clean
                - cluster3d_pcluster
                - particle_corrected
                #- particle_mpv
                - sparse3d_pcluster_semantics
            particle_graph:
                - parse_particle_graph_corrected
                - particle_corrected
                - cluster3d_pcluster
            particles_asis:
                - parse_particles
                - particle_pcluster
                - cluster3d_pcluster


    Instructions
    ----------------------------------------------------------------

    The FullChainEvaluator share the same methods as FullChainPredictor,
    with additional methods to retrieve ground truth information for each
    abstraction level.
    '''
    LABEL_TO_COLUMN = {
        'segment': -1,
        'charge': 4,
        'fragment': 5,
        'group': 6,
        'interaction': 7,
        'pdg': 9,
        'nu': 8
    }


    def __init__(self, data_blob, result, cfg, processor_cfg={}, **kwargs):
        super(FullChainEvaluator, self).__init__(data_blob, result, cfg, processor_cfg, **kwargs)
        self.michel_primary_ionization_only = processor_cfg.get('michel_primary_ionization_only', False)

    def get_true_label(self, entry, name, schema='cluster_label', volume=None):
        """
        Retrieve tensor in data blob, labelled with `schema`.

        Parameters
        ==========
        entry: int
        name: str
            Must be a predefined name within `['segment', 'fragment', 'group',
            'interaction', 'pdg', 'nu', 'charge']`.
        schema: str
            Key for dataset schema to retrieve the info from.
        volume: int, default None

        Returns
        =======
        np.array
        """
        if name not in self.LABEL_TO_COLUMN:
            raise KeyError("Invalid label identifier name: {}. "\
                "Available column names = {}".format(
                    name, str(list(self.LABEL_TO_COLUMN.keys()))))
        column_idx = self.LABEL_TO_COLUMN[name]

        self._check_volume(volume)

        entries = self._get_entries(entry, volume)
        out = []
        for entry in entries:
            out.append(self.data_blob[schema][entry][:, column_idx])
        return np.concatenate(out, axis=0)


    def get_predicted_label(self, entry, name, volume=None):
        """
        Returns predicted quantities to label a plot.

        Parameters
        ==========
        entry: int
        name: str
            Must be a predefined name within `['segment', 'fragment', 'group',
            'interaction', 'pdg', 'nu']`.
        volume: int, default None

        Returns
        =======
        np.array
        """
        pred = self.fit_predict_labels(entry, volume=volume)
        return pred[name]


    def _apply_true_voxel_cut(self, entry):

        labels = self.data_blob['cluster_label_noghost'][entry]

        particle_ids = set(list(np.unique(labels[:, 6]).astype(int)))
        particles_exclude = []

        for idx, p in enumerate(self.data_blob['particles_asis'][entry]):
            pid = int(p.id())
            if pid not in particle_ids:
                continue
            is_primary = p.group_id() == p.parent_id()
            if p.pdg_code() not in TYPE_LABELS:
                continue
            mask = labels[:, 6].astype(int) == pid
            coords = labels[mask][:, 1:4]
            if coords.shape[0] < self.min_particle_voxel_count:
                particles_exclude.append(p.id())

        return set(particles_exclude)


    def get_true_fragments(self, entry, verbose=False, volume=None) -> List[TruthParticleFragment]:
        '''
        Get list of <TruthParticleFragment> instances for given <entry> batch id.
        '''
        self._check_volume(volume)

        entries = self._get_entries(entry, volume)

        out_fragments_list = []
        for entry in entries:
            volume = entry % self._num_volumes

            # Both are "adapted" labels
            labels = self.data_blob['cluster_label'][entry]
            segment_label = self.data_blob['segment_label'][entry][:, -1]
            rescaled_input_charge = self.result['input_rescaled'][entry][:, 4]

            fragment_ids = set(list(np.unique(labels[:, 5]).astype(int)))
            fragments = []

            for fid in fragment_ids:
                mask = labels[:, 5] == fid

                semantic_type, counts = np.unique(labels[:, -1][mask], return_counts=True)
                if semantic_type.shape[0] > 1:
                    if verbose:
                        print("Semantic Type of Fragment {} is not "\
                            "unique: {}, {}".format(fid,
                                                    str(semantic_type),
                                                    str(counts)))
                    perm = counts.argmax()
                    semantic_type = semantic_type[perm]
                else:
                    semantic_type = semantic_type[0]

                points = labels[mask][:, 1:4]
                size = points.shape[0]
                depositions = rescaled_input_charge[mask]
                depositions_MeV = labels[mask][:, 4]
                voxel_indices = np.where(mask)[0]

                group_id, counts = np.unique(labels[:, 6][mask].astype(int), return_counts=True)
                if group_id.shape[0] > 1:
                    if verbose:
                        print("Group ID of Fragment {} is not "\
                            "unique: {}, {}".format(fid,
                                                    str(group_id),
                                                    str(counts)))
                    perm = counts.argmax()
                    group_id = group_id[perm]
                else:
                    group_id = group_id[0]

                interaction_id, counts = np.unique(labels[:, 7][mask].astype(int), return_counts=True)
                if interaction_id.shape[0] > 1:
                    if verbose:
                        print("Interaction ID of Fragment {} is not "\
                            "unique: {}, {}".format(fid,
                                                    str(interaction_id),
                                                    str(counts)))
                    perm = counts.argmax()
                    interaction_id = interaction_id[perm]
                else:
                    interaction_id = interaction_id[0]


                is_primary, counts = np.unique(labels[:, -2][mask].astype(bool), return_counts=True)
                if is_primary.shape[0] > 1:
                    if verbose:
                        print("Primary label of Fragment {} is not "\
                            "unique: {}, {}".format(fid,
                                                    str(is_primary),
                                                    str(counts)))
                    perm = counts.argmax()
                    is_primary = is_primary[perm]
                else:
                    is_primary = is_primary[0]

                part = TruthParticleFragment(self._translate(points, volume),
                                fid, semantic_type,
                                interaction_id=interaction_id,
                                group_id=group_id,
                                image_id=entry,
                                voxel_indices=voxel_indices,
                                depositions=depositions,
                                depositions_MeV=depositions_MeV,
                                is_primary=is_primary,
                                alias='Fragment',
                                volume=volume)

                fragments.append(part)
            out_fragments_list.extend(fragments)

        return out_fragments_list


    def get_true_particles(self, entry, only_primaries=True,
                           verbose=False, volume=None) -> List[TruthParticle]:
        '''
        Get list of <TruthParticle> instances for given <entry> batch id.

        The method will return particles only if its id number appears in
        the group_id column of cluster_label.

        Each TruthParticle will contain the following information (attributes):

            points: N x 3 coordinate array for particle's full image.
            id: group_id
            semantic_type: true semantic type
            interaction_id: true interaction id
            pid: PDG type (photons: 0, electrons: 1, ...)
            fragments: list of integers corresponding to constituent fragment
                id number
            p: true momentum vector
        '''
        self._check_volume(volume)

        entries = self._get_entries(entry, volume)

        out_particles_list = []
        global_entry = entry
        for entry in entries:
            volume = entry % self._num_volumes

            labels = self.data_blob['cluster_label'][entry]
            if self.deghosting:
                labels_noghost = self.data_blob['cluster_label_noghost'][entry]
            segment_label = self.data_blob['segment_label'][entry][:, -1]
            particle_ids = set(list(np.unique(labels[:, 6]).astype(int)))
            rescaled_input_charge = self.result['input_rescaled'][entry][:, 4]

            particles = []
            exclude_ids = set([])

            for idx, p in enumerate(self.data_blob['particles_asis'][global_entry]):
                pid = int(p.id())
                # 1. Check if current pid is one of the existing group ids
                if pid not in particle_ids:
                    # print("PID {} not in particle_ids".format(pid))
                    continue
                is_primary = p.group_id() == p.parent_id()
                if p.pdg_code() not in TYPE_LABELS:
                    # print("PID {} not in TYPE LABELS".format(pid))
                    continue
                # For deghosting inputs, perform voxel cut with true nonghost coords.
                if self.deghosting:
                    exclude_ids = self._apply_true_voxel_cut(global_entry)
                    if pid in exclude_ids:
                        # Skip this particle if its below the voxel minimum requirement
                        # print("PID {} was excluded from the list of particles due"\
                        #     " to true nonghost voxel cut. Exclude IDS = {}".format(
                        #         p.id(), str(exclude_ids)
                        #     ))
                        continue

                pdg = TYPE_LABELS[p.pdg_code()]
                mask = labels[:, 6].astype(int) == pid
                if self.deghosting:
                    mask_noghost = labels_noghost[:, 6].astype(int) == pid
                # If particle is Michel electron, we have the option to
                # only consider the primary ionization.
                # Semantic labels only label the primary ionization as Michel.
                # Cluster labels will have the entire Michel together.
                if self.michel_primary_ionization_only and 2 in labels[mask][:, -1].astype(int):
                    mask = mask & (labels[:, -1].astype(int) == 2)
                    if self.deghosting:
                        mask_noghost = mask_noghost & (labels_noghost[:, -1].astype(int) == 2)

                # Check semantics
                semantic_type, sem_counts = np.unique(
                    labels[mask][:, -1].astype(int), return_counts=True)

                if semantic_type.shape[0] > 1:
                    if verbose:
                        print("Semantic Type of Particle {} is not "\
                            "unique: {}, {}".format(pid,
                                                    str(semantic_type),
                                                    str(sem_counts)))
                    perm = sem_counts.argmax()
                    semantic_type = semantic_type[perm]
                else:
                    semantic_type = semantic_type[0]



                coords = self.data_blob['input_data'][entry][mask][:, 1:4]

                interaction_id, int_counts = np.unique(labels[mask][:, 7].astype(int),
                                                       return_counts=True)
                if interaction_id.shape[0] > 1:
                    if verbose:
                        print("Interaction ID of Particle {} is not "\
                            "unique: {}".format(pid, str(interaction_id)))
                    perm = int_counts.argmax()
                    interaction_id = interaction_id[perm]
                else:
                    interaction_id = interaction_id[0]

                nu_id, nu_counts = np.unique(labels[mask][:, 8].astype(int),
                                             return_counts=True)
                if nu_id.shape[0] > 1:
                    if verbose:
                        print("Neutrino ID of Particle {} is not "\
                            "unique: {}".format(pid, str(nu_id)))
                    perm = nu_counts.argmax()
                    nu_id = nu_id[perm]
                else:
                    nu_id = nu_id[0]

                fragments = np.unique(labels[mask][:, 5].astype(int))
                depositions_MeV = labels[mask][:, 4]
                depositions = rescaled_input_charge[mask] # Will be in ADC
                coords_noghost, depositions_noghost = None, None
                if self.deghosting:
                    coords_noghost = labels_noghost[mask_noghost][:, 1:4]
                    depositions_noghost = labels_noghost[mask_noghost][:, 4].squeeze()

                particle = TruthParticle(self._translate(coords, volume),
                    pid,
                    semantic_type, interaction_id, pdg,
                    particle_asis=p,
                    batch_id=entry,
                    depositions=depositions,
                    is_primary=is_primary,
                    coords_noghost=coords_noghost,
                    depositions_noghost=depositions_noghost,
                    depositions_MeV=depositions_MeV,
                    volume=entry % self._num_volumes)

                particle.p = np.array([p.px(), p.py(), p.pz()])
                particle.fragments = fragments
                particle.particle_asis = p
                particle.nu_id = nu_id
                particle.voxel_indices = np.where(mask)[0]

                particle.startpoint = np.array([p.first_step().x(),
                                                p.first_step().y(),
                                                p.first_step().z()])

                if semantic_type == 1:
                    particle.endpoint = np.array([p.last_step().x(),
                                                  p.last_step().y(),
                                                  p.last_step().z()])

                if particle.voxel_indices.shape[0] >= self.min_particle_voxel_count:
                    particles.append(particle)

            out_particles_list.extend(particles)

        if only_primaries:
            out_particles_list = [p for p in out_particles_list if p.is_primary]

        return out_particles_list


    def get_true_interactions(self, entry, drop_nonprimary_particles=True,
                              min_particle_voxel_count=-1,
                              volume=None,
                              compute_vertex=True) -> List[Interaction]:
        self._check_volume(volume)
        if min_particle_voxel_count < 0:
            min_particle_voxel_count = self.min_particle_voxel_count

        entries = self._get_entries(entry, volume)
        out_interactions_list = []
        for e in entries:
            volume = e % self._num_volumes if self.vb is not None else volume
            true_particles = self.get_true_particles(entry, only_primaries=drop_nonprimary_particles, volume=volume)
            out = group_particles_to_interactions_fn(true_particles,
                                                     get_nu_id=True, mode='truth')
            if compute_vertex:
                vertices = self.get_true_vertices(entry, volume=volume)
            for ia in out:
                if compute_vertex:
                    ia.vertex = vertices[ia.id]
                ia.volume = volume
            out_interactions_list.extend(out)

        return out_interactions_list


    def get_true_vertices(self, entry, volume=None):
        """
        Parameters
        ==========
        entry: int
        volume: int, default None

        Returns
        =======
        dict
            Keys are true interactions ids, values are np.array of shape (N, 3)
            with true vertices coordinates.
        """
        self._check_volume(volume)

        entries = self._get_entries(entry, volume)
        out = {}
        for entry in entries:
            volume = entry % self._num_volumes if self.vb is not None else volume
            inter_idxs = np.unique(
                self.data_blob['cluster_label'][entry][:, 7].astype(int))
            for inter_idx in inter_idxs:
                if inter_idx < 0:
                    continue
                vtx = get_vertex(self.data_blob['kinematics_label'],
                                self.data_blob['cluster_label'],
                                data_idx=entry,
                                inter_idx=inter_idx)
                out[inter_idx] = self._translate(vtx, volume)

        return out


    def match_particles(self, entry,
                        only_primaries=False,
                        mode='pred_to_true',
                        volume=None, **kwargs):
        '''
        Returns (<Particle>, None) if no match was found

        Parameters
        ==========
        entry: int
        only_primaries: bool, default False
        mode: str, default 'pred_to_true'
            Must be either 'pred_to_true' or 'true_to_pred'
        volume: int, default None
        '''
        self._check_volume(volume)

        entries = self._get_entries(entry, volume)
        all_matches = []
        for e in entries:
            volume = e % self._num_volumes if self.vb is not None else volume
            print('matching', entries, volume)
            if mode == 'pred_to_true':
                # Match each pred to one in true
                particles_from = self.get_particles(entry, only_primaries=only_primaries, volume=volume)
                particles_to = self.get_true_particles(entry, only_primaries=only_primaries, volume=volume)
            elif mode == 'true_to_pred':
                # Match each true to one in pred
                particles_to = self.get_particles(entry, only_primaries=only_primaries, volume=volume)
                particles_from = self.get_true_particles(entry, only_primaries=only_primaries, volume=volume)
            else:
                raise ValueError("Mode {} is not valid. For matching each"\
                    " prediction to truth, use 'pred_to_true' (and vice versa).".format(mode))
            all_kwargs = {"min_overlap": self.min_overlap_count, "overlap_mode": self.overlap_mode, **kwargs}
            matched_pairs, _, _ = match_particles_fn(particles_from, particles_to,
                                                    **all_kwargs)
            all_matches.extend(matched_pairs)
        return all_matches


    def match_interactions(self, entry, mode='pred_to_true',
                           drop_nonprimary_particles=True,
                           match_particles=True,
                           return_counts=False,
                           volume=None,
                           compute_vertex=True,
                           **kwargs):
        """
        Parameters
        ==========
        entry: int
        mode: str, default 'pred_to_true'
            Must be either 'pred_to_true' or 'true_to_pred'.
        drop_nonprimary_particles: bool, default True
        match_particles: bool, default True
        return_counts: bool, default False
        volume: int, default None

        Returns
        =======
        List[Tuple[Interaction, Interaction]]
            List of tuples, indicating the matched interactions.
        """
        self._check_volume(volume)

        entries = self._get_entries(entry, volume)
        all_matches, all_counts = [], []
        for e in entries:
            volume = e % self._num_volumes if self.vb is not None else volume
            if mode == 'pred_to_true':
                ints_from = self.get_interactions(entry, drop_nonprimary_particles=drop_nonprimary_particles, volume=volume, compute_vertex=compute_vertex)
                ints_to = self.get_true_interactions(entry, drop_nonprimary_particles=drop_nonprimary_particles, volume=volume, compute_vertex=compute_vertex)
            elif mode == 'true_to_pred':
                ints_to = self.get_interactions(entry, drop_nonprimary_particles=drop_nonprimary_particles, volume=volume, compute_vertex=compute_vertex)
                ints_from = self.get_true_interactions(entry, drop_nonprimary_particles=drop_nonprimary_particles, volume=volume, compute_vertex=compute_vertex)
            else:
                raise ValueError("Mode {} is not valid. For matching each"\
                    " prediction to truth, use 'pred_to_true' (and vice versa).".format(mode))

            all_kwargs = {"min_overlap": self.min_overlap_count, **kwargs}
            matched_interactions, _, counts = match_interactions_fn(ints_from, ints_to,
                                                                    **all_kwargs)

            if match_particles:
                for interactions in matched_interactions:
                    domain, codomain = interactions
                    if codomain is None:
                        domain_particles, codomain_particles = domain.particles, []
                    else:
                        domain_particles, codomain_particles = domain.particles, codomain.particles
                        # continue
                    matched_particles, _, _ = match_particles_fn(domain_particles, codomain_particles,
                                                                min_overlap=self.min_overlap_count,
                                                                overlap_mode=self.overlap_mode)
            all_matches.extend(matched_interactions)
            all_counts.extend(counts)

        if return_counts:
            return all_matches, all_counts
        else:
            return all_matches
