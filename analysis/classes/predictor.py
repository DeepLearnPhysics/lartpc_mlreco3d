from typing import List
import numpy as np
import os
import time
from collections import OrderedDict

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.metrics import unique_label

from scipy.special import softmax
from analysis.classes import (Particle, 
                              Interaction, 
                              ParticleBuilder, 
                              InteractionBuilder, 
                              FragmentBuilder)
from analysis.producers.point_matching import *

from scipy.special import softmax


class FullChainPredictor:
    '''
    User Interface for full chain inference.

    Usage Example:

        model = Trainer._net.module
        entry = 0   # batch id
        predictor = FullChainPredictor(model, data_blob, res, 
                                       predictor_cfg=predictor_cfg)
        particles = predictor.get_particles(entry)

    Instructions
    -----------------------------------------------------------------------
    '''
    def __init__(self, data_blob, result, predictor_cfg={}):

        self.data_blob = data_blob
        self.result = result

        self.particle_builder    = ParticleBuilder()
        self.interaction_builder = InteractionBuilder()
        self.fragment_builder    = FragmentBuilder()

        build_reps = predictor_cfg.get('build_reps', ['particles', 'interactions'])
        self.builders = OrderedDict()
        for key in build_reps:
            if key == 'particles':
                self.builders[key] = ParticleBuilder()
            if key == 'interactions':
                self.builders[key] = InteractionBuilder()
            if key == 'Fragments':
                self.builders[key] = FragmentBuilder()

        # Data Structure Scopes
        self.scope = predictor_cfg.get('scope', ['particles', 'interactions'])

        # self.build_representations()

        self.num_images = len(self.data_blob['index'])
        self.index = self.data_blob['index']

        self.spatial_size             = predictor_cfg.get('spatial_size', 6144)
        # Minimum voxel count for a true non-ghost particle to be considered
        self.min_particle_voxel_count = predictor_cfg.get('min_particle_voxel_count', 20)
        # We want to count how well we identify interactions with some PDGs
        # as primary particles
        self.primary_pdgs             = np.unique(predictor_cfg.get('primary_pdgs', []))

        self.primary_score_threshold  = predictor_cfg.get('primary_score_threshold', None)
        # This is used to apply fiducial volume cuts.
        # Min/max boundaries in each dimension haev to be specified.
        self.vb = predictor_cfg.get('volume_boundaries', None)
        self.set_volume_boundaries()


    def set_volume_boundaries(self, use_pixels=True):
        if self.vb is None:
            # Using ICARUS Cryo 0 as a default
            pass
        else:
            self.vb = np.array(self.vb, dtype=np.float64)
            if 'meta' not in self.data_blob:
                msg = "Cannot use volume boundaries because meta is "\
                    "missing from iotools config."
                raise Exception(msg)
            else:
                meta = self.data_blob['meta'][0]
                if use_pixels:
                    min_x, min_y, min_z = meta[0:3]
                    size_voxel_x, size_voxel_y, size_voxel_z = meta[6:9]

                    self.vb[0, :] = (self.vb[0, :] - min_x) / size_voxel_x
                    self.vb[1, :] = (self.vb[1, :] - min_y) / size_voxel_y
                    self.vb[2, :] = (self.vb[2, :] - min_z) / size_voxel_z
                else:
                    max_x, max_y, max_z = meta[3:6]
                    self.vb[0, 0], self.vb[0, 1] = min_x, max_x
                    self.vb[1, 0], self.vb[1, 1] = min_y, max_y
                    self.vb[2, 0], self.vb[2, 1] = min_z, max_z


    def build_representations(self):
        for key in self.builders:
            if key not in self.result and key in self.scope:
                self.result[key] = self.builders[key].build(self.data_blob, 
                                                            self.result, 
                                                            mode='reco')
    def __repr__(self):
        msg = "FullChainEvaluator(num_images={})".format(int(self.num_images))
        return msg


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

        # min_points = self.module_config['graph_spice'].get('min_points', 1)
        # invert = self.module_config['graph_spice_loss'].get('invert', True)

        graph_info['Index'] = graph_info['Index'].map(index_mapping)
        constructor_cfg = self.cluster_graph_constructor.constructor_cfg
        gs_manager = ClusterGraphConstructor(constructor_cfg,
                                             graph_batch=graph,
                                             graph_info=graph_info,
                                             batch_col=0,
                                             training=False)
        pred, G, subgraph = gs_manager.fit_predict_one(entry,
                                                       invert=True,
                                                       min_points=1)

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
        fragments = self.result['fragment_clusts'][entry]

        num_voxels = self.result['input_rescaled'][entry].shape[0]
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
        particles = self.result['particle_clusts'][entry]
        num_voxels = self.result['input_rescaled'][entry].shape[0]
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
        inter_group_pred = self.result['particle_group_pred'][entry]
        particles = self.result['particle_clusts'][entry]
        num_voxels = self.result['input_rescaled'][entry].shape[0]
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
        particles = self.result['particle_clusts'][entry]
        type_logits = self.result['particle_node_pred_type'][entry]
        pids = np.argmax(type_logits, axis=1)
        num_voxels = self.result['input_rescaled'][entry].shape[0]

        pred_pids = -np.ones(num_voxels).astype(int)

        for i, mask in enumerate(particles):
            pred_pids[mask] = pids[i]

        return pred_pids

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

    def get_fragments(self, entry, only_primaries=False,
                      min_particle_voxel_count=-1,
                      attaching_threshold=2,
                      semantic_type=None, verbose=False,
                      true_id=False, volume=None, allow_nodes=[0, 2, 3]) -> List[Particle]:
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
        out_fragment_list = self.result['ParticleFragments'][entry]
        return out_fragment_list
    
    def _get_primary_labels(self, node_pred_vtx):
        primary_labels = -np.ones(len(node_pred_vtx)).astype(int)
        primary_scores = np.zeros(len(node_pred_vtx)).astype(float)
        if node_pred_vtx.shape[1] == 5:
            primary_scores = node_pred_vtx[:, 3:]
        elif node_pred_vtx.shape[1] == 2:
            primary_scores = node_pred_vtx
        else:
            raise ValueError('<node_pred_vtx> must either be (N, 5) or (N, 2)')
        primary_scores = softmax(node_pred_vtx, axis=1)
        if self.primary_score_threshold is None:
            primary_labels = np.argmax(primary_scores, axis=1)
        else:
            primary_labels = primary_scores[:, 1] > self.primary_score_threshold
        return primary_labels


    def get_particles(self, entry, only_primaries=False, volume=None) -> List[Particle]:
        '''
        Method for retriving particle list for given batch index.

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
        out = self.result['particles'][entry]
        out = self._decorate_particles(entry, out,
                                       only_primaries=only_primaries,
                                       volume=volume)
        return out
    

    def _decorate_particles(self, entry, particles, **kwargs):
        
        # Decorate particles
        for i, p in enumerate(particles):
            if 'particle_length' in self.result:
                p.length = self.result['particle_length'][entry][i]
            if 'particle_range_based_energy' in self.result:
                energy = self.result['particle_range_based_energy'][entry][i]
                if energy > 0: p.csda_energy = energy
            if 'particle_calo_energy' in self.result:
                p.calo_energy = self.result['particle_calo_energy'][entry][i]
            if 'particle_start_directions' in self.result:
                p.direction = self.result['particle_start_directions'][entry][i]

        out = [p for p in particles]
        # Filtering actions on particles
        if kwargs.get('only_primaries', False):
            out = [p for p in particles if p.is_primary]

        if len(out) == 0:
            return out

        volume = kwargs.get('volume', None)
        if volume is not None:
            out = [p for p in out if p.volume == volume]
        return out

    def _decorate_interactions(self, interactions, **kwargs):
        pass

    def get_interactions(self, entry, 
                         drop_nonprimary_particles=True, 
                         volume=None,
                         get_vertex=True) -> List[Interaction]:
        '''
        Method for retriving interaction list for given batch index.

        The output particles will have its constituent particles attached as
        attributes as List[Particle].

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
        get_vertex: bool, default True

        Returns:
            - out: List of <Interaction> instances (see particle.Interaction).
        '''
        out = self.result['interactions'][entry]
        return out


    def fit_predict_labels(self, entry):
        '''
        Predict all labels of a given batch index <entry>.

        We define <labels> to be 1d tensors that annotate voxels.
        '''

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

        self._pred = pred

        return pred


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

        for entry in range(self.num_images):

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
