from typing import Callable, Tuple, List
import numpy as np
import os
import time

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.metrics import unique_label
from collections import defaultdict

from scipy.special import softmax
from analysis.classes import Particle, ParticleFragment, TruthParticleFragment, \
        TruthParticle, Interaction, TruthInteraction, FlashManager
from analysis.classes.particle import matrix_counts, matrix_iou, \
        match_particles_fn, match_interactions_fn, group_particles_to_interactions_fn, \
        match_interactions_optimal, match_particles_optimal
from analysis.algorithms.point_matching import *

from mlreco.utils.groups import type_labels as TYPE_LABELS
from mlreco.utils.vertex import get_vertex
from analysis.algorithms.vertex import estimate_vertex
from analysis.algorithms.utils import correct_track_points
from mlreco.utils.deghosting import deghost_labels_and_predictions

from mlreco.utils.gnn.cluster import get_cluster_label, form_clusters
from mlreco.iotools.collates import VolumeBoundaries

from analysis.classes.predictor import FullChainPredictor


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

        labels = self.data_blob['cluster_label_true_nonghost'][entry]

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
                labels_noghost = self.data_blob['cluster_label_true_nonghost'][entry]
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
                    semantic_type, interaction_id, pdg, entry,
                    particle_asis=p,
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
                        volume=None, 
                        matching_mode='one_way', 
                        **kwargs):
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
            if matching_mode == 'one_way':
                matched_pairs, _ = match_particles_fn(particles_from, particles_to,
                                                        **all_kwargs)
            elif matching_mode == 'optimal':
                matched_pairs, _ = match_particles_optimal(particles_from, particles_to,
                                                           **all_kwargs)
            else:
                raise ValueError
            all_matches.extend(matched_pairs)
        return all_matches

    
    def match_interactions(self, entry, mode='pred_to_true',
                           drop_nonprimary_particles=True,
                           match_particles=True,
                           return_counts=False,
                           volume=None,
                           compute_vertex=True,
                           vertex_mode='all',
                           matching_mode='one_way',
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
                ints_from = self.get_interactions(entry, 
                                                  drop_nonprimary_particles=drop_nonprimary_particles, 
                                                  volume=volume, 
                                                  compute_vertex=compute_vertex,
                                                  vertex_mode=vertex_mode)
                ints_to = self.get_true_interactions(entry, 
                                                     drop_nonprimary_particles=drop_nonprimary_particles, 
                                                     volume=volume, 
                                                     compute_vertex=compute_vertex)
            elif mode == 'true_to_pred':
                ints_to = self.get_interactions(entry, 
                                                drop_nonprimary_particles=drop_nonprimary_particles, 
                                                volume=volume, 
                                                compute_vertex=compute_vertex,
                                                vertex_mode=vertex_mode)
                ints_from = self.get_true_interactions(entry, 
                                                       drop_nonprimary_particles=drop_nonprimary_particles, 
                                                       volume=volume, 
                                                       compute_vertex=compute_vertex)
            else:
                raise ValueError("Mode {} is not valid. For matching each"\
                    " prediction to truth, use 'pred_to_true' (and vice versa).".format(mode))

            all_kwargs = {"min_overlap": self.min_overlap_count, "overlap_mode": self.overlap_mode, **kwargs}
            if matching_mode == 'one_way':
                matched_interactions, counts = match_interactions_fn(ints_from, ints_to,
                                                                        **all_kwargs)
            elif matching_mode == 'optimal':
                matched_interactions, counts = match_interactions_optimal(ints_from, ints_to,
                                                                          **all_kwargs)
            else:
                raise ValueError
            if len(matched_interactions) == 0:
                continue
            if match_particles:
                for interactions in matched_interactions:
                    domain, codomain = interactions
                    domain_particles, codomain_particles = [], []
                    if domain is not None:
                        domain_particles = domain.particles
                    if codomain is not None:
                        codomain_particles = codomain.particles
                        # continue
                    if matching_mode == 'one_way':
                        matched_particles, _ = match_particles_fn(domain_particles, codomain_particles,
                                                                    min_overlap=self.min_overlap_count,
                                                                    overlap_mode=self.overlap_mode)
                    elif matching_mode == 'optimal':
                        matched_particles, _ = match_particles_optimal(domain_particles, codomain_particles,
                                                                       min_overlap=self.min_overlap_count,
                                                                       overlap_mode=self.overlap_mode)
                    else:
                        raise ValueError
            all_matches.extend(matched_interactions)
            all_counts.extend(counts)

        if return_counts:
            return all_matches, all_counts
        else:
            return all_matches