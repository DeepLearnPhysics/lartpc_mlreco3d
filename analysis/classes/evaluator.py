from typing import List
import numpy as np

from mlreco.utils.globals import VTX_COLS, INTER_COL, COORD_COLS

from analysis.classes import TruthParticleFragment, TruthParticle, Interaction
from analysis.classes.particle_utils import (match_particles_fn, 
                                             match_interactions_fn, 
                                             group_particles_to_interactions_fn, 
                                             match_interactions_optimal, 
                                             match_particles_optimal)
from analysis.algorithms.point_matching import *

from mlreco.utils.groups import type_labels as TYPE_LABELS
from mlreco.utils.vertex import get_vertex

from analysis.classes.predictor import FullChainPredictor


def get_true_particle_labels(labels, mask, pid=-1, verbose=False):
    semantic_type, sem_counts = np.unique(labels[mask][:, -1].astype(int), 
                                            return_counts=True)
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

    return semantic_type, interaction_id, nu_id


def handle_empty_true_particles(labels_noghost,  mask_noghost, p, entry, 
                                verbose=False):
    pid = int(p.id())
    pdg = TYPE_LABELS.get(p.pdg_code(), -1)
    is_primary = p.group_id() == p.parent_id()

    semantic_type, interaction_id, nu_id = -1, -1, -1
    coords, depositions, voxel_indices = np.array([]), np.array([]), np.array([])
    coords_noghost, depositions_noghost = np.array([]), np.array([])
    if np.count_nonzero(mask_noghost) > 0:
        coords_noghost = labels_noghost[mask_noghost][:, 1:4]
        depositions_noghost = labels_noghost[mask_noghost][:, 4].squeeze()
        semantic_type, interaction_id, nu_id = get_true_particle_labels(labels_noghost, 
                                                                        mask_noghost, 
                                                                        pid=pid, 
                                                                        verbose=verbose)
    particle = TruthParticle(coords,
        pid, semantic_type, interaction_id, pdg, 
        entry, particle_asis=p,
        depositions=depositions,
        is_primary=is_primary,
        coords_noghost=coords_noghost,
        depositions_noghost=depositions_noghost,
        depositions_MeV=depositions)
    particle.p = np.array([p.px(), p.py(), p.pz()])
    particle.fragments = []
    particle.particle_asis = p
    particle.nu_id = nu_id
    particle.voxel_indices = voxel_indices

    particle.startpoint = np.array([p.first_step().x(),
                                    p.first_step().y(),
                                    p.first_step().z()])

    if semantic_type == 1:
        particle.endpoint = np.array([p.last_step().x(),
                                    p.last_step().y(),
                                    p.last_step().z()])
    return particle


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

    def get_true_label(self, entry, name, schema='cluster_label_adapted'):
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

        out = self.result[schema][entry][:, column_idx]
        return np.concatenate(out, axis=0)


    def get_predicted_label(self, entry, name):
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
        pred = self.fit_predict_labels(entry)
        return pred[name]


    def _apply_true_voxel_cut(self, entry):

        labels = self.data_blob['cluster_label'][entry]

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


    def get_true_fragments(self, entry, verbose=False) -> List[TruthParticleFragment]:
        '''
        Get list of <TruthParticleFragment> instances for given <entry> batch id.
        '''

        fragments = []

        # Both are "adapted" labels
        labels = self.result['cluster_label_adapted'][entry]
        rescaled_input_charge = self.result['input_rescaled'][entry][:, 4]

        fragment_ids = set(list(np.unique(labels[:, 5]).astype(int)))

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
                            alias='Fragment')

            fragments.append(part)

        return fragments


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
        out_particles_list = []

        labels = self.result['cluster_label_adapted'][entry]
        labels_noghost = self.data_blob['cluster_label'][entry]
        particle_ids = set(list(np.unique(labels[:, 6]).astype(int)))
        rescaled_input_charge = self.result['input_rescaled'][entry][:, 4]

        particles = []
        exclude_ids = set([])

        for idx, p in enumerate(self.data_blob['particles_asis'][entry]):
            pid = int(p.id())
            pdg = TYPE_LABELS.get(p.pdg_code(), -1)
            is_primary = p.group_id() == p.parent_id()
            mask_noghost = labels_noghost[:, 6].astype(int) == pid
            if np.count_nonzero(mask_noghost) <= 0:
                continue
            # 1. Check if current pid is one of the existing group ids
            if pid not in particle_ids:
                particle = handle_empty_true_particles(labels_noghost, mask_noghost, p, entry, 
                                                       verbose=verbose)
                particles.append(particle)
                continue

            # 1. Process voxels
            mask = labels[:, 6].astype(int) == pid
            # If particle is Michel electron, we have the option to
            # only consider the primary ionization.
            # Semantic labels only label the primary ionization as Michel.
            # Cluster labels will have the entire Michel together.
            if self.michel_primary_ionization_only and 2 in labels[mask][:, -1].astype(int):
                mask = mask & (labels[:, -1].astype(int) == 2)
                mask_noghost = mask_noghost & (labels_noghost[:, -1].astype(int) == 2)

            coords = self.result['input_rescaled'][entry][mask][:, 1:4]
            voxel_indices = np.where(mask)[0]
            fragments = np.unique(labels[mask][:, 5].astype(int))
            depositions_MeV = labels[mask][:, 4]
            depositions = rescaled_input_charge[mask] # Will be in ADC
            coords_noghost = labels_noghost[mask_noghost][:, 1:4]
            depositions_noghost = labels_noghost[mask_noghost][:, 4].squeeze()

            volume_labels = labels_noghost[mask_noghost][:, 0]
            volume_id, cts = np.unique(volume_labels, return_counts=True)
            volume_id = int(volume_id[cts.argmax()])

            # 2. Process particle-level labels
            if p.pdg_code() not in TYPE_LABELS:
                # print("PID {} not in TYPE LABELS".format(pid))
                continue
            exclude_ids = self._apply_true_voxel_cut(entry)
            if pid in exclude_ids:
                # Skip this particle if its below the voxel minimum requirement
                continue
            
            semantic_type, interaction_id, nu_id = get_true_particle_labels(labels, mask, pid=pid, verbose=verbose)

            particle = TruthParticle(coords,
                pid,
                semantic_type, interaction_id, pdg, entry,
                particle_asis=p,
                depositions=depositions,
                is_primary=is_primary,
                coords_noghost=coords_noghost,
                depositions_noghost=depositions_noghost,
                depositions_MeV=depositions_MeV,
                volume=volume_id)

            particle.p = np.array([p.px(), p.py(), p.pz()])
            particle.fragments = fragments
            particle.particle_asis = p
            particle.nu_id = nu_id
            particle.voxel_indices = voxel_indices

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
        if volume is not None:
            out_particles_list = [p for p in out_particles_list if p.volume == volume]

        return out_particles_list


    def get_true_interactions(self, entry, drop_nonprimary_particles=True,
                              min_particle_voxel_count=-1,
                              compute_vertex=True,
                              volume=None,
                              tag_pi0=False) -> List[Interaction]:
        if min_particle_voxel_count < 0:
            min_particle_voxel_count = self.min_particle_voxel_count

        out = []
        true_particles = self.get_true_particles(entry, 
                                                 only_primaries=drop_nonprimary_particles,
                                                 volume=volume)
        out = group_particles_to_interactions_fn(true_particles,
                                                 get_nu_id=True, 
                                                 mode='truth',
                                                 tag_pi0=tag_pi0)
        if compute_vertex:
            vertices = self.get_true_vertices(entry)
        for ia in out:
            if compute_vertex and ia.id in vertices:
                ia.vertex = vertices[ia.id]

            if 'neutrino_asis' in self.data_blob and ia.nu_id > 0:
                # assert 'particles_asis' in data_blob
                # particles = data_blob['particles_asis'][i]
                neutrinos = self.data_blob['neutrino_asis'][entry]
                if len(neutrinos) > 1 or len(neutrinos) == 0: continue
                nu = neutrinos[0]
                # Get larcv::Particle objects for each
                # particle of the true interaction
                # true_particles = np.array(particles)[np.array([p.id for p in true_int.particles])]
                # true_particles_track_ids = [p.track_id() for p in true_particles]
                # for nu in neutrinos:
                #     if nu.mct_index() not in true_particles_track_ids: continue
                ia.nu_info = {
                    'interaction_type': nu.interaction_type(),
                    'interaction_mode': nu.interaction_mode(),
                    'current_type': nu.current_type(),
                    'energy_init': nu.energy_init()
                }

        return out


    def get_true_vertices(self, entry):
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
        out = {}
        inter_idxs = np.unique(
            self.data_blob['cluster_label'][entry][:, INTER_COL].astype(int))
        for inter_idx in inter_idxs:
            if inter_idx < 0:
                continue
            vtx = get_vertex(self.data_blob['cluster_label'],
                            self.data_blob['cluster_label'],
                            data_idx=entry,
                            inter_idx=inter_idx,
                            vtx_col=VTX_COLS[0])
            mask = self.data_blob['cluster_label'][entry][:, INTER_COL].astype(int) == inter_idx
            points = self.data_blob['cluster_label'][entry][:, COORD_COLS[0]:COORD_COLS[-1]+1]
            new_vtx = points[mask][np.linalg.norm(points[mask] - vtx, axis=1).argmin()]
            out[inter_idx] = new_vtx

        return out


    def match_particles(self, entry,
                        only_primaries=False,
                        mode='pred_to_true',
                        matching_mode='one_way', 
                        return_counts=False,
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
        all_matches = []
        all_counts = []
        # print('matching', entries, volume)
        if mode == 'pred_to_true':
            # Match each pred to one in true
            particles_from = self.get_particles(entry, only_primaries=only_primaries)
            particles_to = self.get_true_particles(entry, only_primaries=only_primaries)
        elif mode == 'true_to_pred':
            # Match each true to one in pred
            particles_to = self.get_particles(entry, only_primaries=only_primaries)
            particles_from = self.get_true_particles(entry, only_primaries=only_primaries)
        else:
            raise ValueError("Mode {} is not valid. For matching each"\
                " prediction to truth, use 'pred_to_true' (and vice versa).".format(mode))
        all_kwargs = {"min_overlap": self.min_overlap_count, "overlap_mode": self.overlap_mode, **kwargs}
        if matching_mode == 'one_way':
            matched_pairs, counts = match_particles_fn(particles_from, particles_to,
                                                    **all_kwargs)
        elif matching_mode == 'optimal':
            matched_pairs, counts = match_particles_optimal(particles_from, particles_to,
                                                        **all_kwargs)
        else:
            raise ValueError
        if return_counts:
            return matched_pairs, counts
        else:
            return matched_pairs

    
    def match_interactions(self, entry, mode='pred_to_true',
                           drop_nonprimary_particles=True,
                           match_particles=True,
                           return_counts=False,
                           compute_vertex=True,
                           vertex_mode='all',
                           matching_mode='one_way',
                           tag_pi0=False,
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

        all_matches, all_counts = [], []
        if mode == 'pred_to_true':
            ints_from = self.get_interactions(entry, 
                                                drop_nonprimary_particles=drop_nonprimary_particles, 
                                                compute_vertex=compute_vertex,
                                                vertex_mode=vertex_mode,
                                                tag_pi0=tag_pi0)
            ints_to = self.get_true_interactions(entry, 
                                                 drop_nonprimary_particles=drop_nonprimary_particles, 
                                                 compute_vertex=compute_vertex,
                                                 tag_pi0=tag_pi0)
        elif mode == 'true_to_pred':
            ints_to = self.get_interactions(entry, 
                                            drop_nonprimary_particles=drop_nonprimary_particles, 
                                            compute_vertex=compute_vertex,
                                            vertex_mode=vertex_mode,
                                            tag_pi0=tag_pi0)
            ints_from = self.get_true_interactions(entry, 
                                                   drop_nonprimary_particles=drop_nonprimary_particles, 
                                                   compute_vertex=compute_vertex,
                                                   tag_pi0=tag_pi0)
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
            return [], []
        if match_particles:
            for interactions in matched_interactions:
                domain, codomain = interactions
                domain_particles, codomain_particles = [], []
                if domain is not None:
                    domain_particles = domain.particles
                if codomain is not None:
                    codomain_particles = codomain.particles
                    # continue
                domain_particles = [p for p in domain_particles if p.points.shape[0] > 0]
                codomain_particles = [p for p in codomain_particles if p.points.shape[0] > 0]
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

        if return_counts:
            return matched_interactions, counts
        else:
            return matched_interactions