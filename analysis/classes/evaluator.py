from typing import List
import numpy as np

from mlreco.utils.globals import VTX_COLS, INTER_COL, COORD_COLS, PDG_TO_PID

from analysis.classes import TruthParticleFragment, TruthParticle, Interaction
from analysis.classes.particle_utils import (match_particles_fn, 
                                             match_interactions_fn, 
                                             group_particles_to_interactions_fn, 
                                             match_interactions_optimal, 
                                             match_particles_optimal)
from analysis.algorithms.point_matching import *

from mlreco.utils.vertex import get_vertex

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


    def __init__(self, data_blob, result, evaluator_cfg={}, **kwargs):
        super(FullChainEvaluator, self).__init__(data_blob, result, evaluator_cfg, **kwargs)
        self.build_representations()
        self.michel_primary_ionization_only = evaluator_cfg.get('michel_primary_ionization_only', False)

    def build_representations(self):
        if 'Particles' not in self.result:
            self.result['Particles'] = self.particle_builder.build(self.data_blob, self.result, mode='reco')
        if 'TruthParticles' not in self.result:
            self.result['TruthParticles'] = self.particle_builder.build(self.data_blob, self.result, mode='truth')
        if 'Interactions' not in self.result:
            self.result['Interactions'] = self.interaction_builder.build(self.data_blob, self.result, mode='reco')
        if 'TruthInteractions' not in self.result:
            self.result['TruthInteractions'] = self.interaction_builder.build(self.data_blob, self.result, mode='truth')
        if 'ParticleFragments' not in self.result:
            self.result['ParticleFragments'] = self.fragment_builder.build(self.data_blob, self.result, mode='reco')
        if 'TruthParticleFragments' not in self.result:
            self.result['TruthParticleFragments'] = self.fragment_builder.build(self.data_blob, self.result, mode='truth')

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
            if p.pdg_code() not in PDG_TO_PID:
                continue
            mask = labels[:, 6].astype(int) == pid
            coords = labels[mask][:, 1:4]
            if coords.shape[0] < self.min_particle_voxel_count:
                particles_exclude.append(p.id())

        return set(particles_exclude)


    def get_true_fragments(self, entry) -> List[TruthParticleFragment]:
        '''
        Get list of <TruthParticleFragment> instances for given <entry> batch id.
        '''

        fragments = self.result['ParticleFragments'][entry]
        return fragments


    def get_true_particles(self, entry, 
                           only_primaries=True, 
                           volume=None) -> List[TruthParticle]:
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
        particles = self.result['TruthParticles'][entry]

        if only_primaries:
            out_particles_list = [p for p in particles if p.is_primary]
        if volume is not None:
            out_particles_list = [p for p in particles if p.volume == volume]

        return out_particles_list


    def get_true_interactions(self, entry) -> List[Interaction]:
        
        out = self.result['TruthInteractions'][entry]
        return out
    
    @staticmethod
    def match_parts_within_ints(int_matches):
        '''
        Given list of Tuple[(Truth)Interaction, (Truth)Interaction], 
        return list of particle matches Tuple[TruthParticle, Particle]. 

        If no match, (Truth)Particle is replaced with None.
        '''

        matched_particles, match_counts = [], []

        for m in int_matches:
            ia1, ia2 = m[0], m[1]
            num_parts_1, num_parts_2 = -1, -1
            if m[0] is not None:
                num_parts_1 = len(m[0].particles)
            if m[1] is not None:
                num_parts_2 = len(m[1].particles)
            if num_parts_1 <= num_parts_2:
                ia1, ia2 = m[0], m[1]
            else:
                ia1, ia2 = m[1], m[0]
                
            for p in ia2.particles:
                if len(p.match) == 0:
                    if type(p) is Particle:
                        matched_particles.append((None, p))
                        match_counts.append(-1)
                    else:
                        matched_particles.append((p, None))
                        match_counts.append(-1)
                for match_id in p.match:
                    if type(p) is Particle:
                        matched_particles.append((ia1[match_id], p))
                    else:
                        matched_particles.append((p, ia1[match_id]))
                    match_counts.append(p._match_counts[match_id])
        return matched_particles, np.array(match_counts)


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
            particles_from = self.get_particles(entry, 
                                                only_primaries=only_primaries)
            particles_to = self.get_true_particles(entry,
                                                   only_primaries=only_primaries)
        elif mode == 'true_to_pred':
            # Match each true to one in pred
            particles_to = self.get_particles(entry, 
                                              only_primaries=only_primaries)
            particles_from = self.get_true_particles(entry, 
                                                     only_primaries=only_primaries)
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

        all_matches, all_counts = [], []
        if mode == 'pred_to_true':
            ints_from = self.get_interactions(entry, 
                                              drop_nonprimary_particles=drop_nonprimary_particles)
            ints_to = self.get_true_interactions(entry)
        elif mode == 'true_to_pred':
            ints_to = self.get_interactions(entry, 
                                            drop_nonprimary_particles=drop_nonprimary_particles)
            ints_from = self.get_true_interactions(entry)
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
