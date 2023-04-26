from typing import List
import numpy as np

from analysis.classes import TruthParticleFragment, TruthParticle, Interaction
from analysis.classes.matching import (match_particles_fn, 
                                             match_interactions_fn, 
                                             match_interactions_optimal, 
                                             match_particles_optimal)

from analysis.classes.predictor import FullChainPredictor
from mlreco.utils.globals import *
from analysis.classes.data import *


class FullChainEvaluator(FullChainPredictor):
    '''
    User Interface for full chain prediction and evaluation.
    
    The FullChainEvaluator shares the same methods as FullChainPredictor,
    but with additional methods to retrieve ground truth information and
    evaluate performance metrics.

    Usage:

        # <data>, <result> are full chain input/output dictionaries.
        evaluator = FullChainEvaluator(data, result)
        
        # Get labels
        pred_seg = evaluator.get_true_label(entry, mode='segmentation')
        # Get Particle instances
        matched_particles = evaluator.match_particles(entry)
        # Get Interaction instances
        matched_interactions = evaluator.match_interactions(entry)
    '''
    LABEL_TO_COLUMN = {
        'segment': SEG_COL,
        'charge': VALUE_COL,
        'fragment': CLUST_COL,
        'group': GROUP_COL,
        'interaction': INTER_COL,
        'pdg': PID_COL,
        'nu': NU_COL
    }


    def __init__(self, data_blob, result, evaluator_cfg={}, **kwargs):
        super(FullChainEvaluator, self).__init__(data_blob, result, evaluator_cfg, **kwargs)
        self.build_representations()
        self.michel_primary_ionization_only = evaluator_cfg.get('michel_primary_ionization_only', False)
        # For matching particles and interactions
        self.min_overlap_count        = evaluator_cfg.get('min_overlap_count', 0)
        # Idem, can be 'count' or 'iou'
        self.overlap_mode             = evaluator_cfg.get('overlap_mode', 'iou')
        if self.overlap_mode == 'iou':
            assert self.min_overlap_count <= 1 and self.min_overlap_count >= 0
        if self.overlap_mode == 'counts':
            assert self.min_overlap_count >= 0

    def build_representations(self, mode='all'):
        """
        Method using DataBuilders to construct high level data structures. 
        The constructed data structures are stored inside result dict. 
        
        Will not build data structures if the key corresponding to 
        the data structure class is already contained in the result dictionary.
        
        For example, if result['Particles'] exists and contains lists of
        reconstructed <Particle> instances, then methods inside the 
        Evaluator will use the already existing result['Particles'] 
        rather than building new lists from scratch. 
        
        Returns
        -------
        None (operation is in-place)
        """
        for key in self.builders:
            if key not in self.result and key in self.scope:
                self.result[key] = self.builders[key].build(self.data_blob, 
                                                            self.result, 
                                                            mode=mode)

    def get_true_label(self, entry, name, schema='cluster_label_adapted'):
        """
        Retrieve tensor in data blob, labelled with `schema`.

        Parameters
        ----------
        entry: int
        name: str
            Must be a predefined name within `['segment', 'fragment', 'group',
            'interaction', 'pdg', 'nu', 'charge']`.
        schema: str
            Key for dataset schema to retrieve the info from.
        volume: int, default None

        Returns
        -------
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


    def get_true_fragments(self, entry) -> List[TruthParticleFragment]:
        '''
        Get list of <TruthParticleFragment> instances for given batch id.
        
        Returns
        -------
        fragments: List[TruthParticleFragment]
            All track/shower fragments contained in image #<entry>.
        '''
        fragments = self.result['TruthParticleFragments'][entry]
        return fragments


    def get_true_particles(self, entry, 
                           only_primaries=True, 
                           volume=None) -> List[TruthParticle]:
        '''
        Get list of <TruthParticle> instances for given <entry> batch id.
        
        Can construct TruthParticles with no TruthParticle.points attribute
        (predicted nonghost coordinates), if the corresponding larcv::Particle
        object has nonzero true nonghost voxel depositions. 
        
        See TruthParticle for more information.
        
        Parameters
        ----------
        entry: int
            Image # (batch id) to fetch true particles.
        only_primaries: bool, optional
            If True, discards non-primary true particles from output.
        volume: int, optional
            Indicator for fetching TruthParticles only within a given cryostat. 
            Currently, 0 corresponds to east and 1 to west.
            
        Returns
        -------
        out_particles_list: List[TruthParticle]
            List of TruthParticles in image #<entry>
        '''
        out_particles_list = []
        particles = self.result['TruthParticles'][entry]

        if only_primaries:
            out_particles_list = [p for p in particles if p.is_primary]
        else:
            out_particles_list = [p for p in particles]
            
        if volume is not None:
            out_particles_list = [p for p in out_particles_list if p.volume_id == volume]
        return out_particles_list


    def get_true_interactions(self, entry) -> List[Interaction]:
        '''
        Get list of <TruthInteraction> instances for given <entry> batch id.
        
        Can construct TruthInteraction with no TruthInteraction.points
        (predicted nonghost coordinates), if all particles that compose the
        interaction has no predicted nonghost coordinates and nonzero
        true nonghost coordinates. 
        
        See TruthInteraction for more information.
        
        Parameters
        ----------
        entry: int
            Image # (batch id) to fetch true particles.
            
        Returns
        -------
        out: List[Interaction]
            List of TruthInteraction in image #<entry>
        '''
        out = self.result['TruthInteractions'][entry]
        return out
    
    
    @staticmethod
    def match_parts_within_ints(int_matches):
        '''
        Given list of matches Tuple[(Truth)Interaction, (Truth)Interaction], 
        return list of particle matches Tuple[TruthParticle, Particle]. 

        This means rather than matching all predicted particles againts
        all true particles, it has an additional constraint that only
        particles within a matched interaction pair can be considered
        for matching. 
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
        Method for matching reco and true particles by 3D voxel coordinate.
        
        Parameters
        ----------
        entry: int
            Image # (batch id)
        only_primaries: bool (default False)
            If true, non-primary particles will be discarded from beginning.
        mode: str (default "pred_to_true")
            Whether to match reco to true, or true to reco. This 
            affects the output if matching_mode="one_way".
        matching_mode: str (default "one_way")
            The algorithm used to establish matches. Currently there are
            only two options:
                - one_way: loops over true/reco particles, and chooses a
                reco/true particle with the highest overlap.
                - optimal: finds an optimal assignment between reco/true
                particles so that the sum of overlap metric (counts or IoU)
                is maximized. 
        return_counts: bool (default False)
            If True, returns the overlap metric (counts or IoU) value for
            each match. 
            
        Returns
        -------
        matched_pairs: List[Tuple[Particle, TruthParticle]]
        counts: np.ndarray
            overlap metric values corresponding to each matched pair. 
        '''
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
            raise ValueError(f"Particle matching mode {matching_mode} not suppored!")
        self._matched_particles = matched_pairs
        self._matched_particles_counts = counts
        if return_counts:
            return matched_pairs, counts
        else:
            return matched_pairs

    
    def match_interactions(self, entry, mode='pred_to_true',
                           drop_nonprimary_particles=False,
                           match_particles=True,
                           return_counts=False,
                           matching_mode='one_way',
                           **kwargs):
        """
        Method for matching reco and true interactions.
        
        Parameters
        ----------
        entry: int
            Image # (batch id)
        drop_nonprimary_particles: bool (default False)
            If true, non-primary particles will be discarded from beginning.
        match_particles: bool (default True)
            Option to match particles within matched interactions.
        matching_mode: str (default "one_way")
            The algorithm used to establish matches. Currently there are
            only two options:
                - one_way: loops over true/reco particles, and chooses a
                reco/true particle with the highest overlap.
                - optimal: finds an optimal assignment between reco/true
                particles so that the sum of overlap metric (counts or IoU)
                is maximized. 
        return_counts: bool (default False)
            If True, returns the overlap metric (counts or IoU) value for
            each match. 
            
        Returns
        -------
        matched_pairs: List[Tuple[Particle, TruthParticle]]
        counts: np.ndarray
            overlap metric values corresponding to each matched pair. 
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
                    matched_particles, _ = match_particles_fn(domain_particles, 
                                                              codomain_particles,
                                                              min_overlap=self.min_overlap_count,
                                                              overlap_mode=self.overlap_mode)
                elif matching_mode == 'optimal':
                    matched_particles, _ = match_particles_optimal(domain_particles, codomain_particles,
                                                                    min_overlap=self.min_overlap_count,
                                                                    overlap_mode=self.overlap_mode)
                else:
                    raise ValueError(f"Particle matching mode {matching_mode} is not supported!")

            pmatches, pcounts = self.match_parts_within_ints(matched_interactions)
            
            self._matched_particles = pmatches
            self._matched_particles_counts = pcounts
            
        self._matched_interactions = matched_interactions
        self._matched_interactions_counts = counts

        if return_counts:
            return matched_interactions, counts
        else:
            return matched_interactions
