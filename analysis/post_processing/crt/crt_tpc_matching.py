from analysis.post_processing import PostProcessor
from matcha.match_candidate import MatchCandidate


class CRTTPCMatchingProcessor(PostProcessor):
    '''
    Associates TPC interactions with optical flashes.
    '''
    name = 'run_crt_tpc_matching'
    data_cap = ['crthits']
    result_cap = ['interactions'] # TODO: Should be done at particle level

    def __init__(self,
                 crthit_keys,
                 **kwargs):
        '''
        Post processor for running CRT-TPC matching using matcha.
        
        Parameters
        ----------
        crthit_keys : List[str]
            List of keys that provide the CRT information in the data dictionary
        **kwargs : dict
            Keyword arguments to pass to the CRT-TPC matching algorithm
        '''
        # Store the relevant attributes
        self.crthit_keys = crthit_keys

    def process(self, data_dict, result_dict):
        '''
        Find [interaction, flash] pairs

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary

        Notes
        -----
        This post-processor also modifies the list of Interactions
        in-place by adding the following attributes:
            interaction.crthit_matched: (bool)
                Indicator for whether the given interaction has a CRT-TPC match
            interaction.crthit_id: (list of ints)
                List of IDs for CRT hits that were matched to one or more tracks
        '''
        crthits = {}
        assert len(self.crthit_keys) > 0
        for key in self.crthit_keys:
            crthits[key] = data_dict[key]
        
        interactions = result_dict['interactions']
        
        crt_tpc_matches = crt_tpc_manager.get_crt_tpc_matches(int(entry), 
                                                              interactions,
                                                              crthits,
                                                              use_true_tpc_objects=False,
                                                              restrict_interactions=[])

        assert all(isinstance(item, MatchCandidate) for item in crt_tpc_matches)

        # crt_tpc_matches is a list of matcha.MatchCandidates. Each MatchCandidate
        # contains a Track and CRTHit instance. The Track class contains the 
        # interaction_id.
        #matched_interaction_ids = [int_id for int_id in crt_tpc_matches.track.interaction_id]
        #matched_interaction_ids = []
        #for match in crt_tpc_matches:
        #    matched_interaction_ids.append(match.track.interaction_id)
        #
        #matched_interactions = [i for i in interactions 
        #                        if i.id in matched_interaction_ids]


        for match in crt_tpc_matches:
            matched_track = match.track
            # To modify the interaction in place, we need to find it in the interactions list
            matched_interaction = None
            for interaction in interactions:
                if matched_track.interaction_id == interaction.id:
                    matched_interaction = interaction
                    break
            matched_crthit = match.crthit
            # Sanity check
            if matched_interaction is None: continue
            matched_interaction.crthit_matched = True
            matched_interaction.crthit_matched_particle_id = matched_track.id
            matched_interaction.crthit_id = matched_crthit.id

            # update_dict['interactions'].append(matched_interaction)
        # update_dict['crt_tpc_matches'].append(crt_tpc_dict)

        return {}, {}
