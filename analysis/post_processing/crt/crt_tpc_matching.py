import numpy as np
from collections import defaultdict
from analysis.post_processing import post_processing
from mlreco.utils.globals import *
#from matcha.match_candidate import MatchCandidate

@post_processing(data_capture=['meta', 'index', 'crthits'], 
                 result_capture=['interactions'])
def run_crt_tpc_matching(data_dict, result_dict, 
                         crt_tpc_manager=None,
                         crthit_keys=[]):
    """
    Post processor for running CRT-TPC matching using matcha.
    
    Parameters
    ----------

    Returns
    -------
    update_dict: dict of list
        Dictionary of a list of length batch_size, where each entry in 
        the list is a mapping:
            interaction_id : (matcha.CRTHit, matcha.MatchCandidate)
        
    NOTE: This post-processor also modifies the list of Interactions
    in-place by adding the following attributes:
        interaction.crthit_matched: (bool)
            Indicator for whether the given interaction has a CRT-TPC match
        interaction.crthit_id: (list of ints)
            List of IDs for CRT hits that were matched to one or more tracks
    """
    print('Running CRT matching...')
    from matcha.match_candidate import MatchCandidate

    crthits = {}
    assert len(crthit_keys) > 0
    for key in crthit_keys:
        crthits[key] = data_dict[key]
    
    interactions = result_dict['interactions']
    entry        = data_dict['index']
    
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

    # update_dict = defaultdict(list)

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
    print('Done CRT matching.')
    return {}




















