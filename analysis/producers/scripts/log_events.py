from collections import OrderedDict

import numpy as np
from analysis.producers.decorator import write_to
from analysis.classes.evaluator import FullChainEvaluator
from analysis.classes.TruthInteraction import TruthInteraction
from analysis.classes.Interaction import Interaction
from analysis.producers.logger import ParticleLogger, InteractionLogger
from pprint import pprint
import copy

@write_to(['events'])
def event_data(data_blob, res, **kwargs):
    """_summary_

    Parameters
    ----------
    data_blob : _type_
        _description_
    res : _type_
        _description_
    """
    
    output = []
    
    events_dict = OrderedDict({
        'Index': -1,
        'num_reco_particles': -1,
        'num_truth_particles': -1,
        'num_reco_primaries': -1,
        'num_truth_primaries': -1,
        'num_reco_voxels': -1,
        'num_truth_voxels': -1,
        'num_truth_showers': -1,
        'num_reco_showers': -1,
        'num_truth_primary_showers': -1,
        'num_reco_primary_showers': -1
    })
    
    image_idxs = data_blob['index']
    
    for idx, index in enumerate(image_idxs):
        row_dict = copy.deepcopy(events_dict)
        row_dict['Index'] = index
        
        row_dict['num_reco_particles'] = len(res['particles'][idx])
        row_dict['num_truth_particles'] = len(res['truth_particles'][idx])
        
        row_dict['num_reco_primaries'] = len([p for p in res['particles'][idx] if p.is_primary])
        row_dict['num_truth_primaries'] = len([p for p in res['truth_particles'][idx] if p.is_primary])
        
        row_dict['num_reco_voxels'] = np.sum([p.points.shape for p in res['particles'][idx]])
        row_dict['num_truth_voxels'] = np.sum([p.points.shape for p in res['truth_particles'][idx]])
        
        row_dict['num_reco_showers'] = len([p for p in res['particles'][idx] if p.semantic_type == 0])
        row_dict['num_truth_showers'] = len([p for p in res['truth_particles'][idx] if p.semantic_type == 0])
        
        row_dict['num_reco_primary_showers'] = len([p for p in res['particles'][idx] \
            if p.semantic_type == 0 and p.is_primary])
        row_dict['num_truth_primary_showers'] = len([p for p in res['truth_particles'][idx] \
            if p.semantic_type == 0 and p.is_primary])
        
        row_dict['num_reco_electron_showers'] = len([p for p in res['particles'][idx] \
            if p.semantic_type == 0 and p.pid == 1])
        row_dict['num_truth_electron_showers'] = len([p for p in res['truth_particles'][idx] \
            if p.semantic_type == 0 and p.pid == 1])
        
        row_dict['num_reco_primary_electron_showers'] = len([p for p in res['particles'][idx] \
            if p.semantic_type == 0 and p.is_primary and p.pid == 1])
        row_dict['num_truth_primary_electron_showers'] = len([p for p in res['truth_particles'][idx] \
            if p.semantic_type == 0 and p.is_primary and p.pid == 1])
        
        output.append(row_dict)
        
    return [output]