from abc import ABC, abstractmethod
from collections import OrderedDict
import os


class DataProductLoader(ABC):
    
    def __init__(self):
        self._data = OrderedDict()
        self._result = OrderedDict()
        self._file_keys = []
        
    def load_matches(self):
        raise NotImplementedError
    
    @abstractmethod
    def _load_reco(self, entry):
        raise NotImplementedError
    
    @abstractmethod
    def _load_true(self, entry):
        raise NotImplementedError
    
    def load_image(self, entry: int, mode='reco'):
        """Load single image worth of entity blueprint from HDF5 
        and construct original data structure instance.

        Parameters
        ----------
        entry : int
            Image ID
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        mode : str, optional
            Whether to load reco or true entities, by default 'reco'

        Returns
        -------
        entities: List[Any]
            List of constructed entities from their HDF5 blueprints. 
        """
        if mode == 'truth':
            entities = self._load_truth(entry)
        elif mode == 'reco':
            entities = self._load_reco(entry)
        else:
            raise ValueError(f"Particle loader mode {mode} not supported!")
        
        return entities
    
    def load(self, mode='reco'):
        """Process all images in the current batch of HDF5 data and
        construct original data structures.
        
        Parameters
        ----------
        data: dict
            Data dictionary
        result: dict
            Result dictionary
        mode: str
            Indicator for building reconstructed vs true data formats.
            In other words, mode='reco' will produce <Particle> and
            <Interaction> data formats, while mode='truth' is reserved for
            <TruthParticle> and <TruthInteraction>
        """
        output = []
        num_batches = len(self._data['index'])
        for bidx in range(num_batches):
            entities = self.load_image(bidx, mode=mode)
            output.append(entities)
        return output
    
    
class ParticleLoader(DataProductLoader):
    
    def __init__(self):
        super(ParticleLoader, self).__init__()
        
    def _load_reco(self, entry, data: dict, result: dict):
        """Construct Particle objects from loading HDF5 blueprints.

        Parameters
        ----------
        entry : int
            Image ID
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        out : List[Particle]
            List of restored particle instances built from HDF5 blueprints.
        """
        if 'input_rescaled' in result:
            point_cloud = result['input_rescaled'][0]
        elif 'input_data' in data:
            point_cloud = data['input_data'][0]
        else:
            msg = "To build Particle objects from HDF5 data, need either "\
                "input_data inside data dictionary or input_rescaled inside"\
                " result dictionary."
            raise KeyError(msg)
        out  = []
        blueprints = result['particles'][0]
        for i, bp in enumerate(blueprints):
            mask = bp['index']
            prepared_bp = copy.deepcopy(bp)
            
            match = prepared_bp.pop('match', [])
            match_overlap = prepared_bp.pop('match_overlap', [])
            assert len(match) == len(match_overlap)
            
            prepared_bp.pop('depositions_sum', None)
            group_id = prepared_bp.pop('id', -1)
            prepared_bp['group_id'] = group_id
            prepared_bp.update({
                'points': point_cloud[mask][:, COORD_COLS],
                'depositions': point_cloud[mask][:, VALUE_COL],
            })
            particle = Particle(**prepared_bp)
            if len(match) > 0:
                particle.match_overlap = OrderedDict({
                    key : val for key, val in zip(match, match_overlap)})
            # assert particle.image_id == entry
            out.append(particle)
        
        return out
    
    
    def _load_true(self, entry, data, result):
        out = []
        true_nonghost = data['cluster_label'][0]
        pred_nonghost = result['cluster_label_adapted'][0]
        blueprints = result['truth_particles'][0]
        for i, bp in enumerate(blueprints):
            mask = bp['index']
            true_mask = bp['truth_index']
            
            prepared_bp = copy.deepcopy(bp)
            
            group_id = prepared_bp.pop('id', -1)
            prepared_bp['group_id'] = group_id
            prepared_bp.pop('depositions_sum', None)
            prepared_bp.update({
                'points': pred_nonghost[mask][:, COORD_COLS],
                'depositions': pred_nonghost[mask][:, VALUE_COL],
                'truth_points': true_nonghost[true_mask][:, COORD_COLS],
                'truth_depositions': true_nonghost[true_mask][:, VALUE_COL]
            })
            
            match = prepared_bp.pop('match', [])
            match_overlap = prepared_bp.pop('match_overlap', [])
            
            truth_particle = TruthParticle(**prepared_bp)
            if len(match) > 0:
                truth_particle.match_overlap = OrderedDict({
                    key : val for key, val in zip(match, match_overlap)})
            # assert truth_particle.image_id == entry
            assert truth_particle.truth_size > 0
            truth_particle.id = len(out)
            out.append(truth_particle)
            
        return out
    
    def load_matches(self, particles, truth_particles):
        part_dict = {}
        truth_dict = {}
        
        matches = OrderedDict()
        match_values = OrderedDict()
        
        for p in particles:
            part_dict[p.id] = p 
        for p in truth_particles:
            truth_dict[p.id] = p
        
        for p in truth_particles:
            match_ids = p.match
            mvals = p.match_overlap
            for i, mid in enumerate(match_ids):
                p_other = part_dict[mid]
                key = (p.id, p_other.id)
                pair = (p, p_other)
                if key not in matches:
                    matches[key] = pair
                    match_values[key] = mvals[i]
                else:
                    print(p, key)
                    raise ValueError
                    
        for p in particles:
            match_ids = p.match
            mvals = p.match_overlap
            for i, mid in enumerate(match_ids):
                p_other = truth_dict[mid]
                key = (p_other.id, p.id)
                pair = (p_other, p)
                if key not in matches:
                    matches[key] = pair
                    match_values[key] = mvals[i]
        
        return matches, match_values
