from abc import ABC, abstractmethod
from collections import OrderedDict
import os

from mlreco.iotools.readers import HDF5Reader

class DataProductLoader(ABC):
    
    def __init__(self):
        self._data = OrderedDict()
        self._result = OrderedDict()
        self._file_keys = []
        
    def register_file(self, path):
        if os.path.exists(path):
            self._file_keys.append(path)
        else:
            raise FileNotFoundError
        
    def load_matches(self):
        raise NotImplementedError
    
    def load_by_key(self):
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