from analysis.classes.data import *


class ParticleAdaptor:
    
    def __init__(self, meta=None):
        self._meta = meta
        
    def cast(self, blueprint):
        pass
    
    def make_blueprint(self, particle: Particle):
        pass