import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from . import Interaction, TruthParticle


class TruthInteraction(Interaction):
    """
    Analogous data structure for Interactions retrieved from true labels.
    """
    def __init__(self, *args, **kwargs):
        super(TruthInteraction, self).__init__(*args, **kwargs)
        self.match = []
        self._match_counts = {}
        self.depositions_MeV = []
        self.num_primaries = 0
        for p in self.particles:
            self.depositions_MeV.append(p.depositions_MeV)
            if p.is_primary: self.num_primaries += 1
        self.depositions_MeV = np.hstack(self.depositions_MeV)


    @property
    def particles(self):
        return list(self._particles.values())

    @particles.setter
    def particles(self, value):
        assert isinstance(value, OrderedDict)
        parts = {}
        for p in value.values():
            self.check_particle_input(p)
            # Clear match information since Interaction is rebuilt
            p.match = []
            p._match_counts = {}
            parts[p.id] = p
        self._particles = OrderedDict(sorted(parts.items(), key=lambda t: t[0]))
        self.update_info()

    @staticmethod
    def check_particle_input(x):
        assert isinstance(x, TruthParticle)

    def __str__(self):

        self.get_particles_summary()
        msg = "TruthInteraction {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "-----------------------------------------------\n".format(
            self.id, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __repr__(self):
        return "TruthInteraction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))

