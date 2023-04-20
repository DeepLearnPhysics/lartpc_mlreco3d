import numpy as np
import pandas as pd
from collections import OrderedDict
from . import Interaction, TruthParticle


class TruthInteraction(Interaction):
    """
    Data structure mirroring <Interaction>, reserved for true interactions
    derived from true labels / true MC information.

    See <Interaction> documentation for shared attributes.
    Below are attributes exclusive to TruthInteraction

    Attributes
    ----------
    depositions_MeV : np.ndarray, default np.array([])
        Similar as `depositions`, i.e. using adapted true labels.
        Using true MeV energy deposits instead of rescaled ADC units.
    """

    def __init__(self,
                 interaction_id, 
                 particles,
                 **kwargs):
        super(TruthInteraction, self).__init__(interaction_id, particles, **kwargs)

        self.depositions_MeV = np.empty(0, dtype=np.float32)
        if particles is not None:
            depositions_MeV_list = []
            for p in particles:
                depositions_MeV_list.append(p.depositions_MeV)
            self.depositions_MeV = np.concatenate(depositions_MeV_list)

        # Neutrino-specific information to be filled elsewhere
        self.nu_interaction_type = -1
        self.nu_interaction_mode = -1
        self.nu_current_type = -1
        self.nu_energy_init = -1.

#    @property
#    def particles(self):
#        return list(self._particles.values())
#
#    @particles.setter
#    def particles(self, value):
#        assert isinstance(value, OrderedDict)
#        parts = {}
#        for p in value.values():
#            self.check_particle_input(p)
#            # Clear match information since Interaction is rebuilt
#            p.match = []
#            p._match_counts = {}
#            parts[p.id] = p
#        self._particles = OrderedDict(sorted(parts.items(), key=lambda t: t[0]))
#        self.update_info()

    @staticmethod
    def check_particle_input(x):
        assert isinstance(x, TruthParticle)

    def __repr__(self):
        msg = super(TruthInteraction, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthInteraction, self).__repr__()
        return 'Truth'+msg

