import numpy as np
import pandas as pd

from typing import Counter, List, Union
from . import ParticleFragment


class TruthParticleFragment(ParticleFragment):

    def __init__(self, *args, depositions_MeV=None, **kwargs):
        super(TruthParticleFragment, self).__init__(*args, **kwargs)
        self.depositions_MeV = depositions_MeV

    def __repr__(self):
        fmt = "TruthParticleFragment( Image ID={:<3} | Fragment ID={:<3} | Semantic_type: {:<15}"\
                " | Group ID: {:<3} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2})"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type] if self.semantic_type in self.semantic_keys else "None",
                         self.group_id,
                         self.is_primary,
                         self.interaction_id,
                         self.points.shape[0],
                         self.volume)
        return msg
