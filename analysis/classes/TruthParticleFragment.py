import numpy as np

from typing import Counter, List, Union
from . import ParticleFragment


class TruthParticleFragment(ParticleFragment):
    """
    Data structure mirroring <ParticleFragment>, reserved for true fragments
    derived from true labels / true MC information.

    See <ParticleFragment> documentation for shared attributes.
    Below are attributes exclusive to TruthInteraction

    Attributes
    ----------
    depositions_MeV : np.ndarray, default np.array([])
        Similar as `depositions`, i.e. using adapted true labels.
        Using true MeV energy deposits instead of rescaled ADC units.
    """

    def __init__(self, 
                 *args, 
                 depositions_MeV: np.ndarray = np.empty(0, dtype=np.float32),
                 **kwargs):
        super(TruthParticleFragment, self).__init__(*args, **kwargs)
        self.depositions_MeV = depositions_MeV

    def __repr__(self):
        msg = super(TruthParticleFragment, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthParticleFragment, self).__str__()
        return 'Truth'+msg
