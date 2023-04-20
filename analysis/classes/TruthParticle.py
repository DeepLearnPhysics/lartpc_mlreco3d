import numpy as np
import pandas as pd

from typing import Counter, List, Union
from . import Particle


class TruthParticle(Particle):
    '''
    Data structure mirroring <Particle>, reserved for true particles
    derived from true labels / true MC information.

    See <Particle> documentation for shared attributes.
    Below are attributes exclusive to TruthParticle.

    Attributes
    ----------
    depositions_MeV : np.ndarray
        Similar as `depositions`, i.e. using adapted true labels.
        Using true MeV energy deposits instead of rescaled ADC units.
    true_depositions : np.ndarray
        Rescaled charge depositions in the set of true voxels associated
        with the particle.
    true_depositions_MeV : np.ndarray
        MeV charge depositions in the set of true voxels associated
        with the particle.
    start_position : np.ndarray
        True start position of the particle
    end_position : np.ndarray
        True end position of the particle
    '''
    def __init__(self, 
                 *args, 
                 depositions_MeV=np.empty(0, dtype=np.float32),
                 true_index=np.empty(0, dtype=np.int64), 
                 true_depositions=np.empty(0, dtype=np.float32),
                 true_depositions_MeV=np.empty(0, dtype=np.float32),
                 particle_asis=None, 
                 **kwargs):
        super(TruthParticle, self).__init__(*args, **kwargs)

        # Initialize attributes
        self.depositions_MeV      = depositions_MeV
        self.true_index           = true_index
        self.true_depositions     = true_depositions
        self.true_depositions_MeV = true_depositions_MeV
        if particle_asis is not None:
            self.start_position = particle_asis.position()
            self.end_position   = particle_asis.end_position()

    def __repr__(self):
        msg = super(TruthParticle, self).__repr__()
        return 'Truth'+msg

    def __str__(self):
        msg = super(TruthParticle, self).__str__()
        return 'Truth'+msg

    def is_contained(self, spatial_size):

        check_contained = self.start_position.x() >= 0 and self.start_position.x() <= spatial_size \
                      and self.start_position.y() >= 0 and self.start_position.y() <= spatial_size \
                      and self.start_position.z() >= 0 and self.start_position.z() <= spatial_size \
                      and self.end_position.x()   >= 0 and self.end_position.x()   <= spatial_size \
                      and self.end_position.y()   >= 0 and self.end_position.y()   <= spatial_size \
                      and self.end_position.z()   >= 0 and self.end_position.z()   <= spatial_size
        return check_contained

    def purity_efficiency(self, other_particle):
        overlap = len(np.intersect1d(self.index, other_particle.index))
        return {
            "purity": overlap / len(other_particle.index),
            "efficiency": overlap / len(self.index)
        }

