import numpy as np
import pandas as pd

from typing import Counter, List, Union
from . import Particle


class TruthParticle(Particle):
    '''
    Data structure mirroring <Particle>, reserved for true particles
    derived from true labels / true MC information.

    Attributes
    ----------
    See <Particle> documentation for shared attributes.
    Below are attributes exclusive to TruthParticle

    asis: larcv.Particle C++ object (Optional)
        Raw larcv.Particle C++ object as retrived from parse_particles_asis.
    match: List[int]
        List of Particle IDs that match to this TruthParticle
    coords_noghost:
        Coordinates using true labels (not adapted to deghosting output)
    depositions_noghost:
        Depositions using true labels (not adapted to deghosting output), in MeV.
    depositions_MeV:
        Similar as `depositions`, i.e. using adapted true labels.
        Using true MeV energy deposits instead of rescaled ADC units.
    '''
    def __init__(self, *args, particle_asis=None, coords_noghost=None, depositions_noghost=None,
                depositions_MeV=None, **kwargs):
        super(TruthParticle, self).__init__(*args, **kwargs)
        self.asis = particle_asis
        self.match = []
        self._match_counts = {}
        self.coords_noghost = coords_noghost
        self.depositions_noghost = depositions_noghost
        self.depositions_MeV = depositions_MeV
        self.startpoint = None
        self.endpoint = None


    def __repr__(self):
        msg = "TruthParticle(image_id={}, id={}, pid={}, size={})".format(self.image_id, self.id, self.pid, self.size)
        return msg


    def __str__(self):
        fmt = "TruthParticle( Image ID={:<3} | Particle ID={:<3} | Semantic_type: {:<15}"\
                " | PID: {:<8} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2} )"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type] if self.semantic_type in self.semantic_keys else "None",
                         self.pid_keys[self.pid] if self.pid in self.pid_keys else "None",
                         self.is_primary,
                         self.interaction_id,
                         self.points.shape[0],
                         self.volume)
        return msg


    def is_contained(self, spatial_size):

        p = self.particle_asis
        check_contained = p.position().x() >= 0 and p.position().x() <= spatial_size \
            and p.position().y() >= 0 and p.position().y() <= spatial_size \
            and p.position().z() >= 0 and p.position().z() <= spatial_size \
            and p.end_position().x() >= 0 and p.end_position().x() <= spatial_size \
            and p.end_position().y() >= 0 and p.end_position().y() <= spatial_size \
            and p.end_position().z() >= 0 and p.end_position().z() <= spatial_size
        return check_contained

    def purity_efficiency(self, other_particle):
        overlap = len(np.intersect1d(self.voxel_indices, other_particle.voxel_indices))
        return {
            "purity": overlap / len(other_particle.voxel_indices),
            "efficiency": overlap / len(self.voxel_indices)
        }

