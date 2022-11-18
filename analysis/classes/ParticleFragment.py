import numpy as np
import pandas as pd

from typing import Counter, List, Union
from . import Particle


class ParticleFragment(Particle):
    '''
    Data structure for managing fragment-level
    full chain output information

    Attributes
    ----------
    See <Particle> documentation for shared attributes.
    Below are attributes exclusive to ParticleFragment

    id: int
        fragment ID of this particle fragment (different from particle id)
    group_id: int
        Group ID (alias for Particle ID) for which this fragment belongs to.
    is_primary: bool
        If True, then this particle fragment corresponds to
        a primary ionization trajectory within the group of fragments that
        compose a particle.
    '''
    def __init__(self, coords, fragment_id, semantic_type, interaction_id,
                 group_id, image_id=0, voxel_indices=None,
                 depositions=None, volume=0, **kwargs):
        self.id = fragment_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions # In rescaled ADC
        self.voxel_indices = voxel_indices
        self.semantic_type = semantic_type
        self.group_id = group_id
        self.interaction_id = interaction_id
        self.image_id = image_id
        self.is_primary = kwargs.get('is_primary', False)
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }
        self.volume = volume

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        fmt = "ParticleFragment( Image ID={:<3} | Fragment ID={:<3} | Semantic_type: {:<15}"\
                " | Group ID: {:<3} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2})"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type] if self.semantic_type in self.semantic_keys else "None",
                         self.group_id,
                         self.is_primary,
                         self.interaction_id,
                         self.points.shape[0],
                         self.volume)
        return msg

