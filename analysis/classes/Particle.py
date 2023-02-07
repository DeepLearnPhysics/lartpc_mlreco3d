import numpy as np
import pandas as pd

from typing import Counter, List, Union


class Particle:
    '''
    Data Structure for managing Particle-level
    full chain output information

    Attributes
    ----------
    id: int
        Unique ID of the particle
    points: (N, 3) np.array
        3D coordinates of the voxels that belong to this particle
    size: int
        Total number of voxels that belong to this particle
    depositions: (N, 1) np.array
        Array of energy deposition values for each voxel (rescaled, ADC units)
    voxel_indices: (N, ) np.array
        Numeric integer indices of voxel positions of this particle
        with respect to the total array of point in a single image.
    semantic_type: int
        Semantic type (shower fragment (0), track (1),
        michel (2), delta (3), lowE (4)) of this particle.
    pid: int
        PDG Type (Photon (0), Electron (1), Muon (2),
        Charged Pion (3), Proton (4)) of this particle.
    pid_conf: float
        Softmax probability score for the most likely pid prediction
    interaction_id: int
        Integer ID of the particle's parent interaction
    image_id: int
        ID of the image in which this particle resides in
    is_primary: bool
        Indicator whether this particle is a primary from an interaction.
    match: List[int]
        List of TruthParticle IDs for which this particle is matched to

    startpoint: (1,3) np.array
        (1, 3) array of particle's startpoint, if it could be assigned
    endpoint: (1,3) np.array
        (1, 3) array of particle's endpoint, if it could be assigned
    '''
    def __init__(self, coords, group_id, semantic_type, interaction_id,
                 pid, image_id, voxel_indices=None, depositions=None, volume=0, **kwargs):
        self.id = group_id
        self.points = coords
        self.size = coords.shape[0]
        self.depositions = depositions # In rescaled ADC
        self.voxel_indices = voxel_indices
        self.semantic_type = semantic_type
        self.pid = pid
        self.pid_conf = kwargs.get('pid_conf', None)
        self.interaction_id = interaction_id
        self.image_id = image_id
        self.is_primary = kwargs.get('is_primary', False)
        self.match = []
        self._match_counts = {}
#         self.fragments = fragment_ids
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }

        self.pid_keys = {
            -1: 'None',
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }

        self.sum_edep = np.sum(self.depositions)
        self.volume = volume
        self.startpoint = None
        self.endpoint = None

    def __repr__(self):
        msg = "Particle(image_id={}, id={}, pid={}, size={})".format(self.image_id, self.id, self.pid, self.size)
        return msg

    def __str__(self):
        fmt = "Particle( Image ID={:<3} | Particle ID={:<3} | Semantic_type: {:<15}"\
                " | PID: {:<8} | Primary: {:<2} | Score = {:.2f}% | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2} )"
        msg = fmt.format(self.image_id, self.id,
                         self.semantic_keys[self.semantic_type] if self.semantic_type in self.semantic_keys else "None",
                         self.pid_keys[self.pid] if self.pid in self.pid_keys else "None",
                         self.is_primary,
                         self.pid_conf * 100,
                         self.interaction_id,
                         self.points.shape[0],
                         self.volume)
        return msg

