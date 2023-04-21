import numpy as np
import pandas as pd

from typing import Counter, List, Union
from collections import OrderedDict, Counter
from . import Particle
from mlreco.utils.globals import PID_LABELS


class Interaction:
    """
    Data structure for managing interaction-level
    full chain output information.

    Attributes
    ----------
    id : int, default -1
        Unique ID (Interaction ID) of this interaction.
    particle_ids : np.ndarray, default np.array([])
        List of Particle IDs that make up this interaction
    num_particles: int, default 0
        Total number of particles in this interaction
    num_primaries: int, default 0
        Total number of primary particles in this interaction
    nu_id : int, default -1
        ID of the particle's parent neutrino
    volume_id : int, default -1
        ID of the detector volume the interaction lives in
    image_id : int, default -1
        ID of the image the interaction lives in
    index : np.ndarray, default np.array([])
        (N) IDs of voxels that correspondn to the particle within the image coordinate tensor that
    points : np.dnarray, default np.array([], shape=(0,3))
        (N,3) Set of voxel coordinates that make up this interaction in the input tensor
    vertex : np.ndarray, optional
        3D coordinates of the predicted interaction vertex
        in reconstruction (used for debugging)
    """
    def __init__(self,
                 interaction_id: int = -1,
                 particles: List[Particle] = None,
                 nu_id: int = -1,
                 volume_id: int = -1,
                 image_id: int = -1,
                 vertex: np.ndarray = -np.ones(3, dtype=np.float32),
                 is_neutrino: bool = False):

        # Initialize private attributes to be set by setter only
        self._particles   = None

        # Initialize attributes
        self.id           = interaction_id
        self.nu_id        = nu_id
        self.volume_id    = volume_id
        self.image_id     = image_id
        self.vertex       = vertex

        # Aggregate individual particle information 
        self.particle_ids  = np.empty(0, dtype=np.int64)
        self.num_particles = 0
        self.num_primaries = 0
        self.index         = np.empty(0, dtype=np.int64)
        self.points        = np.empty((0,3), dtype=np.float32)
        self.depositions   = np.empty(0, dtype=np.float32)
        self.particles     = particles
        self.size          = len(self.index)

        # Quantities to be set by the particle matcher
        self.match = np.empty(0, np.int64)
        self._match_counts = np.empty(0, np.float32)

    def check_particle_input(self, x):
        """
        Consistency check for particle interaction id and self.id
        """
        assert isinstance(x, Particle)
        assert x.interaction_id == self.id

    def update_info(self):
        """
        Method for updating basic interaction particle count information.
        """
        self.particle_ids = list(self._particles.keys())
        self.particle_counts = Counter({ PID_LABELS[i] : 0 for i in PID_LABELS.keys() })
        self.particle_counts.update([PID_LABELS[p.pid] for p in self._particles.values()])

        self.primary_particle_counts = Counter({ PID_LABELS[i] : 0 for i in PID_LABELS.keys() })
        self.primary_particle_counts.update([PID_LABELS[p.pid] for p in self._particles.values() if p.is_primary])
        if sum(self.primary_particle_counts.values()) > 0:
            self.is_valid = True
        else:
            self.is_valid = False

    @property
    def particles(self):
        return self._particles

    @particles.setter
    def particles(self, particles):
        '''
        <Particle> list getter/setter. The setter also sets
        the general interaction properties
        '''
        self._particles    = particles

        if particles is not None:
            id_list, index_list, points_list, depositions_list = [], [], [], []
            for p in particles:
                id_list.append(p.id)
                index_list.append(p.index)
                points_list.append(p.points)
                depositions_list.append(p.depositions)
                self.num_primaries += int(p.is_primary)

            self.particle_ids = np.array(id_list, dtype=np.int64)
            self.num_particles = len(particles)
            self.index = np.concatenate(index_list)
            self.points = np.vstack(points_list)
            self.depositions = np.concatenate(depositions_list)

        self._get_particles_summary(particles)

    def __getitem__(self, key):
        return self._particles[key]

    def __repr__(self):
        return "Interaction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))

    def __str__(self):
        msg = "Interaction {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "--------------------------------------------------------------------\n".format(
            self.id, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self._particles_summary

    def _get_particles_summary(self, particles):

        primary_str = {True: '*', False: '-'}
        self._particles_summary = ""
        if particles is None: return
        for p in sorted(particles, key=lambda x: x.is_primary, reverse=True):
            pmsg = "    {} Particle {}: PID = {}, Size = {}, Match = {} \n".format(
                primary_str[p.is_primary], p.id, PID_LABELS[p.pid], p.size, str(p.match))
            self._particles_summary += pmsg
