import numpy as np
import pandas as pd

from typing import Counter, List, Union
from collections import OrderedDict, Counter
from . import Particle


class Interaction:
    """
    Data structure for managing interaction-level
    full chain output information.

    Attributes
    ----------
    id: int
        Unique ID (Interaction ID) of this interaction.
    particles: List[Particle]
        List of <Particle> objects that belong to this Interaction.
    vertex: (1,3) np.array (Optional)
        3D coordinates of the predicted interaction vertex
    nu_id: int (Optional, TODO)
        Label indicating whether this interaction is a neutrino interaction
        WARNING: The nu_id label is most likely unreliable. Don't use this
        in reconstruction (used for debugging)
    num_particles: int
        total number of particles in this interaction.
    """
    def __init__(self, interaction_id: int, particles : OrderedDict, vertex=None, nu_id=-1, volume=0):
        self.id = interaction_id
        self.pid_keys = {
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }
        self.particles = particles
        self.match = []
        self._match_counts = {}
        # Voxel indices of an interaction is defined by the union of
        # constituent particle voxel indices
        self.voxel_indices = []
        self.points = []
        self.depositions = []
        for p in self.particles:
            self.voxel_indices.append(p.voxel_indices)
            self.points.append(p.points)
            self.depositions.append(p.depositions)
            assert p.interaction_id == interaction_id
        self.voxel_indices = np.hstack(self.voxel_indices)
        self.points = np.concatenate(self.points, axis=0)
        self.depositions = np.hstack(self.depositions)

        self.size = self.voxel_indices.shape[0]
        self.num_particles = len(self.particles)

        self.get_particles_summary()

        self.vertex = vertex
        self.vertex_candidate_count = -1
        if self.vertex is None:
            self.vertex = np.array([-1, -1, -1])

        self.nu_id = nu_id
        self.volume = volume


    @property
    def particles(self):
        return list(self._particles.values())

    def check_particle_input(self, x):
        assert isinstance(x, Particle)
        assert x.interaction_id == self.id

    def update_info(self):
        self.particle_ids = list(self._particles.keys())
        self.particle_counts = Counter({ self.pid_keys[i] : 0 for i in range(len(self.pid_keys))})
        self.particle_counts.update([self.pid_keys[p.pid] for p in self._particles.values()])

        self.primary_particle_counts = Counter({ self.pid_keys[i] : 0 for i in range(len(self.pid_keys))})
        self.primary_particle_counts.update([self.pid_keys[p.pid] for p in self._particles.values() if p.is_primary])
        if sum(self.primary_particle_counts.values()) > 0:
            self.is_valid = True
        else:
            self.is_valid = False


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


    def get_particles_summary(self):
        self.particles_summary = ""
        for p in self.particles:
            pmsg = "    - Particle {}: PID = {}, Size = {}, Match = {} \n".format(
                p.id, self.pid_keys[p.pid], p.points.shape[0], str(p.match))
            self.particles_summary += pmsg


    def __getitem__(self, key):
        return self._particles[key]


    def __str__(self):

        self.get_particles_summary()
        msg = "Interaction {}, Valid: {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "--------------------------------------------------------------------\n".format(
            self.id, self.is_valid, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __repr__(self):
        return "Interaction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))

