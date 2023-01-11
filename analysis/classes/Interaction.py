import numpy as np
import pandas as pd

from typing import Counter, List, Union
from collections import defaultdict, Counter
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
    def __init__(self, interaction_id, particles, vertex=None, nu_id=-1, volume=0):
        self.id = interaction_id
        self.particles = particles
        self.match = []
        self._match_counts = {}
        self.check_validity()
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

        self.pid_keys = {
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }

        self.get_particles_summary()

        self.vertex = vertex
        if self.vertex is None:
            self.vertex = np.array([-1, -1, -1])

        self.nu_id = nu_id
        self.volume = volume

        self.particle_ids = [p.id for p in self.particles]
        self.particle_counts = Counter({ self.pid_keys[i] : 0 for i in range(len(self.pid_keys))})
        self.particle_counts.update([self.pid_keys[p.pid] for p in self.particles])

        self.primary_particle_counts = Counter({ self.pid_keys[i] : 0 for i in range(len(self.pid_keys))})
        self.primary_particle_counts.update([self.pid_keys[p.pid] for p in self.particles if p.is_primary])

        if sum(self.primary_particle_counts.values()) == 0:
            # print("Interaction {} has no primary particles!".format(self.id))
            self.is_valid = False
        else:
            self.is_valid = True

    def check_validity(self):
        for p in self.particles:
            assert isinstance(p, Particle)

    def get_particles_summary(self):
        self.particles_summary = ""
        self.particles = sorted(self.particles, key=lambda x: x.id)
        for p in self.particles:
            pmsg = "    - Particle {}: PID = {}, Size = {}, Match = {} \n".format(
                p.id, self.pid_keys[p.pid], p.points.shape[0], str(p.match))
            self.particles_summary += pmsg


    def __repr__(self):

        self.get_particles_summary()
        msg = "Interaction {}, Valid: {}, Vertex: x={:.2f}, y={:.2f}, z={:.2f}\n"\
            "--------------------------------------------------------------------\n".format(
            self.id, self.is_valid, self.vertex[0], self.vertex[1], self.vertex[2])
        return msg + self.particles_summary

    def __str__(self):
        return "Interaction(id={}, vertex={}, nu_id={}, Particles={})".format(
            self.id, str(self.vertex), self.nu_id, str(self.particle_ids))

