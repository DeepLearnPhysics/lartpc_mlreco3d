from analysis.classes.particle import Particle
import numpy as np
import numba as nb


def compute_sum_deposited(particle : Particle):
    assert hasattr(particle, 'deposition')
    sum_E = particle.deposition.sum()
    return sum_E


def proton_energy_tabular(particle: Particle):
    assert particle.pid == 4  # Proton
    