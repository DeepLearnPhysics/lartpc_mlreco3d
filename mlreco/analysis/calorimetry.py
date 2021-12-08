from mlreco.analysis.particle import Particle
import numpy as np
import numba as nb


def compute_sum_deposited(particle : Particle):
    assert hasattr(particle, 'deposition')
    sum_E = particle.deposition.sum()
    return sum_E