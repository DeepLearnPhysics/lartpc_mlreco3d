from analysis.classes.particle import Particle
import numpy as np
import numba as nb


def compute_sum_deposited(particle : Particle):
    assert hasattr(particle, 'deposition')
    sum_E = particle.deposition.sum()
    return sum_E

# TODO:
# def proton_energy_tabular(particle: Particle):
#     assert particle.pid == 4  # Proton
#     x, y = particle.endpoints[0], particle.endpoints[1]
#     l = np.sqrt(np.power(x - y, 2).sum())

# def multiple_coulomb_scattering(particle: Particle):
#     assert particle.pid == 2  # Muon
#     pass