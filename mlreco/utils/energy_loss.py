import os
import pathlib
import numpy as np
import numba as nb
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import fsolve

from .globals import MUON_PID, PION_PID, KAON_PID, PROT_PID, \
        ELEC_MASS, LAR_DENSITY, LAR_Z, LAR_A, LAR_MEE, \
        LAR_a, LAR_k, LAR_x0, LAR_x1, LAR_Cbar, LAR_delta0


def csda_table_spline(particle_type, table_dir='csda_tables'):
    '''
    Interpolates a CSDA table to form a spline which maps
    a range to a kinematic energy estimate.

    Parameters
    ----------
    particle_type : int
        Particle type ID to construct splines. Maps are
        avaible for muons, pions, kaons and protons.
    table_dir : str, default 'csda_tables'
        Relative path to the CSDA range tables

    Returns
    -------
    callable
        Function mapping range (cm) to Kinetic E (MeV)
    '''
    # Check that the table for the requested PID exists
    path = pathlib.Path(__file__).parent
    suffix = 'E_liquid_argon'
    name_mapping = {MUON_PID: 'mu',
                    PION_PID: 'pi',
                    KAON_PID: 'ka',
                    PROT_PID: 'p'}
    if particle_type not in name_mapping.keys():
        raise ValueError('CSDA table for particle type ' \
                f'{particle_type} is not available')

    # Fetch the table and fit a spline
    pid = name_mapping[particle_type]
    file_name = os.path.join(path, table_dir, f'{pid}{suffix}')
    if os.path.isfile(f'{file_name}.txt'):
        path = f'{file_name}.txt'
    else:
        path = f'{file_name}_bethe.txt'

    tab = pd.read_csv(path, delimiter=' ', index_col=False)
    f = CubicSpline(tab['CSDARange'] / LAR_DENSITY, tab['T'])

    return f


def csda_ke_lar(R, M, z = 1, start = 1000):
    '''
    Numerically optimizes the kinetic energy necessary to observe the
    range of a particle that has been measured, under the CSDA.

    Parameters
    ----------
    R : float
        Range that the particle travelled through liquid argon in cm
    M : float
        Particle mass in MeV/c^2
    z : int, default 1
       Impinging partile charge in multiples of electron charge
    start : float, default 1000
        Starting estimate in MeV

    Returns
    -------
    float
        CSDA kinetic energy in MeV
    '''
    func = lambda x: csda_range_lar(x, M, z) - R
    return fsolve(func, start)[0]


def csda_range_lar(T0, M, z = 1):
    '''
    Numerically integrates the inverse Bethe-Bloch formula to find the
    CSDA range of a particle for a given initial kinetic energy.

    Parameters
    ----------
    T0 : float
        Initial kinetic energy in MeV
    M : float
        Particle mass in MeV/c^2
    z : int, default 1
       impinging partile charge in multiples of electron charge

    Returns
    -------
    float
        CSDA range in cm
    '''
    if T0 <= 0.:
        return 0.

    return -quad(inv_bethe_bloch_lar, 0., T0, args=(M, z))[0]


@nb.njit(cache=True)
def step_energy_loss_lar(T0, M, dx, z = 1, num_steps=None):
    '''
    Steps the initial energy of a particle down by pushing it through
    steps of dx of liquid argon. If `num_steps` is not specified, it
    will iterate until the particle kinetic energy reaches 0.

    Parameters
    ----------
    T0 : float
        Initial kinetic energy in MeV
    M : float
        Particle mass in MeV/c^2
    dx : float
        Step size in cm
    z : int, default 1
       impinging partile charge in multiples of electron charge
    num_steps : int, optional
       If specified, only step a maximum of `num_steps` times

    Returns
    -------
    np.array
        Array of kinetic energies at each step
    '''
    # Initialize the list
    assert T0 > 0., 'Must start with positive kinetic energy'
    ke_list = [T0]

    # Step down
    step = 0
    Ti = T0
    while Ti > 0 and (step is None or step < num_steps):
        step += 1
        Ti += dx * bethe_bloch_lar(Ti, M, z)
        if Ti > 0.:
            ke_list.append(Ti)
        else:
            ke_list.append(0.)
            break

    # Return
    return np.array(ke_list)


@nb.njit(cache=True)
def bethe_bloch_lar(T, M, z = 1):
    '''
    Bethe-Bloch energy loss function for liquid argon

    Parameters
    ----------
    T : float
       Kinetic energy in MeV
    M : float
       Impinging particle mass in MeV/c^2
    z : int, default 1
       Impinging partile charge in multiples of electron charge

    Returns
    -------
    float
       Value of the energy loss rate in liquid argon in MeV/cm
    '''
    # Constants
    K = 0.307075 # Bethe-Bloch constant [MeV/mol/cm^2]

    # Kinematics
    gamma = 1. + T/M
    beta = np.sqrt(1. - 1./gamma**2)
    bg = beta*gamma

    # Prefactor
    F = -K * z**2 * (LAR_Z/LAR_A) * LAR_DENSITY / beta**2

    # Compute the max energy transfer
    W = W_max(beta, gamma, M)

    return F * (0.5*np.log((2 * ELEC_MASS * bg**2 * W)/ LAR_MEE**2) \
            - beta**2 - 0.5 * delta(bg))


@nb.njit(cache=True)
def inv_bethe_bloch_lar(T, M, z = 1):
    '''
    Inverse Bethe-Bloch energy loss function for liquid argon

    Parameters
    ----------
    T : float
       Kinetic energy in MeV
    M : float
       Impinging particle mass in MeV/c^2
    z : int, default 1
       Impinging partile charge in multiples of electron charge

    Returns
    -------
    float
       Value of the inverse energy loss rate in liquid argon in MeV/cm
    '''
    return 1./bethe_bloch_lar(T, M, z)


@nb.njit(cache=True)
def W_max(beta, gamma, M):
    '''
    Maximum energy transfer in a single colision

    Parameters
    ----------
    beta : float
        Lorentz beta (v/c)
    gamma : float
        Lorentz gamma (1/sqrt(1-beta**2))
    M : float
        Mass of the impinging particle in MeV/c^2

    Returns
    -------
    float
        Maximum energy transferred in a single colision
    '''
    bg = beta*gamma
    return (2 * ELEC_MASS * bg**2) / \
            (1. + 2*gamma*ELEC_MASS/M + (ELEC_MASS/M)**2)


@nb.njit(cache=True)
def delta(bg):
    '''
    Density correction

    Parameters
    ----------
    bg : float
        Product of Lorentz beta and gamma (beta/sqrt(1-beta**2))

    Returns
    -------
    float
        Density correction to the Bethe-Bloch function
    '''
    x = np.log10(bg)
    if x < LAR_x0:
        return LAR_delta0 * 10**(2 * (x - LAR_x0))
    elif x >= LAR_x0 and x < LAR_x1:
        return 2 * np.log(10) * x - LAR_Cbar + LAR_a * (LAR_x1 - x)**LAR_k
    else:
        return 2 * np.log(10) * x - LAR_Cbar
