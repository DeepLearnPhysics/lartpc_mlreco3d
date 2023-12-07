import os
import pathlib
import numpy as np
import numba as nb
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.constants import fine_structure

from .globals import MUON_PID, PION_PID, KAON_PID, PROT_PID, \
        ELEC_MASS, MUON_MASS, LAR_DENSITY, LAR_Z, LAR_A, LAR_MEE, \
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


def csda_ke_lar(R, M, z = 1, T_max=1e6, epsrel=1e-3, epsabs=1e-3):
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
    T_max : float, default 1e6
        Maximum allowed kinetic energy
    epsrel : float, default 1e-3
        Relative error tolerance
    epsabs : float, default 1e-3
        Asbolute error tolerance

    Returns
    -------
    float
        CSDA kinetic energy in MeV
    '''
    func = lambda x: csda_range_lar(x, M, z, epsrel, epsabs) - R
    return brentq(func, 0., T_max, rtol=epsrel, xtol=epsabs)


def csda_range_lar(T0, M, z = 1, epsrel=1e-3, epsabs=1e-3):
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
        Impinging partile charge in multiples of electron charge
    epsrel : float, default 1e-3
        Relative error tolerance
    epsabs : float, default 1e-3
        Asbolute error tolerance

    Returns
    -------
    float
        CSDA range in cm
    '''
    if T0 <= 0.:
        return 0.

    return -quad(inv_bethe_bloch_lar, 0., T0,
            args=(M, z), epsrel=epsrel, epsabs=epsabs)[0]


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
        Impinging partile charge in multiples of electron charge
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
def bethe_bloch_lar(T, M, z = 1):
    '''
    Bethe-Bloch energy loss function for liquid argon

    https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf

    Corrections taken from https://pdg.lbl.gov/2023/AtomicNuclearProperties/adndt.pdf

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

    # Compute the density effects
    delta = delta_lar(bg)

    # Compute the low energy corrections
    le_corr = 0. # le_corr_lar(beta, z)

    # Compute the Bremsstrahlung correction
    del_dedx = - K * fine_structure * (LAR_Z/LAR_A) / (4 * np.pi) * \
            (np.log(2 * gamma) - 1./3*(np.log(2 * W/ELEC_MASS))) * \
            np.log(2 * W/ELEC_MASS)**2

    # Compute the muon-specific spin correction
    spin_corr_muon = (1./8) * (W/gamma/M)**2 * (M==MUON_MASS)

    return F * (0.5*np.log((2 * ELEC_MASS * bg**2 * W)/ LAR_MEE**2) \
            - beta**2 - 0.5 * delta + le_corr + spin_corr_muon) + del_dedx


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
def delta_lar(bg):
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


#@nb.njit(cache=True) # Find an alternative to scipy's digamma to support njit
def le_corr_lar(beta, z = 1):
    '''
    Low energy corrections to the Bethe-Bloch formula

    Parameters
    ----------
    beta : float
        Lorentz beta (v/c)
    z : int, default 1
        Impinging partile charge in multiples of electron charge

    Returns
    -------
    float
        Low energy correction to the energy loss function
    '''
    # Shell corrections (tabulated, ignored for now)
    C = 0.

    # Barkas Correction (tabulated, ignored for now)
    L1 = 0.

    # Bloch Correction
    y = fine_structure*z/beta
    L2 = -abs(y) - np.real(digamma(1 + y*1j))

    # Low energy correction
    return -C/LAR_Z + z * L1 + z**2 * L2
