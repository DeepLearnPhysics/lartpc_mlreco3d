import scipy
import numpy as np
import numba as nb

from .globals import LAR_X0
from .energy_loss import step_energy_loss_lar


def mcs_fit(theta, M, dx, z = 1):
    '''
    Finds the kinetic energy which best fits a set of scattering angles
    measured between successive segments along a particle track.

    Parameters
    ----------
    theta : np.ndarray
        (N) Vector of scattering angle at each step in radians
    M : float
        Particle mass in MeV/c^2
    dx : float
        Step length in cm
    z : int, default 1
       Impinging partile charge in multiples of electron charge
    '''
    # TODO: give a lower bound using CSDA (upper bound?)
    fit_min = scipy.optimize.minimize_scalar(mcs_nll_lar,
            args=(theta, M, dx, z), bounds=[10., 100000.])

    return fit_min.x


@nb.njit(cache=True)
def mcs_nll_lar(T0, theta, M, dx, z = 1):
    '''
    Computes the MCS negative log likelihood for a given list of segment angles
    and an initial momentum. This function checks the agreement between the
    scattering expection and the observed scattering at each step.

    Parameters
    ----------
    T0 : float
        Candidate particle kinetic energy in MeV
    theta : np.ndarray
        (N) Vector of scattering angle at each step in radians
    M : float
        Particle mass in MeV/c^2
    dx : float
        Step length in cm
    z : int, default 1
       Impinging partile charge in multiples of electron charge
    '''
    # Compute the kinetic energy at each step
    assert len(theta), 'Must provide angles to esimate the MCS loss'
    num_steps = len(theta + 1)
    ke_array  = step_energy_loss_lar(T0, M, dx, num_steps=num_steps)

    # If there are less steps than expected, T0 is too low
    if len(ke_array) < num_steps + 1:
        return np.inf

    # Convert the kinetic energy array to momenta
    mom_array = np.sqrt(ke_array**2 + 2 * M * ke_array)

    # Define each segment momentum as the geometric mean of its end points
    mom_steps = np.sqrt(mom_array[1:] * mom_array[:-1])

    # Get the expected scattering angle for each step
    theta0 = highland(mom_steps, M, dx, z, projected=False)

    # Compute the negative log likelihood, return
    nll = np.sum(0.5 * (theta/theta0)**2 + 2*np.log(theta0))

    return nll


@nb.njit(cache=True)
def highland(p, M, dx, z = 1, projected = False, X0 = LAR_X0):
    '''
    Highland scattering formula

    Parameters
    ----------
    p : float
       Momentum in MeV/c
    M : float
       Impinging particle mass in MeV/c^2
    dx : float
        Step length in cm
    z : int, default 1
       Impinging partile charge in multiples of electron charge
    projected : int, default `False`
       If `True`, returns the expectation in a projected 2D plane
    X0 : float, default LAR_X0
       Radiation length in the material of interest in cm

    Results
    -------
    float
        Expected scattering angle in radians
    '''
    # Highland formula
    beta = p / np.sqrt(p**2 + M**2)
    prefactor = np.sqrt(2)**(not projected) * 13.6 * z / beta / p

    return prefactor * np.sqrt(dx/LAR_X0) * \
            (1. + 0.038*np.log(z**2 * dx / LAR_X0 / beta**2))
