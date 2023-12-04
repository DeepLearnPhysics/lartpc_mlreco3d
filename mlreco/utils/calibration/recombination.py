import numpy as np

from mlreco.utils.globals import LAR_DENSITY, LAR_WION
from mlreco.utils.tracking import get_track_segment_dedxs


class RecombinationCalibrator:
    '''
    Applies a recombination correction factor to account for some of the
    ionization electrons recombining with the Argon ions, which is an effect
    that depends on the local rate of energy deposition and the angle of
    the deposition trail (track) w.r.t. to the drift field.

    Notes
    -----
    Must call the gain calibrator upstream, which converts the number of ADCs
    to a number of observed ionization electrons.
    '''
    name = 'recombination'

    def __init__(self,
                 efield,
                 drift_dir,
                 model = 'mbox',
                 A = 0.800,
                 k = 0.0486,
                 alpha = 0.906,
                 beta = 0.203,
                 R = 1.25,
                 tracking_mode = 'bin_pca',
                 **kwargs):
        '''
        Initialize the recombination model and its constants.

        Parameters
        ----------
        efield : float
            Electric field in kV/cm
        drift_dir : np.ndarray
            (3) three-vector indicating the direction of the drift field
        model : str, default 'mbox'
            Recombination model name (one of 'birks', 'mbox' or 'mbox_ell')
        A : float, default 0.800 (ICARUS CNGS fit)
            Birks model A parameter
        k : float, default 0.0486 (ICARUS CNGS fit)
            Birks model k parameter in (kV/cm)(gm/cm^2)/MeV
        alpha : float, default 0.906 (ICARUS fit)
            Modified box model alpha parameter
        beta : float, default 0.203 (ICARUS fit)
            Modified box model beta parameter in (kV/cm)(g/cm^2)/MeV
        R : float, default 1.25 (ICARUS fit)
            Modified box model ellipsoid correction R parameter
        **kwargs : dict, optional
            Additional arguments to pass to the tracking algorithm
        '''
        # Store the drift direction
        self.drift_dir = drift_dir

        # Initialize the model parameters
        self.use_angles = False
        if model == 'birks':
            self.model = 'birks'
            self.A = A
            self.k = k/efield/LAR_DENSITY # cm/MeV
        elif model in ['mbox', 'mbox_ell']:
            self.model = 'mbox'
            self.alpha = alpha
            self.beta = beta/efield/LAR_DENSITY # cm/MeV
            self.R = None
            if model == 'mbox_ell':
                self.use_angles = True
                self.R = R
        else:
            raise ValueError(f'Recombination model not recognized: {model}. ' \
                    'Must be one of birks, mbox or mbox_ell')

        # Store the tracking parameters
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = kwargs

    def birks(self, dedx):
        '''
        Birks equation to calculate electron quenching (higher local energy
        deposition are prone to more electron-ion recombination).

        Parameters
        ----------
        dedx : Union[float, np.ndarray]
            Value or array of values of dE/dx in MeV/cm

        Returns
        -------
        Union[float, np.ndarray]
           Quenching factors in electrons/MeV
        '''
        return self.A / (1. + self.k * dedx)

    def inv_birks(self, dqdx):
        '''
        Inverse Birks equation to undo electron quenching (higher local energy
        deposition are prone to more electron-ion recombination).

        Parameters
        ----------
        dqdx : Union[float, np.ndarray]
            Value or array of values of dQ/dx in electrons/cm

        Returns
        -------
        Union[float, np.ndarray]
            Inverse quenching factors in MeV/electrons
        '''
        return 1. / (self.A/LAR_WION - self.k * dqdx)

    def mbox(self, dedx, cosphi=None):
        '''
        Modified box model equation to calculate electron quenching (higher
        local energy deposition are prone to more electron-ion recombination)

        Parameters
        ----------
        dedx : Union[float, np.ndarray]
            Value or array of values of dE/dx in MeV/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Quenching factors in electrons/MeV
        '''
        Beta = self.beta
        if cosphi is not None:
            Beta /= np.sqrt(1 - (1 - 1./self.R**2)*cosphi**2)

        return np.log(self.alpha + Beta * dedx) / Beta / dedx

    def inv_mbox(self, dqdx, cosphi=None):
        '''
        Inverse modified box model equation to undo electron quenching (higher
        local energy deposition are prone to more electron-ion recombination)

        Parameters
        ----------
        dqdx : Union[float, np.ndarray]
            Value or array of values of dQ/dx in electrons/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Inverse quenching factors in MeV/electrons
        '''
        Beta = self.beta
        if cosphi is not None:
            Beta /= np.sqrt(1 - (1 - 1./self.R**2)*cosphi**2)

        return (np.exp(Beta * LAR_WION * dqdx) - self.alpha) / Beta / dqdx

    def recombination_factor(self, dedx, cosphi=None):
        '''
        Calls the predefined recombination models to evaluate
        the appropriate quenching factors.

        Parameters
        ----------
        dedx : Union[float, np.ndarray]
            Value or array of values of dEdx in MeV/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Quenching factors in electrons/MeV
        '''
        if self.model == 'birks':
            return self.birks(dedx)
        else:
            return self.mbox(dedx, cosphi)

    def inv_recombination_factor(self, dqdx, cosphi=None):
        '''
        Calls the predefined inverse recombination models to evaluate
        the appropriate correction factors.

        Parameters
        ----------
        dqdx : Union[float, np.ndarray]
            Value or array of values of dQ/dx in electrons/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Inverse quenching factors in MeV/electrons
        '''
        if self.model == 'birks':
            return self.inv_birks(dqdx)
        else:
            return self.inv_mbox(dqdx, cosphi)

    def process(self, values, points=None, dedx=None, track=False):
        '''
        Corrects for electron recombination.

        Parameters
        ----------
        values : np.ndarray
            (N) array of depositions in number of electrons
        points : np.ndarray, optional
            (N, 3) array of point coordinates associated with one particle.
            Only needed if `track` is set to `True`.
        dedx : float, optional
            If specified, use a flat value of dE/dx in MeV/cm to apply
            the recombination correction.
        track : bool, defaut `False`
            Whether the object is a track or not. If it is, the track gets
            segmented to evaluate local dE/dx and track angle.

        Returns
        -------
        np.ndarray
            (N) array of depositions in MeV
        '''
        # If the dE/dx value is fixed, use it to compute a flat recombination
        if not track:
            assert dedx is not None, \
                    'If the object is not tracked, must specify a flat dE/dx'
            recomb = self.recombination_factor(dedx)
            return values * LAR_WION / recomb

        # If the object is a track, segment the track use each segment to
        # compute a local dQ/dx (+ angle w.r.t. to the drift direction, if
        # requested) and assign a correction for all points in the segment.
        assert points is not None, \
                'Cannot track the object without point coordinates'
        seg_dqdxs, _, seg_clusts, seg_dirs, _ = get_track_segment_dedxs(points,
                values, method=self.tracking_mode, **self.tracking_kwargs)

        corr_values = np.empty(len(values), dtype=values.dtype)
        for i, c in enumerate(seg_clusts):
            if not self.use_angles:
                corr = self.inv_recombination_factor(seg_dqdxs[i])
            else:
                seg_cosphi = np.abs(np.dot(seg_dirs[i], self.drift_dir))
                corr = self.inv_recombination_factor(seg_dqdxs[i], seg_cosphi)

            corr_values[c] = corr * values[c]

        return corr_values
