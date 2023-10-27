import numpy as np

from analysis.post_processing import PostProcessor

class MCSEnergyProcessor(PostProcessor):
    
    def __init__(self, **kwargs):
        self._min_hits_per_segment = kwargs.get('min_hits_per_segment', 2)
        self._min_num_segments = kwargs.get('min_num_segments', 3)
        self._segment_length = kwargs.get('segment_length', 14)
        self._num_eloss_steps = kwargs.get('num_eloss_steps', 10)
        self._p_min = kwargs.get('p_min', 0.01)
        self._p_max = kwargs.get('p_max', 7.50)
        self._p_step_coarse = kwargs.get('p_step_coarse', 0.01)
        self._p_step = kwargs.get('p_step', 0.01)
        
    @staticmethod   
    def energy_loss_landau(mass2, e2, x):
        Iinv2    = 1. / ((188 * 1e-6)**2)
        matConst = 1.4 * 18./40. # density * Z / A
        m_e      = 0.511 # electron mass, MeV
        kappa    = 0.307075
        j        = 0.200
        
        beta2   = (e2 - mass2) / e2
        gamma2  = 1. / (1. - beta2)
        epsilon = 0.5 * kappa * x * matConst / beta2
        
        return 0.001 * epsilon * ( np.log(2. * m_e * beta2 * gamma2 * epsilon * Iinv2) + j  - beta2 )
        
        
