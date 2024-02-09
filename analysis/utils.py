import numpy as np
import pandas as pd
from functools import partial
import numba as nb
import sys

from scipy.stats import beta
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

class TopologySelector:
    
    _LABEL_DICT = {
        'num_primary_muons': '$\\mu$',
        'num_primary_protons': '$p$',
        'num_primary_electrons': '$e$',
        'num_primary_pions': '$\\pi^\\pm$',
        'num_primary_photons': '$\\gamma$'
    }
    
    _FLASHMATCH_BOUNDS = [0., 1.6]
    
    def __init__(self, df_t2r, df_r2t):
        self.df_t2r = df_t2r
        self.df_r2t = df_r2t
        
        self._truth_labels = []
        self._reco_labels  = []
        
        self.df_eff = pd.DataFrame()
        self.df_eff['Iteration'] = self.df_t2r['Iteration']
        self.df_eff['Index'] = self.df_t2r['Index']
        self.df_eff['Truth'] = np.full(self.df_t2r.shape[0], 'N/A')
        self.df_eff['Prediction'] = np.full(self.df_t2r.shape[0], 'N/A')
        self.df_pur = pd.DataFrame()
        self.df_pur['Iteration'] = self.df_r2t['Iteration']
        self.df_pur['Index'] = self.df_r2t['Index']
        self.df_pur['Truth'] = np.full(self.df_r2t.shape[0], 'N/A')
        self.df_pur['Prediction'] = np.full(self.df_r2t.shape[0], 'N/A')
        
    @property
    def truth_labels(self):
        return self._truth_labels
    
    @property
    def reco_labels(self):
        return self._reco_labels
    
    def _generate_mask(self, config):
        
        mask_eff = np.zeros(self.df_t2r.shape[0], dtype=bool)
        mask_pur = np.zeros(self.df_r2t.shape[0], dtype=bool)
        
        mask_eff[self.df_t2r.query(config).index.values] = True
        mask_pur[self.df_r2t.query(config).index.values] = True

        # for key, val in config.items():
        #     operator, value = val.split(' ')
        #     if operator == '==':
        #         mask_eff &= (self.df_t2r[key] == eval(value)).values
        #         mask_pur &= (self.df_r2t[key] == eval(value)).values
        #     elif operator == '>=':
        #         mask_eff &= (self.df_t2r[key] >= eval(value)).values
        #         mask_pur &= (self.df_r2t[key] >= eval(value)).values
        #     elif operator == '<=':
        #         mask_eff &= (self.df_t2r[key] <= eval(value)).values
        #         mask_pur &= (self.df_r2t[key] <= eval(value)).values
        #     elif operator == '!=':
        #         mask_eff &= (self.df_t2r[key] != eval(value)).values
        #         mask_pur &= (self.df_r2t[key] != eval(value)).values
        #     else:
        #         raise ValueError("Selection operator may only be one of '==', '<=', or '>='.")
        #     print(key, mask_eff.sum(), mask_pur.sum(), self.df_t2r[key].value_counts())
        return mask_eff, mask_pur
        
        
    def add_truth_definition(self, truth_config: dict, label=''):
        
        assert len(truth_config) > 0
        
        mask_eff, mask_pur = self._generate_mask(truth_config)
        
        # self.df_eff.loc[mask_eff, 'Truth'] = label
        # self.df_pur.loc[mask_pur, 'Truth'] = label
        self.df_eff[label] = mask_eff
        self.df_pur[label] = mask_pur
        self._truth_labels.append(label)
        
    def add_reco_definition(self, reco_config: dict, label='', fm=False):
        
        assert len(reco_config) > 0
        
        mask_eff, mask_pur = self._generate_mask(reco_config)
                
        if fm:
            self.df_t2r['reco_flash_within_window'] = (self.df_t2r['reco_flash_time'] > self._FLASHMATCH_BOUNDS[0]) \
                                                    & (self.df_t2r['reco_flash_time'] < self._FLASHMATCH_BOUNDS[1])
            self.df_r2t['reco_flash_within_window'] = (self.df_r2t['reco_flash_time'] > self._FLASHMATCH_BOUNDS[0]) \
                                                    & (self.df_r2t['reco_flash_time'] < self._FLASHMATCH_BOUNDS[1])
            mask_eff &= self.df_t2r['reco_flash_within_window'].values
            mask_pur &= self.df_r2t['reco_flash_within_window'].values

        # self.df_eff.loc[mask_eff, 'Prediction'] = label
        # self.df_pur.loc[mask_pur, 'Prediction'] = label
        self.df_eff[label] = mask_eff
        self.df_pur[label] = mask_pur
        self._reco_labels.append(label)
        
    def get_mask(self, label, m2='eff'):
        if m2 == 'eff':
            return self.df_eff[label]
        elif m2 == 'pur':
            return self.df_pur[label]
            
    def cast_labels(self, label_list, m1='truth'):
        for l in label_list:
            if m1 == 'truth':
                mask = self.get_mask(l, m2='eff')
                self.df_eff['Truth'][mask] = l
                mask = self.get_mask(l, m2='pur')
                self.df_pur['Truth'][mask] = l
            elif m1 == 'reco':
                mask = self.get_mask(l, m2='eff')
                self.df_eff['Prediction'][mask] = l
                mask = self.get_mask(l, m2='pur')
                self.df_pur['Prediction'][mask] = l
        
        
class InteractionTopologyManager:
    """Helper interface for applying cuts to particles / interactions.
    """
    
    _MAPPING = {
        0: 'photons',
        1: 'electrons',
        2: 'muons',
        3: 'pions',
        4: 'protons'
    }
    
    def __init__(self, 
                 df_intrs: pd.DataFrame,
                 df_parts: pd.DataFrame,
                 mode='reco'):
        """Initializer for TopologyManager class.

        Parameters
        ----------
        df_intrs : pd.DataFrame
            Interaction-level dataframe, where one row contains information of
            one interaction.
        df_parts : pd.DataFrame
            Particle-level dataframe, where one row contains information of
            one particle.
        mode : str, optional
            Flag for using truth/reco listed dataframes. For example, if the
            dataframe was created from a true to reco matched information dump,
            one must use mode='truth' to make sure that for each unique 
            interaction, there exists a unique row in <df_intrs>. 
        """
        self._df_intrs = df_intrs
        self._df_parts = df_parts
        # self._df_parts = df_parts.query(f'{mode}_particle_interaction_id >= 0')
        
        print(f"Total Number of Interactions = {df_intrs.shape[0]}")
        print(f"Total Number of Particles = {df_parts.shape[0]}")
        
        self.df = pd.merge(self._df_intrs, self._df_parts, 
                           left_on=['Index', 'file_index', f'{mode}_interaction_id'], 
                           right_on=['Index', 'file_index', f'{mode}_particle_interaction_id'])
        
        non_matches = df_parts.query(f'{mode}_particle_interaction_id < 0').shape[0]
        
        if (self.df.shape[0] != self.df_parts.shape[0]):
            print(f"There are {non_matches} {mode}_particles that do not have any match.")
        
        self.df_altered = False
        self._mode = mode
        
    @property
    def df_intrs(self):
        return self._df_intrs
    
    @property
    def df_parts(self):
        return self._df_parts
    
    @property
    def mode(self):
        return self._mode
    
    def apply_particle_cut(self, query: str, pid: int = None):
        """Apply a cut globally or on a specific type of particle using a
        text query (similar to pandas df.query)

        Parameters
        ----------
        query : str
            Text description of the cut, simliar to pandas df.query
            ex) apply_particle_cut("true_energy_deposit > 40", pid=4) will
            keep all protons that have a true_energy_deposit greater than 40 MeV.
        pid : int, optional
            Integer ID for particle type. If pid is not provided, the cut
            specified by <query> will be applied to all particles. If pid is
            provided, the cut will only be applied to particles with th given
            pid. 
        """
        
        if pid is None:
            # Global cuts
            self.df = self.df.query(query)
        
        if pid is not None:
            self.df = self.df.query(f'not ({self._mode}_particle_type == {pid}) or ({query})')
            
            alive = self.df.query(f'{self._mode}_particle_type == {pid}').shape[0]
            total = self.df_parts.query(f'{self._mode}_particle_type == {pid}').shape[0]
            perc = alive / total * 100
            print(f"{alive} / {total} ({perc:.2f}%) of all {self._MAPPING[pid]} (before any cut) will survive the cut.")
            
        perc = self.df.shape[0] / self.df_parts.shape[0] * 100
        print(f"{self.df.shape[0]} / {self.df_parts.shape[0]} ({perc:.2f}%) of all particles (before any cut) will survive the cut.")
            
        self.df_altered = True
        
        
    def _get_recount_topologies(self, df_all, pids=[0,1,2,3,4]):
        df_new = pd.DataFrame()
        df_new['Index'] = df_all['Index']
        df_new['file_index'] = df_all['file_index']
        df_new[f'{self.mode}_interaction_id'] = df_all[f'{self.mode}_interaction_id']
        
        print(f"Recounting interaction topologies for pids: {str(pids)}...")
        
        for pid in pids:
            df_new[f'{self.mode}_num_primary_{self._MAPPING[pid]}_new'] = \
                df_all[f'{self.mode}_particle_is_primary'].astype(int) \
                    * (df_all[f'{self.mode}_particle_type'] == pid).astype(int)
            
        df_recount = df_new.groupby(['Index', 'file_index', f'{self.mode}_interaction_id']).sum().astype(int).reset_index()
        return df_recount
    
    def update(self, pids=[0,1,2,3,4]):
        df_recount = self._get_recount_topologies(self.df, pids=pids)
        df_parts_mgd = pd.merge(self.df, 
                                df_recount, 
                                on=['Index', 'file_index', f'{self.mode}_interaction_id'],
                                how='left')
        df_intrs_mgd = pd.merge(self.df_intrs, 
                                df_recount, 
                                on=['Index', 'file_index', f'{self.mode}_interaction_id'],
                                how='left')
        
        for pid in pids:
            self.df_intrs[f'{self.mode}_num_primary_{self._MAPPING[pid]}'] \
                = df_intrs_mgd[f'{self.mode}_num_primary_{self._MAPPING[pid]}_new'].fillna(0).astype(int)
            self.df[f'true_num_primary_{self._MAPPING[pid]}'] \
                = df_parts_mgd[f'{self.mode}_num_primary_{self._MAPPING[pid]}_new'].fillna(0).astype(int)
        
        self.df_altered = False
                
    def update_interaction_df(self, df_other, pids=[0,1,2,3,4]):
        if self.df_altered:
            raise AssertionError("It looks like you applied a cut and did not update/recounted topologies. Try update() before doing anything else. ")
        
        df_small = self.df_intrs[['Index', 'file_index', f'{self.mode}_interaction_id']]
        for pid in pids:
            df_small[f'{self.mode}_num_primary_{self._MAPPING[pid]}_new'] = self.df_intrs[f'{self.mode}_num_primary_{self._MAPPING[pid]}']
        
        df_intrs_mgd = pd.merge(df_other,
                                df_small,
                                on=['Index', 'file_index', f'{self.mode}_interaction_id'],
                                how='left')

        for pid in pids:
            self.df_intrs[f'{self.mode}_num_primary_{self._MAPPING[pid]}'] \
                = df_intrs_mgd[f'{self.mode}_num_primary_{self._MAPPING[pid]}_new'].fillna(0).astype(int)
                

@nb.njit(cache=True)
def intrinsic_loss_nb_fn(interval : nb.float32[:], 
                         intrinsic_disc : nb.float32[:,:], 
                         pdf : nb.float32[:],
                         n):
    
    mesh_size = interval.shape[0]
    output = np.zeros(mesh_size)

    for i, eps in enumerate(interval):
        val, valid = 0.0, False
        for j in range(mesh_size-1):
            dx = interval[j+1] - interval[j]
            f1 = intrinsic_disc[i, j] * pdf[j]
            f2 = intrinsic_disc[i, j+1] * pdf[j+1]
            integrand = 0.5 * (f1 + f2) * dx
            if not np.isnan(integrand):
                valid = True
                val += integrand
        if valid:
            output[i] = val
        else:
            output[i] = np.inf
            
    return output * n

def intrinsic_loss_approx(k, n, interval):

    a_n = (n + 4) / (4 * n + 10)
    mu = np.sqrt( (k + a_n) / (n + 2 * a_n) )
    
    output = 0.5 + 2 * n * np.power(np.arcsin(np.sqrt(interval)) - np.arcsin(mu), 2)
    
    return output


class BayesianBinomialEstimator:
    '''
    Helper class for estimating selection uncertainties
    (i.e., uncertainties on efficiency/purity) using Bayesian inference.
    '''
    
    _LARGE_N_CRITERION = 20
    
    def __init__(self, k, n, mesh_size=2000, mode=None):
        self.index = np.arange(mesh_size)
        self.interval = np.linspace(0, 1, mesh_size)
        self.pdf = np.zeros(mesh_size)
        self.cdf = np.zeros(mesh_size)
        self.intrinsic_disc = np.zeros((mesh_size, mesh_size))
        self.intrinsic_loss = np.zeros(mesh_size)
        
        self.k = k
        self.n = n
        self.mesh_size = mesh_size
        
        self.pdf_fn = self.reference_posterior(self.k, self.n)
        self.cdf_fn = self.reference_cumulative(self.k, self.n)
        
        self.out = {'eps': None, 'intrinsic_loss': None}
        
        if mode is None:
            if n < self._LARGE_N_CRITERION:
                self.mode = 'brute_force'
            else:
                self.mode = 'arcsin'
        else:
            self.mode = mode
            
        self.precompute_functions()
        self.compute_intrinsic_loss()
        
    def kl_divergence(self, p, q):
        return q * np.log(q / p) + (1 - q) * np.log((1-q) / (1-p))

    def intrinsic_discrepancy(self, p, q, n=1):
        return np.minimum(self.kl_divergence(p,q), self.kl_divergence(q,p))
    
    @staticmethod
    def reference_posterior(k, n):
        return partial(beta.pdf, a=(k+0.5), b=(n-k+0.5))

    @staticmethod
    def reference_cumulative(k, n):
        return partial(beta.cdf, a=(k+0.5), b=(n-k+0.5))
        
    def precompute_functions(self):

        for i, x1 in enumerate(self.interval):
            self.intrinsic_disc[i] = self.intrinsic_discrepancy(x1, self.interval)

        for i, x in enumerate(self.interval):
            self.pdf[i] = self.pdf_fn(x)
            self.cdf[i] = self.cdf_fn(x)
            
    def compute_intrinsic_loss(self):
        
        if self.mode == 'brute_force':
            output = intrinsic_loss_nb_fn(self.interval, 
                                          self.intrinsic_disc, 
                                          self.pdf,
                                          self.n)
        elif self.mode == 'arcsin':
            output = intrinsic_loss_approx(self.k,
                                           self.n, 
                                           self.interval)
        min_index = np.argmin(output)
        centroids = 0.5 * (self.interval[1:] + self.interval[:-1])
        
        self.intrinsic_loss = output
        self.out['eps'] = centroids[min_index]
        self.out['intrinsic_loss'] = output[min_index]
        self.out['index'] = min_index
        self.out['mean'] = self.k / self.n
        
    def compute_q_credible_interval(self, q=0.683, tol=1e-3, 
                                    mode='length', estimator='mean'):
        
        self.q_intvs = []
        
        # Collect possible q-intervals
        for i in range(0, self.out['index']+1):
            for j in range(self.out['index'], self.mesh_size):
                prob = self.cdf[j] - self.cdf[i]
                if np.abs(prob - q) < tol:
                    loss = self.intrinsic_loss[i:j].sum()
                    qintv = {
                        'lb': self.interval[i],
                        'lb_index': i,
                        'ub': self.interval[j],
                        'ub_index': j,
                        'length': self.interval[j] - self.interval[i],
                        'q': prob,
                        'loss': loss,
                        'estimator': self.out[estimator]
                    }
                    self.q_intvs.append(qintv)
                    
        # Pick q-interval according to criterion
        min_val, min_idx = np.inf, 0
        for i, qintv in enumerate(self.q_intvs):
            if qintv[mode] < min_val:
                min_idx = i
                min_val = qintv[mode]
        if len(self.q_intvs) == 0:
            print("No q-credible interval was found.")
            index = self.out['index']
            null_result = {
                'lb': self.interval[index-1],
                'lb_index': index-1,
                'ub': self.interval[index+1],
                'ub_index': index+1,
                'length': self.interval[index+1] - self.interval[index-1],
                'q': self.cdf[index+1] - self.cdf[index-1],
                'loss': 0.0,
                'estimator': self.out['eps']
            }
            return null_result
        return self.q_intvs[min_idx]


def twopoint_iou(reco_pair, true_pair):
    
    reco_1, reco_2 = reco_pair
    true_1, true_2 = true_pair
    
    if reco_1 is None or true_1 is None:
        iou_1 = 0
    else:
        cap_1 = np.intersect1d(true_1.index, reco_1.index)
        cup_1 = np.union1d(true_1.index, reco_1.index)
        iou_1 = float(cap_1.shape[0]) / float(cup_1.shape[0])
        
    if reco_2 is None or true_2 is None:
        iou_2 = 0
    else:
        cap_2 = np.intersect1d(true_2.index, reco_2.index)
        cup_2 = np.union1d(true_2.index, reco_2.index)
        iou_2 = float(cap_2.shape[0]) / float(cup_2.shape[0])
    
    mean_iou = (iou_1 + iou_2) / 2.0
    max_iou = max(iou_1, iou_2)
    
    return mean_iou, max_iou


# From https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
# Author: @sergeyprokudin

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist