import numpy as np
import pandas as pd
from functools import partial
import numba as nb

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
        
        mask_eff = np.ones(self.df_t2r.shape[0], dtype=bool)
        mask_pur = np.ones(self.df_r2t.shape[0], dtype=bool)

        for key, val in config.items():
            operator, value = val.split(' ')
            if operator == '==':
                mask_eff &= (self.df_t2r[key] == eval(value)).values
                mask_pur &= (self.df_r2t[key] == eval(value)).values
            elif operator == '>=':
                mask_eff &= (self.df_t2r[key] >= eval(value)).values
                mask_pur &= (self.df_r2t[key] >= eval(value)).values
            elif operator == '<=':
                mask_eff &= (self.df_t2r[key] <= eval(value)).values
                mask_pur &= (self.df_r2t[key] <= eval(value)).values
            elif operator == '!=':
                mask_eff &= (self.df_t2r[key] != eval(value)).values
                mask_pur &= (self.df_r2t[key] != eval(value)).values
            else:
                raise ValueError("Selection operator may only be one of '==', '<=', or '>='.")
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
        
        print(f"Total Number of Interactions = {df_intrs.shape[0]}")
        print(f"Total Number of Particles = {df_parts.shape[0]}")
        
        self.df = pd.merge(df_intrs, df_parts, 
                           left_on=['Index', 'file_index', f'{mode}_interaction_id'], 
                           right_on=['Index', 'file_index', f'{mode}_particle_interaction_id'])
        
        assert (self.df.shape[0] == self.df_parts.shape[0])
        
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
                
    def update_interaction_df(self, df_other, pids=[0,1,2,3,4]):
        df_intrs_mgd = pd.merge(df_other,
                                self.df_intrs,
                                on=['Index', 'file_index', f'{self.mode}_interaction_id'],
                                how='left')
        for pid in pids:
            self.df_intrs[f'{self.mode}_num_primary_{self._MAPPING[pid]}'] \
                = df_intrs_mgd[f'{self.mode}_num_primary_{self._MAPPING[pid]}_new'].fillna(0).astype(int)