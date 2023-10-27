import numpy as np
import pandas as pd

class TopologySelector:
    
    _LABEL_DICT = {
        'num_primary_muons': '$\\mu$',
        'num_primary_protons': '$p$',
        'num_primary_electrons': '$e$',
        'num_primary_pions': '$\\pi^\\pm$',
        'num_primary_photons': '$\\gamma$'
    }
    
    _FLASHMATCH_BOUNDS = [0., 1.6]
    
    def __init__(self, df_t2r, df_r2t, cut=None):
        self.df_t2r = df_t2r
        self.df_r2t = df_r2t
        
        if cut is not None:
            self._query_cut(cut)
        
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
                
    def _query_cut(self, expression):
        self.df_t2r = self.df_t2r.query(expression)
        self.df_r2t = self.df_r2t.query(expression)