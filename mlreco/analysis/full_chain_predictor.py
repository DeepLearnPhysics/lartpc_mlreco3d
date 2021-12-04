import numpy as np
import pandas as pd

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.metrics import unique_label
from collections import defaultdict

from scipy.special import softmax
from .particle import Particle
from .point_matching import *


class FullChainPredictor:
    '''
    Helper class for full chain prediction.

    Usage:

        model = Trainer._net.module
        entry = 0   # batch id
        predictor = FullChainPredictor(model, data_blob, res, 
                                       cfg['model']['modules'])
        pred_seg = predictor._fit_predict_semantics(entry)

    '''
    def __init__(self, model, data_blob, result, cfg):
        self.module_config = cfg
        self.data_blob = data_blob
        self.result = result
        self.num_batches = len(data_blob['input_data'])
        self.index = self.data_blob['index']
        
        # For Fragment Clustering
        self.cluster_graph_constructor = model.gs_manager
        if model.enable_cnn_clust:
            self.gspice_fragment_manager = model._gspice_fragment_manager
        if model.enable_dbscan:
            self.dbscan_fragment_manager = model.dbscan_fragment_manager
        if not (model.enable_cnn_clust or model.enable_dbscan):
            msg = '''
                Neither CNN clustering nor dbscan clustering is enabled
                in the model. Cannot initialize Fragment Manager!
            '''
            raise AttributeError(msg)
        
            
    def _fit_predict_ppn(self, entry):
        '''
        Requires:
            - segmentation
            - points
            - mask_ppn
            - ppn_layers
            - ppn_coords
        '''
        index = self.index[entry]
        segmentation = self.result['segmentation'][entry]
        pred_seg = np.argmax(segmentation, axis=1).astype(int)
        
        ppn = uresnet_ppn_type_point_selector(self.data_blob['input_data'][entry], 
                                              self.result, 
                                              entry=entry)
        ppn_voxels = ppn[:, 1:4]
        ppn_score = ppn[:, 5]
        ppn_type = ppn[:, 12]
        
        update_dict = defaultdict(list)
        for i, pred_point in enumerate(ppn_voxels):
            pred_point_type, pred_point_score = ppn_type[i], ppn_score[i]
            x, y, z = ppn_voxels[i][0], ppn_voxels[i][1], ppn_voxels[i][2]
            update_dict['Index'].append(index)
            update_dict['PPN Point Type'].append(pred_point_type)
            update_dict['PPN Point Score'].append(pred_point_score)
            update_dict['x'].append(x)
            update_dict['y'].append(y)
            update_dict['z'].append(z)
        
        df = pd.DataFrame(update_dict)
        return df
            
        
    def _fit_predict_semantics(self, entry):
        '''
        Requires:
            - segmentation
        '''
        segmentation = self.result['segmentation'][entry]
        out = np.argmax(segmentation, axis=1)
        return out
        
    
#     def _fit_predict_dbscan_fragments(self, entry):
        
#         wrapped_data = np.vstack(self.data_blob['input_data'])
        
#         cnn_result = {
#             'segmentation': np.vstack(self.result['segmentation']),
#             'points': np.vstack(self.result['points']), 
#             'ppn_coords': np.vstack(self.data_blob['ppn_coords']), 
#             'mask_ppn': np.vstack(self.data_blob['mask_ppn']), 
#         }
        
#         fragment_data = self.dbscan_fragment_manager(wrapped_data, )
        
#         pass
    
    def _fit_predict_gspice_fragments(self, entry):
        '''
        Requires:
            - segmentation
            - graph
            - graph_info
        '''
        import warnings
        warnings.filterwarnings('ignore') 
        
        graph = self.result['graph'][0]
        graph_info = self.result['graph_info'][0]
        index_mapping = { key : val for key, val in zip(
           range(0, len(graph_info.Index.unique())), self.index)}
        
        min_points = self.module_config['graph_spice'].get('min_points', 1)
        invert = self.module_config['graph_spice_loss'].get('invert', True)
        
        graph_info['Index'] = graph_info['Index'].map(index_mapping)
        constructor_cfg = self.cluster_graph_constructor.constructor_cfg
        gs_manager = ClusterGraphConstructor(constructor_cfg,
                                             graph_batch=graph,
                                             graph_info=graph_info,
                                             batch_col=0,
                                             training=False)
        pred, G, subgraph = gs_manager.fit_predict_one(entry, 
                                                       invert=invert, 
                                                       min_points=min_points)
    
        return pred, G, subgraph
    
    
    def randomize_labels(self, labels):
        
        labels, _ = unique_label(labels)
        
        N = np.unique(labels).shape[0]
        perm = np.random.permutation(N)
        
        print(N, perm)
        
        new_labels = -np.ones(labels.shape[0]).astype(int)
        for i, c in enumerate(perm):
            mask = labels == i
            new_labels[mask] = c
        return new_labels
    
    def _fit_predict_fragments(self, entry, randomize=False):
        '''
                
        '''

        fragments = self.result['fragments'][entry]
        
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_frag_labels = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(fragments):
            pred_frag_labels[mask] = i
            
        new_labels = pred_frag_labels
            
        if randomize:
            new_labels = self.randomize_labels(pred_frag_labels)
            new_labels[pred_frag_labels == -1] = -1
            
        return new_labels
    
    def _fit_predict_groups(self, entry, randomize=False):
        
        particles = self.result['particles'][entry]
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_group_labels = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(particles):
            pred_group_labels[mask] = i
            
        new_labels = pred_group_labels
            
        if randomize:
            new_labels = self.randomize_labels(pred_group_labels)
            new_labels[pred_group_labels == -1] = -1
            
        return new_labels
    
    def _fit_predict_interactions(self, entry, randomize=True):
        
        inter_group_pred = self.result['inter_group_pred'][entry]
        particles = self.result['particles'][entry]
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_inter_labels = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(particles):
            pred_inter_labels[mask] = inter_group_pred[i]
            
        new_labels = pred_inter_labels
        if randomize:
            new_labels = self.randomize_labels(pred_inter_labels)
            new_labels[pred_inter_labels == -1] = -1
            
        return new_labels
    
    def _fit_predict_pids(self, entry):
        
        particles = self.result['particles'][entry]
        type_logits = self.result['node_pred_type'][entry]
        pids = np.argmax(type_logits, axis=1)
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        
        pred_pids = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(particles):
            pred_pids[mask] = pids[i]
            
        return pred_pids
    
    def _fit_predict_particles(self, entry, relabel=True):
        
        point_cloud = self.data_blob['input_data'][entry][:, 1:4]
        particles = self.result['particles'][entry]
        inter_group_pred = self.result['inter_group_pred'][entry]
        
        if relabel:
            inter_group_pred, _ = unique_label(inter_group_pred)
        
        particles_seg = self.result['particles_seg'][entry]
        
        type_logits = self.result['node_pred_type'][entry]
        pids = np.argmax(type_logits, axis=1)
        
        out = []
        
        assert len(inter_group_pred) == len(particles)
        assert len(particles_seg) == len(particles)
        assert len(pids) == len(particles)
        
        for i, p in enumerate(particles):
            voxels = point_cloud[p]
            semantic_type = particles_seg[i]
            interaction_id = inter_group_pred[i]
            part = Particle(voxels, i, semantic_type, interaction_id, 
                            pids[i], softmax(type_logits[i])[pids[i]], 0)
            out.append(part)

        ppn_results = self._fit_predict_ppn(entry)

        match_points_to_particles(ppn_results, out)
        for p in out:
            if p.semantic_type == 0:
                pt = get_shower_startpoint(p)
                p.startpoint = pt
            elif p.semantic_type == 1:
                pts = get_track_endpoints(p)
                p.endpoints = pts

        return out


class FullChainEvaluator(FullChainPredictor):
    '''
    TODO
    '''
    def __init__(self, model, data_blob, result, cfg, labels):
        super(FullChainEvaluator, self).__init__(model, data_blob, result, cfg)
