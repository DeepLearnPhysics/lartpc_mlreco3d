from typing import Callable, Tuple
import numpy as np
import pandas as pd

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.metrics import unique_label
from collections import defaultdict

from scipy.special import softmax
from .particle import *
from .point_matching import *

from mlreco.utils.groups import type_labels as TYPE_LABELS
from mlreco.utils.vertex import get_vertex, predict_vertex


class FullChainPredictor:

    CONCAT_RESULT = ['input_node_features', 
                     'input_edge_features', 
                     'points', 'ppn_coords', 'mask_ppn', 'ppn_layers', 
                     'classify_endpoints', 'seediness', 'margins', 
                     'embeddings', 'fragments', 'fragments_seg', 
                     'shower_fragments', 'shower_edge_index',
                     'shower_edge_pred','shower_node_pred',
                     'shower_group_pred','track_fragments', 
                     'track_edge_index', 'track_node_pred', 'track_edge_pred', 
                     'track_group_pred', 'particle_fragments', 
                     'particle_edge_index', 'particle_node_pred', 
                     'particle_edge_pred', 'particle_group_pred', 
                     'particles','inter_edge_index', 'inter_node_pred', 
                     'inter_edge_pred', 'node_pred_p', 'node_pred_type', 
                     'flow_edge_pred', 'kinematics_particles', 
                     'kinematics_edge_index', 'clust_fragments', 
                     'clust_frag_seg', 'interactions', 'inter_cosmic_pred', 
                     'node_pred_vtx', 'total_num_points', 
                     'total_nonghost_points', 'spatial_embeddings', 
                     'occupancy', 'hypergraph_features', 
                     'features', 'feature_embeddings', 'covariance']

    '''
    User Interface for full chain inference.

    Usage Example:

        model = Trainer._net.module
        entry = 0   # batch id
        predictor = FullChainPredictor(model, data_blob, res, cfg)
        pred_seg = predictor._fit_predict_semantics(entry)

    Instructions
    -----------------------------------------------------------------------

    1) To avoid confusion between different quantities, the label namings under
    iotools.schema must be set as follows:

        schema:
            input_data:
                - parse_sparse3d_scn
                - sparse3d_pcluster

    2) By default, unwrapper must be turned ON under trainval:

        trainval:
            unwrapper: unwrap_3d_mink

    3) Some outputs needs to be listed under trainval.concat_result.
    The predictor will run through a checklist to ensure this condition

    4) Does not support deghosting at the moment. (TODO)
    '''
    def __init__(self, model, data_blob, result, cfg):
        self.module_config = cfg['model']['modules']
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

        concat_result = cfg['trainval']['concat_result']
        check_concat = set(self.CONCAT_RESULT)
        for i, key in enumerate(concat_result):
            if not key in check_concat:
                raise ValueError("Output key <{}> should be listed under "\
                    "trainval.concat_result!".format(key))
        
            
    def _fit_predict_ppn(self, entry):
        '''
        Method for predicting ppn predictions.

        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - df (pd.DataFrame): pandas dataframe of ppn points, with
            x, y, z, coordinates, Score, Type, and sample index.
        '''
        index = self.index[entry]
        segmentation = self.result['segmentation'][entry]
        pred_seg = np.argmax(segmentation, axis=1).astype(int)
        
        ppn = uresnet_ppn_type_point_selector(self.data_blob['input_data'][entry], 
                                              self.result, 
                                              entry=entry)
        print(ppn.shape)
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
        Method for predicting semantic segmentation labels. 

        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted segmentation labels.  
        '''
        segmentation = self.result['segmentation'][entry]
        out = np.argmax(segmentation, axis=1).astype(int)
        return out


    def _fit_predict_gspice_fragments(self, entry):
        '''
        Method for predicting fragment labels (dense clustering)
        using graph spice. 

        Inputs:

            - entry: Batch number to retrieve example.

        Returns:

            - pred: 1D numpy integer array of predicted fragment labels.
            The labels only range over classes which were designated to be
            processed in GraphSPICE. 

            - G: networkx graph representing the current entry

            - subgraph: same graph in torch_geometric.Data format. 
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
    
    @staticmethod
    def randomize_labels(labels):
        '''
        Simple method to randomize label order (useful for plotting)
        '''
        labels, _ = unique_label(labels)
        
        N = np.unique(labels).shape[0]
        perm = np.random.permutation(N)
        
        print(N, perm)
        
        new_labels = -np.ones(labels.shape[0]).astype(int)
        for i, c in enumerate(perm):
            mask = labels == i
            new_labels[mask] = c
        return new_labels
    

    def _fit_predict_fragments(self, entry):
        '''
        Method for obtaining voxel-level fragment labels for full image, 
        including labels from both GraphSPICE and DBSCAN. 

        "Voxel-level" means that the label tensor has the same length
        as the full point cloud of the current image (specified by entry #)

        If a voxel is not assigned to any fragment (ex. low E depositions),
        we assign -1 as its fragment label. 


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted fragment labels.  
        '''
        fragments = self.result['fragments'][entry]
        
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_frag_labels = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(fragments):
            pred_frag_labels[mask] = i
            
        new_labels = pred_frag_labels
            
        return new_labels
    

    def _fit_predict_groups(self, entry):
        '''
        Method for obtaining voxel-level group labels. 

        If a voxel does not belong to any particle (ex. low E depositions),
        we assign -1 as its group (particle) label. 


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted group labels.  
        '''
        particles = self.result['particles'][entry]
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_group_labels = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(particles):
            pred_group_labels[mask] = i
            
        new_labels = pred_group_labels
            
        return new_labels
    

    def _fit_predict_interaction_labels(self, entry):
        '''
        Method for obtaining voxel-level interaction labels for full image.

        If a voxel does not belong to any interaction (ex. low E depositions),
        we assign -1 as its interaction (particle) label. 


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted interaction labels.  
        '''
        inter_group_pred = self.result['inter_group_pred'][entry]
        particles = self.result['particles'][entry]
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        pred_inter_labels = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(particles):
            pred_inter_labels[mask] = inter_group_pred[i]
            
        new_labels = pred_inter_labels
            
        return new_labels
    

    def _fit_predict_pids(self, entry):
        '''
        Method for obtaining voxel-level particle type 
        (photon, electron, muon, ...) labels for full image.

        If a voxel does not belong to any particle (ex. low E depositions),
        we assign -1 as its particle type label. 


        Inputs:
            - entry: Batch number to retrieve example.

        Returns:
            - labels: 1D numpy integer array of predicted particle type labels.  
        '''
        particles = self.result['particles'][entry]
        type_logits = self.result['node_pred_type'][entry]
        pids = np.argmax(type_logits, axis=1)
        num_voxels = self.data_blob['input_data'][entry].shape[0]
        
        pred_pids = -np.ones(num_voxels).astype(int)
    
        for i, mask in enumerate(particles):
            pred_pids[mask] = pids[i]
            
        return pred_pids


    def _fit_predict_vertex_info(self, entry, inter_idx, **kwargs):
        '''
        Method for obtaining interaction vertex information given
        entry number and interaction ID number.

        Inputs:
            - entry: Batch number to retrieve example.

            - inter_idx: Interaction ID number. 

        If the interaction specified by <inter_idx> does not exist
        in the sample numbered by <entry>, function will raise a
        ValueError. 

        Returns:
            - vertex_info: tuple of length 4, with the following objects:
                * ppn_candidates: 
                * c_candidates: 
                * vtx_candidate: (x,y,z) coordinate of predicted vertex
                * vtx_std: standard error on the predicted vertex
                
        '''
        vertex_info = predict_vertex(inter_idx, entry, 
                                     self.data_blob['input_data'],
                                     self.result, **kwargs)

        return vertex_info
    

    def get_particles(self, entry, semantic_type=False, threshold=2) -> List[Particle]:
        '''
        Method for retriving particle list for given batch index.

        The output particles will have its ppn candidates attached as
        attributes in the form of pandas dataframes (same as _fit_predict_ppn)

        Method also performs endpoint prediction for tracks and startpoint
        prediction for showers. 

        1) If a track has no or only one ppn candidate, the endpoints
        will be calculated by selecting two voxels that have the largest
        separation distance. Otherwise, the two ppn candidates with the
        largest separation from the particle coordinate centroid will be
        selected. 

        2) If a shower has no ppn candidates, <get_shower_startpoint> 
        will raise an Exception. Otherwise it selects the ppn candidate
        with the closest Hausdorff distance to the particle point cloud
        (smallest point-to-set distance)

        Inputs:
            - entry: Batch number to retrieve example.
            - semantic_type (optional): if True, only ppn candiates with the
            same predicted semantic type will be matched to its corresponding
            particle.
            - threshold (float, optional): threshold distance to attach
            ppn point to particle. 

        Returns:
            - out: List of <Particle> instances (see Particle class definition).
        '''
        point_cloud = self.data_blob['input_data'][entry][:, 1:4]
        depositions = self.data_blob['input_data'][entry][:, 4]
        particles = self.result['particles'][entry]
        inter_group_pred = self.result['inter_group_pred'][entry]
        
        particles_seg = self.result['particles_seg'][entry]
        
        type_logits = self.result['node_pred_type'][entry]
        pids = np.argmax(type_logits, axis=1)
        
        out = []
        
        assert len(inter_group_pred) == len(particles)
        assert len(particles_seg) == len(particles)
        assert len(pids) == len(particles)

        node_pred_vtx = self.result['node_pred_vtx'][entry]

        pred_group_ids = self._fit_predict_groups(entry)

        assert node_pred_vtx.shape[0] == len(particles)
        
        for i, p in enumerate(particles):
            voxels = point_cloud[p]
            seg_label = particles_seg[i]
            interaction_id = inter_group_pred[i]
            is_primary = bool(np.argmax(node_pred_vtx[i][3:]))
            part = Particle(voxels, i, seg_label, interaction_id, 
                            pids[i], batch_id=entry, 
                            depositions=depositions, is_primary=is_primary, 
                            pid_conf=softmax(type_logits[i])[pids[i]])
            part.voxel_indices = p
            out.append(part)

        ppn_results = self._fit_predict_ppn(entry)

        match_points_to_particles(ppn_results, out, threshold=threshold)
        # This should probably be separated to a selection algorithm
        for p in out:
            if p.semantic_type == 0:
                pt = get_shower_startpoint(p)
                p.startpoint = pt
            elif p.semantic_type == 1:
                pts = get_track_endpoints(p)
                p.endpoints = pts

        return out


    def get_interactions(self, entry, **kwargs) -> List[Interaction]:
        '''
        Method for retriving interaction list for given batch index.

        The output particles will have its constituent particles attached as
        attributes as List[Particle]. 

        Method also performs vertex prediction for each interaction.

        Inputs:
            - entry: Batch number to retrieve example.
            - semantic_type (optional): if True, only ppn candiates with the
            same predicted semantic type will be matched to its corresponding
            particle.
            - threshold (float, optional): threshold distance to attach
            ppn point to particle. 

        Returns:
            - out: List of <Interaction> instances (see particle.Interaction).
        '''
        particles = self.get_particles(entry, **kwargs)
        out = group_particles_to_interactions_fn(particles)
        for ia in out:
            vertex_info = self._fit_predict_vertex_info(entry, ia.id)
            ia.vertex = vertex_info[2]
        return out


    def fit_predict_labels(self, entry, **kwargs):
        '''
        Predict all labels of a given batch index <entry>.

        We define <labels> to be 1d tensors that annotate voxels. 
        '''
        pred_seg = self._fit_predict_semantics(entry)
        pred_fragments = self._fit_predict_fragments(entry, **kwargs)
        pred_groups = self._fit_predict_groups(entry, **kwargs)
        pred_interaction_labels = self._fit_predict_interaction_labels(entry, **kwargs)
        pred_ppn = self._fit_predict_ppn(entry, **kwargs)
        pred_pids = self._fit_predict_pids(entry)

        return {
            'pred_seg': pred_seg,
            'pred_fragments': pred_fragments,
            'pred_groups': pred_groups,
            'pred_interaction_labels': pred_interaction_labels,
            # 'pred_ppn': pred_ppn,
            'pred_pids': pred_pids
        }


    def fit_predict(self, **kwargs):
        '''
        Predict all samples in a given batch contained in <data_blob>.

        After calling fit_predict, the prediction information can be accessed
        as follows:

            - self._labels[entry]: labels dict (see fit_predict_labels) for 
            batch id <entry>.

            - self._particles[entry]: list of particles for batch id <entry>.

            - self._interactions[entry]: list of interactions for batch id <entry>.
        '''
        labels = []
        list_particles, list_interactions = [], []

        for entry in range(self.num_batches):

            pred_dict = self.fit_predict_entry(entry)
            labels.append(pred_dict)
            particles = self.get_particles(entry, **kwargs)
            interactions = self.get_interactions(entry)
            list_particles.append(particles)
            list_interactions.append(interactions)

        self._particles = list_particles
        self._interactions = list_interactions
        self._labels = labels

        return labels


class FullChainEvaluator(FullChainPredictor):
    '''
    Helper class for full chain prediction and evaluation.

    Usage:

        model = Trainer._net.module
        entry = 0   # batch id
        predictor = FullChainEvaluator(model, data_blob, res, cfg)
        pred_seg = predictor.get_true_labels(entry, mode='segmentation')

    To avoid confusion between different quantities, the label namings under
    iotools.schema must be set as follows:

        schema:
            input_data:
                - parse_sparse3d_scn
                - sparse3d_pcluster
            segment_label:
                - parse_sparse3d_scn
                - sparse3d_pcluster_semantics
            cluster_label:
                - parse_cluster3d_clean_full
                #- parse_cluster3d_full
                - cluster3d_pcluster
                - particle_pcluster
                #- particle_mpv
                - sparse3d_pcluster_semantics
            particles_label:
                - parse_particle_points_with_tagging
                - sparse3d_pcluster
                - particle_corrected
            kinematics_label:
                - parse_cluster3d_kinematics_clean
                - cluster3d_pcluster
                - particle_corrected
                #- particle_mpv
                - sparse3d_pcluster_semantics
            particle_graph:
                - parse_particle_graph_corrected
                - particle_corrected
                - cluster3d_pcluster
            particles_asis:
                - parse_particle_asis
                - particle_pcluster
                - cluster3d_pcluster


    Instructions
    ----------------------------------------------------------------

    The FullChainEvaluator share the same methods as FullChainPredictor, 
    with additional methods to retrieve ground truth information for each 
    abstraction level. 
    '''


    def __init__(self, model, data_blob, result, cfg):
        super(FullChainEvaluator, self).__init__(model, data_blob, result, cfg)


    def get_true_particles(self, entry) -> List[TruthParticle]:
        '''
        Get list of <TruthParticle> instances for given <entry> batch id. 

        The method will return particles only if its id number appears in
        the group_id column of cluster_label. 

        Each TruthParticle will contain the following information (attributes):

            points: N x 3 coordinate array for particle's full image. 
            id: group_id 
            semantic_type: true semantic type
            interaction_id: true interaction id 
            pid: PDG type (photons: 0, electrons: 1, ...)
            fragments: list of integers corresponding to constituent fragment
                id number
            p: true momentum vector
        '''
        labels = self.data_blob['cluster_label'][entry]
        particle_ids = set(list(np.unique(labels[:, 6]).astype(int)))

        particles = []

        for idx, p in enumerate(self.data_blob['particles_asis'][entry]):
            pid = int(p.id())
            if pid not in particle_ids:
                continue
            is_primary = p.group_id() == p.parent_id()
            if p.pdg_code() not in TYPE_LABELS:
                continue
            pdg = TYPE_LABELS[p.pdg_code()]
            mask = labels[:, 6] == pid
            coords = self.data_blob['input_data'][entry][mask][:, 1:4]
            
            semantic_type = np.unique(labels[mask][:, -1])
            if semantic_type.shape[0] > 1:
                raise ValueError("Interaction ID of Particle {} is not "\
                    "unique: {}".format(pid, str(semantic_type)))
            else:
                semantic_type = semantic_type[0]

            interaction_id = np.unique(labels[mask][:, 7].astype(int))
            if interaction_id.shape[0] > 1:
                raise ValueError("Interaction ID of Particle {} is not "\
                    "unique: {}".format(pid, str(interaction_id)))
            else:
                interaction_id = interaction_id[0]

            nu_id = np.unique(labels[mask][:, 8].astype(int))
            if nu_id.shape[0] > 1:
                raise ValueError("Neutrino ID of Particle {} is not "\
                    "unique: {}".format(pid, str(nu_id)))
            else:
                nu_id = nu_id[0]

            fragments = np.unique(labels[mask][:, 5].astype(int))
            depositions = self.data_blob['input_data'][entry][mask][:, 4].squeeze()
            particle = TruthParticle(coords, pid, semantic_type, interaction_id, 
                pdg, batch_id=entry, depositions=depositions, is_primary=is_primary)
            particle.p = np.array([p.px(), p.py(), p.pz()])
            particle.fragments = fragments
            particle.particle_asis = p
            particle.nu_id = nu_id
            particle.voxel_indices = np.where(mask)[0]
            particles.append(particle)

        return particles


    def get_true_interactions(self, entry) -> List[Interaction]:
        true_particles = self.get_true_particles(entry)
        out = group_particles_to_interactions_fn(true_particles, get_nu_id=True)
        vertices = self.get_true_vertices(entry)
        for ia in out:
            ia.vertex = vertices[ia.id]
        return out


    def get_true_vertices(self, entry):
        inter_idxs = np.unique(self.data_blob['cluster_label'][entry][:, 7].astype(int))
        out = {}
        for inter_idx in inter_idxs:
            if inter_idx < 0:
                continue
            vtx = get_vertex(self.data_blob['kinematics_label'],
                            self.data_blob['cluster_label'],
                            data_idx=entry,
                            inter_idx=inter_idx)
            out[inter_idx] = vtx
        return out


    def match_particles(self, entry, relabel=False, primaries=True,
                        min_overlap_count=1, threshold=2):
        pred_particles = self.get_particles(entry, threshold=threshold)
        true_particles = self.get_true_particles(entry)
        match = match_particles_fn(pred_particles, true_particles, 
                                   relabel=relabel, primaries=primaries,
                                   min_overlap_count=min_overlap_count)
        return match


    def match_interactions(self, entry, min_overlap_count=1):
        pred_ias = self.get_interactions(entry)
        true_ias = self.get_true_interactions(entry)
        match = match_interactions_fn(pred_ias, true_ias, 
                                      min_overlap_count=min_overlap_count)
        return match