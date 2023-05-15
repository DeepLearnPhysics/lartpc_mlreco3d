from abc import ABC, abstractmethod
from typing import List
from pprint import pprint
from collections import OrderedDict

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
import copy

from mlreco.utils.globals import (BATCH_COL, 
                                  COORD_COLS, 
                                  PDG_TO_PID, 
                                  VALUE_COL, 
                                  VTX_COLS, 
                                  INTER_COL,
                                  GROUP_COL,
                                  PSHOW_COL,
                                  CLUST_COL)
from analysis.classes import (Particle, 
                              TruthParticle,
                              Interaction, 
                              TruthInteraction,
                              ParticleFragment,
                              TruthParticleFragment)
from analysis.classes.matching import group_particles_to_interactions_fn
from mlreco.utils.vertex import get_vertex

class DataBuilder(ABC):
    """Abstract base class for building all data structures

    A DataBuilder takes input data and full chain output dictionaries
    and processes them into human-readable data structures.
    
    """
    def build(self, data: dict, result: dict, mode='reco'):
        """Process all images in the current batch and change representation
        into each respective data format.
        
        Parameters
        ----------
        data: dict
        result: dict
        mode: str
            Indicator for building reconstructed vs true data formats.
            In other words, mode='reco' will produce <Particle> and
            <Interaction> data formats, while mode='truth' is reserved for
            <TruthParticle> and <TruthInteraction>
        """
        output = []
        num_batches = len(data['index'])
        for bidx in range(num_batches):
            entities = self.build_image(bidx, data, result, mode=mode)
            output.append(entities)
        return output

    def build_image(self, entry: int, data: dict, result: dict, mode='reco'):
        """Build data format for a single image.
        
        Parameters
        ----------
        entry: int
            Batch id number for the image.
        """
        if mode == 'truth':
            entities = self._build_truth(entry, data, result)
        elif mode == 'reco':
            entities = self._build_reco(entry, data, result)
        else:
            raise ValueError(f"Particle builder mode {mode} not supported!")
        
        return entities
        
    @abstractmethod
    def _build_truth(self, entry, data: dict, result: dict):
        raise NotImplementedError
    
    @abstractmethod
    def _build_reco(self, entry, data: dict, result: dict):
        raise NotImplementedError
    
    # @abstractmethod
    # def _load_reco(self, entry, data: dict, result: dict):
    #     raise NotImplementedError
    
    # @abstractmethod
    # def _load_true(self, entry, data: dict, result: dict):
    #     raise NotImplementedError
    
    def load_image(self, entry: int, data: dict, result: dict, mode='reco'):
        """Load single image worth of entity blueprint from HDF5 
        and construct original data structure instance.

        Parameters
        ----------
        entry : int
            Image ID
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        mode : str, optional
            Whether to load reco or true entities, by default 'reco'

        Returns
        -------
        entities: List[Any]
            List of constructed entities from their HDF5 blueprints. 
        """
        if mode == 'truth':
            entities = self._load_truth(entry, data, result)
        elif mode == 'reco':
            entities = self._load_reco(entry, data, result)
        else:
            raise ValueError(f"Particle loader mode {mode} not supported!")
        
        return entities
    
    def load(self, data: dict, result: dict, mode='reco'):
        """Process all images in the current batch of HDF5 data and
        construct original data structures.
        
        Parameters
        ----------
        data: dict
            Data dictionary
        result: dict
            Result dictionary
        mode: str
            Indicator for building reconstructed vs true data formats.
            In other words, mode='reco' will produce <Particle> and
            <Interaction> data formats, while mode='truth' is reserved for
            <TruthParticle> and <TruthInteraction>
        """
        output = []
        num_batches = len(data['index'])
        for bidx in range(num_batches):
            entities = self.load_image(bidx, data, result, mode=mode)
            output.append(entities)
        return output


class ParticleBuilder(DataBuilder):
    """Builder for constructing Particle and TruthParticle instances
    from full chain output dicts. 
    
    Required result keys:

        reco:
            - input_rescaled
            - particle_clusts
            - particle_seg
            - particle_start_points
            - particle_end_points
            - particle_group_pred
            - particle_node_pred_type
            - particle_node_pred_vtx
        truth:
            - cluster_label
            - cluster_label_adapted
            - particles_asis
            - input_rescaled
    """
    def __init__(self, builder_cfg={}):
        self.cfg = builder_cfg
        
    def _load_reco(self, entry, data: dict, result: dict):
        """Construct Particle objects from loading HDF5 blueprints.

        Parameters
        ----------
        entry : int
            Image ID
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        out : List[Particle]
            List of restored particle instances built from HDF5 blueprints.
        """
        if 'input_rescaled' in result:
            point_cloud = result['input_rescaled'][0]
        elif 'input_data' in data:
            point_cloud = data['input_data'][0]
        else:
            msg = "To build Particle objects from HDF5 data, need either "\
                "input_data inside data dictionary or input_rescaled inside"\
                " result dictionary."
            raise KeyError(msg)
        out  = []
        blueprints = result['particles'][0]
        for i, bp in enumerate(blueprints):
            mask = bp['index']
            prepared_bp = copy.deepcopy(bp)
            
            match = prepared_bp.pop('match', [])
            match_counts = prepared_bp.pop('match_counts', [])
            assert len(match) == len(match_counts)
            
            prepared_bp.pop('depositions_sum', None)
            group_id = prepared_bp.pop('id', -1)
            prepared_bp['group_id'] = group_id
            prepared_bp.update({
                'points': point_cloud[mask][:, COORD_COLS],
                'depositions': point_cloud[mask][:, VALUE_COL],
            })
            particle = Particle(**prepared_bp)
            if len(match) > 0:
                particle.match_counts = OrderedDict({
                    key : val for key, val in zip(match, match_counts)})
            # assert particle.image_id == entry
            out.append(particle)
        
        return out
    
    
    def _load_truth(self, entry, data, result):
        out = []
        true_nonghost = data['cluster_label'][0]
        particles_asis = data['particles_asis'][0]
        pred_nonghost = result['cluster_label_adapted'][0]
        blueprints = result['truth_particles'][0]
        for i, bp in enumerate(blueprints):
            mask = bp['index']
            true_mask = bp['truth_index']
            pasis_selected = None
            # Find particles_asis
            for pasis in particles_asis:
                if pasis.id() == bp['id']:
                    pasis_selected = pasis
            assert pasis_selected is not None
            
            # recipe = {
            #     'index': mask,
            #     'truth_index': true_mask,
            #     'points': pred_nonghost[mask][:, COORD_COLS],
            #     'depositions': pred_nonghost[mask][:, VALUE_COL],
            #     'truth_points': true_nonghost[true_mask][:, COORD_COLS],
            #     'truth_depositions': true_nonghost[true_mask][:, VALUE_COL],
            #     'particle_asis': pasis_selected,
            #     'group_id': group_id
            # }
            
            prepared_bp = copy.deepcopy(bp)
            
            group_id = prepared_bp.pop('id', -1)
            prepared_bp['group_id'] = group_id
            prepared_bp.pop('depositions_sum', None)
            prepared_bp.update({
                
                'points': pred_nonghost[mask][:, COORD_COLS],
                'depositions': pred_nonghost[mask][:, VALUE_COL],
                'truth_points': true_nonghost[true_mask][:, COORD_COLS],
                'truth_depositions': true_nonghost[true_mask][:, VALUE_COL],
                'particle_asis': pasis_selected
            })
            
            match = prepared_bp.pop('match', [])
            match_counts = prepared_bp.pop('match_counts', [])
            
            truth_particle = TruthParticle(**prepared_bp)
            if len(match) > 0:
                truth_particle.match_counts = OrderedDict({
                    key : val for key, val in zip(match, match_counts)})
            # assert truth_particle.image_id == entry
            assert truth_particle.truth_size > 0
            out.append(truth_particle)
            
        return out
        

    def _build_reco(self, 
                    entry: int, 
                    data: dict, 
                    result: dict) -> List[Particle]:
        """
        Returns
        -------
        out : List[Particle]
            list of reco Particle instances of length equal to the
            batch size. 
        """
        out = []

        # Essential Information
        image_index      = data['index'][entry]
        volume_labels    = result['input_rescaled'][entry][:, BATCH_COL]
        point_cloud      = result['input_rescaled'][entry][:, COORD_COLS]
        depositions      = result['input_rescaled'][entry][:, 4]
        particles        = result['particle_clusts'][entry]
        particle_seg     = result['particle_seg'][entry]

        particle_start_points = result['particle_start_points'][entry][:, COORD_COLS]
        particle_end_points   = result['particle_end_points'][entry][:, COORD_COLS]
        inter_ids             = result['particle_group_pred'][entry]

        type_logits           = result['particle_node_pred_type'][entry]
        primary_logits        = result['particle_node_pred_vtx'][entry]

        pid_scores     = softmax(type_logits, axis=1)
        primary_scores = softmax(primary_logits, axis=1)

        for i, p in enumerate(particles):
            volume_id, cts = np.unique(volume_labels[p], return_counts=True)
            volume_id = int(volume_id[cts.argmax()])
            seg_label = particle_seg[i]
            # pid = -1
            # if seg_label == 2 or seg_label == 3: # DANGEROUS
            #     pid = 1
            interaction_id = inter_ids[i]
            part = Particle(group_id=i,
                            interaction_id=interaction_id,
                            image_id=image_index,
                            semantic_type=seg_label, 
                            index=p,
                            points=point_cloud[p],
                            depositions=depositions[p],
                            volume_id=volume_id,
                            pid_scores=pid_scores[i],
                            primary_scores=primary_scores[i],
                            start_point = particle_start_points[i],
                            end_point = particle_end_points[i])

            out.append(part)

        return out
    
    def _build_truth(self, 
                    entry: int, 
                    data: dict, 
                    result: dict) -> List[TruthParticle]:
        """
        Returns
        -------
        out : List[TruthParticle]
            list of true TruthParticle instances of length equal to the
            batch size. 
        """

        out = []
        image_index     = data['index'][entry]
        labels          = result['cluster_label_adapted'][entry]
        labels_nonghost = data['cluster_label'][entry]
        larcv_particles = data['particles_asis'][entry]
        rescaled_charge = result['input_rescaled'][entry][:, 4]
        particle_ids    = set(list(np.unique(labels[:, 6]).astype(int)))
        coordinates     = result['input_rescaled'][entry][:, COORD_COLS]
        # point_labels   = data['point_labels'][entry]    

        for i, lpart in enumerate(larcv_particles):
            id = int(lpart.id())
            pdg = PDG_TO_PID.get(lpart.pdg_code(), -1)
            # print(pdg)
            is_primary = lpart.group_id() == lpart.parent_id()
            mask_nonghost = labels_nonghost[:, 6].astype(int) == id
            if np.count_nonzero(mask_nonghost) <= 0:
                continue  # Skip larcv particles with no true depositions
            # 1. Check if current pid is one of the existing group ids
            if id not in particle_ids:
                particle = handle_empty_truth_particles(labels_nonghost, 
                                                       mask_nonghost, 
                                                       lpart, 
                                                       image_index)
                out.append(particle)
                continue

            # 1. Process voxels
            mask = labels[:, 6].astype(int) == id
            # If particle is Michel electron, we have the option to
            # only consider the primary ionization.
            # Semantic labels only label the primary ionization as Michel.
            # Cluster labels will have the entire Michel together.
            # if self.michel_primary_ionization_only and 2 in labels[mask][:, -1].astype(int):
            #     mask = mask & (labels[:, -1].astype(int) == 2)
            #     mask_noghost = mask_noghost & (labels_nonghost[:, -1].astype(int) == 2)

            coords              = coordinates[mask]
            voxel_indices       = np.where(mask)[0]
            # fragments           = np.unique(labels[mask][:, 5].astype(int))
            depositions_MeV     = labels[mask][:, VALUE_COL]
            depositions         = rescaled_charge[mask] # Will be in ADC
            coords_noghost      = labels_nonghost[mask_nonghost][:, COORD_COLS]
            true_voxel_indices  = np.where(mask_nonghost)[0]
            depositions_noghost = labels_nonghost[mask_nonghost][:, VALUE_COL].squeeze()

            volume_labels       = labels_nonghost[mask_nonghost][:, BATCH_COL]
            volume_id, cts      = np.unique(volume_labels, return_counts=True)
            volume_id           = int(volume_id[cts.argmax()])
    
            # 2. Process particle-level labels
            semantic_type, int_id, nu_id = get_truth_particle_labels(labels, 
                                                                    mask, 
                                                                    pid=pdg)
            
            # 3. Process particle start / end point labels
            start_point, end_point = None, None

            particle = TruthParticle(group_id=id,
                                     interaction_id=int_id, 
                                     nu_id=nu_id,
                                     image_id=image_index,
                                     volume_id=volume_id,
                                     semantic_type=semantic_type, 
                                     index=voxel_indices,
                                     points=coords,
                                     depositions=depositions,
                                     start_point=start_point,
                                     end_point=end_point,
                                     depositions_MeV=depositions_MeV,
                                     truth_index=true_voxel_indices,
                                     truth_points=coords_noghost,
                                     truth_depositions=np.empty(0, dtype=np.float32), #TODO
                                     truth_depositions_MeV=depositions_noghost,
                                     is_primary=is_primary,
                                     pid=pdg,
                                     particle_asis=lpart)

            out.append(particle)

        return out


class InteractionBuilder(DataBuilder):
    """Builder for constructing Interaction and TruthInteraction instances.
    
    Required result keys:

        reco:
            - Particles
        truth:
            - TruthParticles
            - cluster_label
            - neutrino_asis (optional)
    """
    def __init__(self, builder_cfg={}):
        self.cfg = builder_cfg

    def _build_reco(self, entry: int, data: dict, result: dict) -> List[Interaction]:
        particles = result['particles'][entry]
        out = group_particles_to_interactions_fn(particles, 
                                                 get_nu_id=True, 
                                                 mode='pred')
        return out
    
    def _load_reco(self, entry, data, result):
        if 'input_rescaled' in result:
            point_cloud = result['input_rescaled'][0]
        elif 'input_data' in data:
            point_cloud = data['input_data'][0]
        else:
            msg = "To build Particle objects from HDF5 data, need either "\
                "input_data inside data dictionary or input_rescaled inside"\
                " result dictionary."
            raise KeyError(msg)
        
        out = []
        blueprints = result['interactions'][0]
        use_particles = 'particles' in result
        
        if not use_particles:
            msg = "Loading Interactions without building Particles. "\
            "This means Interaction.particles will be empty!"
            print(msg)
            
        for i, bp in enumerate(blueprints):
            info = {
                'interaction_id': bp['id'],
                'image_id': bp['image_id'],
                'is_neutrino': bp['is_neutrino'],
                'nu_id': bp['nu_id'],
                'volume_id': bp['volume_id'],
                'vertex': bp['vertex'],
                'flash_time': bp['flash_time'],
                'fmatched': bp['fmatched'],
                'flash_id': bp['flash_id'],
                'flash_total_pE': bp['flash_total_pE']
            }
            if use_particles:
                particles = []
                for p in result['particles'][0]:
                    if p.interaction_id == bp['id']:
                        particles.append(p)
                        continue
                ia = Interaction.from_particles(particles, 
                                                verbose=False, **info)
            else:
                mask = bp['index']
                info.update({
                    'index': mask, 
                    'points': point_cloud[mask][:, COORD_COLS],
                    'depositions': point_cloud[mask][:, VALUE_COL]
                })
                ia = Interaction(**info)
                
            # Handle matches
            match_counts = OrderedDict({i: val for i, val in zip(bp['match'], bp['match_counts'])})
            ia._match_counts = match_counts
            out.append(ia)
        return out
    
    def _build_truth(self, entry: int, data: dict, result: dict) -> List[TruthInteraction]:
        particles = result['truth_particles'][entry]
        out = group_particles_to_interactions_fn(particles, 
                                                 get_nu_id=True, 
                                                 mode='truth')
        out = self.decorate_truth_interactions(entry, data, out)
        return out
    
    def _load_truth(self, entry, data, result):
        true_nonghost = data['cluster_label'][0]
        pred_nonghost = result['cluster_label_adapted'][0]
        
        out = []
        blueprints = result['truth_interactions'][0]
        use_particles = 'truth_particles' in result
        
        if not use_particles:
            msg = "Loading TruthInteractions without building TruthParticles. "\
            "This means TruthInteraction.particles will be empty!"
            print(msg)
            
        for i, bp in enumerate(blueprints):
            info = {
                'interaction_id': bp['id'],
                'image_id': bp['image_id'],
                'is_neutrino': bp['is_neutrino'],
                'nu_id': bp['nu_id'],
                'volume_id': bp['volume_id'],
                'vertex': bp['vertex']
            }
            if use_particles:
                particles = []
                for p in result['truth_particles'][0]:
                    if p.interaction_id == bp['id']:
                        particles.append(p)
                        continue
                ia = TruthInteraction.from_particles(particles,
                                                     verbose=False, 
                                                     **info)
            else:
                mask = bp['index']
                true_mask = bp['truth_index']
                info.update({
                    'index': mask,
                    'truth_index': true_mask,
                    'points': pred_nonghost[mask][:, COORD_COLS],
                    'depositions': pred_nonghost[mask][:, VALUE_COL],
                    'truth_points': true_nonghost[true_mask][:, COORD_COLS],
                    'truth_depositions_MeV': true_nonghost[true_mask][:, VALUE_COL],
                })
                ia = TruthInteraction(**info)
            out.append(ia)
        return out
    
    def build_truth_using_particles(self, entry, data, particles):
        out = group_particles_to_interactions_fn(particles, 
                                                 get_nu_id=True, 
                                                 mode='truth')
        out = self.decorate_truth_interactions(entry, data, out)
        return out
    
    def decorate_truth_interactions(self, entry, data, interactions):
        """
        Helper function for attaching additional information to
        TruthInteraction instances. 
        """
        vertices = self.get_truth_vertices(entry, data)
        if 'neutrinos' not in data:
            print("Neutrino truth information not found in label data!")
        for ia in interactions:
            if ia.id in vertices:
                ia.vertex = vertices[ia.id]

            if 'neutrinos' in data and ia.nu_id == 1:
                neutrinos = data['neutrinos'][entry]
                if len(neutrinos) > 1 or len(neutrinos) == 0: continue
                nu = neutrinos[0]
                ia.is_neutrino = True
                # nu_pos = np.array([nu.position().x(),
                #                    nu.position().y(),
                #                    nu.position().z()], dtype=np.float32)
                # for p in ia.particles:
                #     pos = np.array([p.asis.ancestor_position().x(),
                #                     p.asis.ancestor_position().y(),
                #                     p.asis.ancestor_position().z()], dtype=np.float32)
                #     check_pos = np.linalg.norm(nu_pos - pos) > 1e-8
                    # if check_pos:
                ia.nu_interaction_type     = nu.interaction_type()
                ia.nu_interaction_mode     = nu.interaction_mode()
                ia.nu_current_type         = nu.current_type()
                ia.nu_energy_init          = nu.energy_init()

        return interactions
        
    def get_truth_vertices(self, entry, data: dict):
        """
        Helper function for retrieving true vertex information. 
        """
        out = {}
        inter_idxs = np.unique(
            data['cluster_label'][entry][:, INTER_COL].astype(int))
        for inter_idx in inter_idxs:
            if inter_idx < 0:
                continue
            vtx = get_vertex(data['cluster_label'],
                             data['cluster_label'],
                             data_idx=entry,
                             inter_idx=inter_idx,
                             vtx_col=VTX_COLS[0])
            mask    = data['cluster_label'][entry][:, INTER_COL].astype(int) == inter_idx
            points  = data['cluster_label'][entry][:, COORD_COLS]
            new_vtx = points[mask][np.linalg.norm(points[mask] - vtx, axis=1).argmin()]
            out[inter_idx] = new_vtx
        return out


class FragmentBuilder(DataBuilder):
    """Builder for constructing Particle and TruthParticle instances
    from full chain output dicts. 
    
    Required result keys:

        reco:
            - input_rescaled
            - fragment_clusts
            - fragment_seg
            - shower_fragment_start_points
            - track_fragment_start_points
            - track_fragment_end_points
            - shower_fragment_group_pred
            - track_fragment_group_pred
            - shower_fragment_node_pred
        truth:
            - cluster_label
            - cluster_label_adapted
            - input_rescaled
    """
    def __init__(self, builder_cfg={}):
        self.cfg = builder_cfg
        self.allow_nodes         = self.cfg.get('allow_nodes', [0,2,3])
        self.min_voxel_cut       = self.cfg.get('min_voxel_cut', -1)
        self.only_primaries      = self.cfg.get('only_primaries', False)
        self.include_semantics   = self.cfg.get('include_semantics', None)
        self.attaching_threshold = self.cfg.get('attaching_threshold', 5.0)
        self.verbose             = self.cfg.get('verbose', False)

    def _build_reco(self, entry, 
                    data: dict, 
                    result: dict):
                    
        volume_labels = result['input_rescaled'][entry][:, BATCH_COL]
        point_cloud = result['input_rescaled'][entry][:, COORD_COLS]
        depositions = result['input_rescaled'][entry][:, VALUE_COL]
        fragments = result['fragment_clusts'][entry]
        fragments_seg = result['fragment_seg'][entry]

        shower_mask = np.isin(fragments_seg, self.allow_nodes)
        shower_frag_primary = np.argmax(
            result['shower_fragment_node_pred'][entry], axis=1)

        shower_start_points = result['shower_fragment_start_points'][entry][:, COORD_COLS]
        track_start_points = result['track_fragment_start_points'][entry][:, COORD_COLS]
        track_end_points = result['track_fragment_end_points'][entry][:, COORD_COLS]

        assert len(fragments_seg) == len(fragments)

        temp = []

        shower_group = result['shower_fragment_group_pred'][entry]
        track_group = result['track_fragment_group_pred'][entry]

        group_ids = np.ones(len(fragments)).astype(int) * -1
        inter_ids = np.ones(len(fragments)).astype(int) * -1

        for i, p in enumerate(fragments):
            voxels = point_cloud[p]
            seg_label = fragments_seg[i]
            volume_id, cts = np.unique(volume_labels[p], return_counts=True)
            volume_id = int(volume_id[cts.argmax()])
            
            part = ParticleFragment(fragment_id=i,
                                    group_id=group_ids[i],
                                    interaction_id=inter_ids[i],
                                    image_id=entry,
                                    volume_id=volume_id,
                                    semantic_type=seg_label,
                                    index=p,
                                    points=point_cloud[p],
                                    depositions=depositions[p],
                                    is_primary=False)
            temp.append(part)

        # Label shower fragments as primaries and attach start_point
        shower_counter = 0
        for p in np.array(temp)[shower_mask]:
            is_primary = shower_frag_primary[shower_counter]
            p.is_primary = bool(is_primary)
            p.start_point = shower_start_points[shower_counter]
            # p.group_id = int(shower_group_pred[shower_counter])
            shower_counter += 1
        assert shower_counter == shower_frag_primary.shape[0]

        # Attach end_point to track fragments
        track_counter = 0
        for p in temp:
            if p.semantic_type == 1:
                # p.group_id = int(track_group_pred[track_counter])
                p.start_point = track_start_points[track_counter]
                p.end_point = track_end_points[track_counter]
                track_counter += 1
        # assert track_counter == track_group_pred.shape[0]

        # Apply fragment voxel cut
        out = []
        for p in temp:
            if p.size < self.min_particle_voxel_count:
                continue
            out.append(p)

        # Check primaries
        if self.only_primaries:
            out = [p for p in out if p.is_primary]

        if self.include_semantics is not None:
            out = [p for p in out if p.semantic_type in self.include_semantics]

        return out
    
    def _build_truth(self, entry, data: dict, result: dict):
        
        fragments = []

        labels = result['cluster_label_adapted'][entry]
        rescaled_input_charge = result['input_rescaled'][entry][:, VALUE_COL]
        fragment_ids = set(list(np.unique(labels[:, CLUST_COL]).astype(int)))

        for fid in fragment_ids:
            mask = labels[:, CLUST_COL] == fid

            semantic_type, counts = np.unique(labels[:, -1][mask].astype(int), 
                                              return_counts=True)
            if semantic_type.shape[0] > 1:
                if self.verbose:
                    print("Semantic Type of Fragment {} is not "\
                        "unique: {}, {}".format(fid,
                                                str(semantic_type),
                                                str(counts)))
                perm = counts.argmax()
                semantic_type = semantic_type[perm]
            else:
                semantic_type = semantic_type[0]

            points = labels[mask][:, COORD_COLS]
            size = points.shape[0]
            depositions = rescaled_input_charge[mask]
            depositions_MeV = labels[mask][:, VALUE_COL]
            voxel_indices = np.where(mask)[0]

            volume_id, cts = np.unique(labels[:, BATCH_COL][mask].astype(int), 
                                       return_counts=True)
            volume_id = int(volume_id[cts.argmax()])

            group_id, counts = np.unique(labels[:, GROUP_COL][mask].astype(int), 
                                         return_counts=True)
            if group_id.shape[0] > 1:
                if self.verbose:
                    print("Group ID of Fragment {} is not "\
                        "unique: {}, {}".format(fid,
                                                str(group_id),
                                                str(counts)))
                perm = counts.argmax()
                group_id = group_id[perm]
            else:
                group_id = group_id[0]

            interaction_id, counts = np.unique(labels[:, INTER_COL][mask].astype(int), 
                                               return_counts=True)
            if interaction_id.shape[0] > 1:
                if self.verbose:
                    print("Interaction ID of Fragment {} is not "\
                        "unique: {}, {}".format(fid,
                                                str(interaction_id),
                                                str(counts)))
                perm = counts.argmax()
                interaction_id = interaction_id[perm]
            else:
                interaction_id = interaction_id[0]


            is_primary, counts = np.unique(labels[:, PSHOW_COL][mask].astype(bool), 
                                           return_counts=True)
            if is_primary.shape[0] > 1:
                if self.verbose:
                    print("Primary label of Fragment {} is not "\
                        "unique: {}, {}".format(fid,
                                                str(is_primary),
                                                str(counts)))
                perm = counts.argmax()
                is_primary = is_primary[perm]
            else:
                is_primary = is_primary[0]

            part = TruthParticleFragment(fragment_id=fid, 
                                         group_id=group_id,
                                         interaction_id=interaction_id,
                                         semantic_type=semantic_type,
                                         image_id=entry,
                                         volume_id=volume_id,
                                         index=voxel_indices,
                                         points=points,
                                         depositions=depositions,
                                         depositions_MeV=depositions_MeV,
                                         is_primary=is_primary)

            fragments.append(part)
        return fragments


# --------------------------Helper functions---------------------------

def handle_empty_truth_particles(labels_noghost,  
                                mask_noghost, 
                                p, 
                                entry, 
                                verbose=False):
    """
    Function for handling true larcv::Particle instances with valid 
    true nonghost voxels but with no predicted nonghost voxels.
    
    Parameters
    ----------
    labels_noghost: np.ndarray
        Label information for true nonghost coordinates
    mask_noghost: np.ndarray
        True nonghost mask for this particle.
    p: larcv::Particle
        larcv::Particle object from particles_asis, containing truth
        information for this particle
    entry: int
        Image ID of this particle (for consistent TruthParticle attributes)
        
    Returns
    -------
    particle: TruthParticle
    """
    pid = int(p.id())
    pdg = PDG_TO_PID.get(p.pdg_code(), -1)
    is_primary = p.group_id() == p.parent_id()

    semantic_type, interaction_id, nu_id = -1, -1, -1
    coords, depositions, voxel_indices = np.empty((0,3)), np.array([]), np.array([])
    coords_noghost, depositions_noghost = np.empty((0,3)), np.array([])
    if np.count_nonzero(mask_noghost) > 0:
        coords_noghost = labels_noghost[mask_noghost][:, COORD_COLS]
        true_voxel_indices = np.where(mask_noghost)[0]
        depositions_noghost = labels_noghost[mask_noghost][:, VALUE_COL].squeeze()
        semantic_type, interaction_id, nu_id = get_truth_particle_labels(labels_noghost, 
                                                                        mask_noghost, 
                                                                        pid=pid, 
                                                                        verbose=verbose)
        volume_id, cts = np.unique(labels_noghost[:, BATCH_COL][mask_noghost].astype(int), 
                                    return_counts=True)
        volume_id = int(volume_id[cts.argmax()])
    particle = TruthParticle(group_id=pid,
                             interaction_id=interaction_id,
                             nu_id=nu_id,
                             volume_id=volume_id,
                             image_id=entry,
                             semantic_type=semantic_type, 
                             index=voxel_indices,
                             points=coords,
                             depositions=depositions,
                             depositions_MeV=np.empty(0, dtype=np.float32),
                             truth_index=true_voxel_indices,
                             truth_points=coords_noghost,
                             truth_depositions=np.empty(0, dtype=np.float32), #TODO
                             truth_depositions_MeV=depositions_noghost,
                             is_primary=is_primary,
                             pid=pdg,
                             particle_asis=p)
    # particle.p = np.array([p.px(), p.py(), p.pz()])
    # particle.fragments = []
    # particle.particle_asis = p
    # particle.nu_id = nu_id
    # particle.voxel_indices = voxel_indices

    particle.start_point = np.array([p.first_step().x(),
                                    p.first_step().y(),
                                    p.first_step().z()])

    if semantic_type == 1:
        particle.end_point = np.array([p.last_step().x(),
                                    p.last_step().y(),
                                    p.last_step().z()])
    return particle


def get_truth_particle_labels(labels, mask, pid=-1, verbose=False):
    """
    Helper function for fetching true particle labels from 
    voxel label array. 
    
    Parameters
    ----------
    labels: np.ndarray
        Predicted nonghost voxel label information
    mask: np.ndarray
        Voxel index mask
    pid: int, optional
        Unique id of this particle (for debugging)
    """
    semantic_type, sem_counts = np.unique(labels[mask][:, -1].astype(int), 
                                            return_counts=True)
    if semantic_type.shape[0] > 1:
        if verbose:
            print("Semantic Type of Particle {} is not "\
                "unique: {}, {}".format(pid,
                                        str(semantic_type),
                                        str(sem_counts)))
        perm = sem_counts.argmax()
        semantic_type = semantic_type[perm]
    else:
        semantic_type = semantic_type[0]
    
    interaction_id, int_counts = np.unique(labels[mask][:, 7].astype(int),
                                        return_counts=True)
    if interaction_id.shape[0] > 1:
        if verbose:
            print("Interaction ID of Particle {} is not "\
                "unique: {}".format(pid, str(interaction_id)))
        perm = int_counts.argmax()
        interaction_id = interaction_id[perm]
    else:
        interaction_id = interaction_id[0]

    nu_id, nu_counts = np.unique(labels[mask][:, 8].astype(int),
                                return_counts=True)
    if nu_id.shape[0] > 1:
        if verbose:
            print("Neutrino ID of Particle {} is not "\
                "unique: {}".format(pid, str(nu_id)))
        perm = nu_counts.argmax()
        nu_id = nu_id[perm]
    else:
        nu_id = nu_id[0]

    return semantic_type, interaction_id, nu_id


def match_points_to_particles(ppn_points : np.ndarray,
                              particles : List[Particle],
                              semantic_type=None, ppn_distance_threshold=2):
    """Function for matching ppn points to particles.

    For each particle, match ppn_points that have hausdorff distance
    less than <threshold> and inplace update particle.ppn_candidates

    If semantic_type is set to a class integer value,
    points will be matched to particles with the same
    predicted semantic type.

    Parameters
    ----------
    ppn_points : (N x 4 np.array)
        PPN point array with (coords, point_type)
    particles : list of <Particle> objects
        List of particles for which to match ppn points.
    semantic_type: int
        If set to an integer, only match ppn points with prescribed
        semantic type
    ppn_distance_threshold: int or float
        Maximum distance required to assign ppn point to particle.

    Returns
    -------
        None (operation is in-place)
    """
    if semantic_type is not None:
        ppn_points_type = ppn_points[ppn_points[:, 5] == semantic_type]
    else:
        ppn_points_type = ppn_points
        # TODO: Fix semantic type ppn selection

    ppn_coords = ppn_points_type[:, :3]
    for particle in particles:
        dist = cdist(ppn_coords, particle.points)
        matches = ppn_points_type[dist.min(axis=1) < ppn_distance_threshold]
        particle.ppn_candidates = matches.reshape(-1, 7)
