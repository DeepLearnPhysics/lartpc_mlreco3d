from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist

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
from analysis.classes.particle_utils import group_particles_to_interactions_fn
from mlreco.utils.vertex import get_vertex
from mlreco.utils.gnn.cluster import get_cluster_label

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
            entities = self._build_true(entry, data, result)
        elif mode == 'reco':
            entities = self._build_reco(entry, data, result)
        else:
            raise ValueError(f"Particle builder mode {mode} not supported!")
        
        return entities
        
    @abstractmethod
    def _build_true(self, entry, data: dict, result: dict):
        raise NotImplementedError
    
    @abstractmethod
    def _build_reco(self, entry, data: dict, result: dict):
        raise NotImplementedError


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
        volume_labels    = result['input_rescaled'][entry][:, BATCH_COL]
        point_cloud      = result['input_rescaled'][entry][:, COORD_COLS]
        depositions      = result['input_rescaled'][entry][:, 4]
        particles        = result['particle_clusts'][entry]
        particle_seg     = result['particle_seg'][entry]

        particle_start_points = result['particle_start_points'][entry][:, COORD_COLS]
        particle_end_points   = result['particle_end_points'][entry][:, COORD_COLS]
        inter_ids             = result['particle_group_pred'][entry]

        type_logits            = result['particle_node_pred_type'][entry]
        primary_logits         = result['particle_node_pred_vtx'][entry]

        pid_scores     = softmax(type_logits, axis=1)
        primary_scores = softmax(primary_logits, axis=1)

        for i, p in enumerate(particles):
            voxels = point_cloud[p]
            volume_id, cts = np.unique(volume_labels[p], return_counts=True)
            volume_id = int(volume_id[cts.argmax()])
            seg_label = particle_seg[i]
            pid = np.argmax(pid_scores[i])
            if seg_label == 2 or seg_label == 3:
                pid = 1
            interaction_id = inter_ids[i]
            part = Particle(voxels,
                            i,
                            seg_label, 
                            interaction_id,
                            entry,
                            pid=pid,
                            voxel_indices=p,
                            depositions=depositions[p],
                            volume=volume_id,
                            primary_scores=primary_scores[i],
                            pid_scores=pid_scores[i])

            part.startpoint = particle_start_points[i]
            part.endpoint   = particle_end_points[i]

            out.append(part)

        return out
    
    def _build_true(self, 
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

        labels          = result['cluster_label_adapted'][entry]
        labels_nonghost = data['cluster_label'][entry]
        larcv_particles = data['particles_asis'][entry]
        rescaled_charge = result['input_rescaled'][entry][:, 4]
        particle_ids    = set(list(np.unique(labels[:, 6]).astype(int)))
        coordinates     = result['input_rescaled'][entry][:, COORD_COLS]
                              

        for i, lpart in enumerate(larcv_particles):
            id = int(lpart.id())
            pdg = PDG_TO_PID.get(lpart.pdg_code(), -1)
            is_primary = lpart.group_id() == lpart.parent_id()
            mask_nonghost = labels_nonghost[:, 6].astype(int) == id
            if np.count_nonzero(mask_nonghost) <= 0:
                continue  # Skip larcv particles with no true depositions
            # 1. Check if current pid is one of the existing group ids
            if id not in particle_ids:
                particle = handle_empty_true_particles(labels_nonghost, 
                                                       mask_nonghost, 
                                                       lpart, 
                                                       entry)
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
            depositions_noghost = labels_nonghost[mask_nonghost][:, VALUE_COL].squeeze()

            volume_labels       = labels_nonghost[mask_nonghost][:, BATCH_COL]
            volume_id, cts      = np.unique(volume_labels, return_counts=True)
            volume_id           = int(volume_id[cts.argmax()])

            # if lpart.pdg_code() not in PDG_TO_PID:
            #     continue
            # exclude_ids = self._apply_true_voxel_cut(entry)
            # if pid in exclude_ids:
            #     # Skip this particle if its below the voxel minimum requirement
            #     continue
    
            # 2. Process particle-level labels
            semantic_type, int_id, nu_id = get_true_particle_labels(labels, 
                                                                    mask, 
                                                                    pid=id)
            
            particle = TruthParticle(coords,
                                     id,
                                     semantic_type, 
                                     int_id, 
                                     entry,
                                     particle_asis=lpart,
                                     coords_noghost=coords_noghost,
                                     depositions_noghost=depositions_noghost,
                                     depositions_MeV=depositions_MeV,
                                     nu_id=nu_id,
                                     voxel_indices=voxel_indices,
                                     depositions=depositions,
                                     volume=volume_id,
                                     is_primary=is_primary,
                                     pid=pdg)

            particle.p = np.array([lpart.px(), lpart.py(), lpart.pz()])
            particle.pmag = np.linalg.norm(particle.p)
            if particle.pmag > 0:
                particle.direction = particle.p / particle.pmag

            particle.startpoint = np.array([lpart.first_step().x(),
                                            lpart.first_step().y(),
                                            lpart.first_step().z()])

            if semantic_type == 1:
                particle.endpoint = np.array([lpart.last_step().x(),
                                              lpart.last_step().y(),
                                              lpart.last_step().z()])
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
        particles = result['Particles'][entry]
        out = group_particles_to_interactions_fn(particles, 
                                                 get_nu_id=True, 
                                                 mode='pred')
        return out
    
    def _build_true(self, entry: int, data: dict, result: dict) -> List[TruthInteraction]:
        particles = result['TruthParticles'][entry]
        out = group_particles_to_interactions_fn(particles, 
                                                 get_nu_id=True, 
                                                 mode='truth')
        
        out = self.decorate_true_interactions(entry, data, out)
        return out
    
    def build_true_using_particles(self, entry, data, particles):
        out = group_particles_to_interactions_fn(particles, 
                                                 get_nu_id=True, 
                                                 mode='truth')
        out = self.decorate_true_interactions(entry, data, out)
        return out
    
    def decorate_true_interactions(self, entry, data, interactions):
        """
        Helper function for attaching additional information to
        TruthInteraction instances. 
        """
        vertices = self.get_true_vertices(entry, data)
        for ia in interactions:
            if ia.id in vertices:
                ia.vertex = vertices[ia.id]

            if 'neutrino_asis' in data and ia.nu_id == 1:
                # assert 'particles_asis' in data_blob
                # particles = data_blob['particles_asis'][i]
                neutrinos = data['neutrino_asis'][entry]
                if len(neutrinos) > 1 or len(neutrinos) == 0: continue
                nu = neutrinos[0]
                # Get larcv::Particle objects for each
                # particle of the true interaction
                # true_particles = np.array(particles)[np.array([p.id for p in true_int.particles])]
                # true_particles_track_ids = [p.track_id() for p in true_particles]
                # for nu in neutrinos:
                #     if nu.mct_index() not in true_particles_track_ids: continue
                ia.nu_info = {
                    'nu_interaction_type': nu.interaction_type(),
                    'nu_interaction_mode': nu.interaction_mode(),
                    'nu_current_type': nu.current_type(),
                    'nu_energy_init': nu.energy_init()
                }
        return interactions
        
    def get_true_vertices(self, entry, data: dict):
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
        self.allow_nodes = self.cfg.get('allow_nodes', [0,2,3])
        self.min_particle_voxel_count = self.cfg.get('min_particle_voxel_cut', -1)
        self.only_primaries = self.cfg.get('only_primaries', False)
        self.include_semantics = self.cfg.get('include_semantics', None)
        self.attaching_threshold = self.cfg.get('attaching_threshold', 5.0)
        self.verbose = self.cfg.get('verbose', False)

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

        shower_startpoints = result['shower_fragment_start_points'][entry][:, COORD_COLS]
        track_startpoints = result['track_fragment_start_points'][entry][:, COORD_COLS]
        track_endpoints = result['track_fragment_end_points'][entry][:, COORD_COLS]

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
            
            part = ParticleFragment(voxels,
                            i, seg_label,
                            group_id=group_ids[i],
                            interaction_id=inter_ids[i],
                            image_id=entry,
                            voxel_indices=p,
                            depositions=depositions[p],
                            is_primary=False,
                            pid_conf=-1,
                            alias='Fragment',
                            volume=volume_id)
            temp.append(part)

        # Label shower fragments as primaries and attach startpoint
        shower_counter = 0
        for p in np.array(temp)[shower_mask]:
            is_primary = shower_frag_primary[shower_counter]
            p.is_primary = bool(is_primary)
            p.startpoint = shower_startpoints[shower_counter]
            # p.group_id = int(shower_group_pred[shower_counter])
            shower_counter += 1
        assert shower_counter == shower_frag_primary.shape[0]

        # Attach endpoint to track fragments
        track_counter = 0
        for p in temp:
            if p.semantic_type == 1:
                # p.group_id = int(track_group_pred[track_counter])
                p.startpoint = track_startpoints[track_counter]
                p.endpoint = track_endpoints[track_counter]
                track_counter += 1
        # assert track_counter == track_group_pred.shape[0]

        # Apply fragment voxel cut
        out = []
        for p in temp:
            if p.points.shape[0] < self.min_particle_voxel_count:
                continue
            out.append(p)

        # Check primaries and assign ppn points
        if self.only_primaries:
            out = [p for p in out if p.is_primary]

        if self.include_semantics is not None:
            out = [p for p in out if p.semantic_type in self.include_semantics]

        return out
    
    def _build_true(self, entry, data: dict, result: dict):
        
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

            part = TruthParticleFragment(points,
                                         fid, 
                                         semantic_type,
                                         interaction_id,
                                         group_id,
                                         image_id=entry,
                                         voxel_indices=voxel_indices,
                                         depositions=depositions,
                                         depositions_MeV=depositions_MeV,
                                         is_primary=is_primary,
                                         volume=volume_id,
                                         alias='Fragment')

            fragments.append(part)
        return fragments


# --------------------------Helper functions---------------------------

def handle_empty_true_particles(labels_noghost,  
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
    coords, depositions, voxel_indices = np.array([]), np.array([]), np.array([])
    coords_noghost, depositions_noghost = np.array([]), np.array([])
    if np.count_nonzero(mask_noghost) > 0:
        coords_noghost = labels_noghost[mask_noghost][:, COORD_COLS]
        depositions_noghost = labels_noghost[mask_noghost][:, VALUE_COL].squeeze()
        semantic_type, interaction_id, nu_id = get_true_particle_labels(labels_noghost, 
                                                                        mask_noghost, 
                                                                        pid=pid, 
                                                                        verbose=verbose)
        volume_id, cts = np.unique(labels_noghost[:, BATCH_COL][mask_noghost].astype(int), 
                                    return_counts=True)
        volume_id = int(volume_id[cts.argmax()])
    particle = TruthParticle(coords,
                                pid,
                                semantic_type, 
                                interaction_id, 
                                entry,
                                particle_asis=p,
                                coords_noghost=coords_noghost,
                                depositions_noghost=depositions_noghost,
                                depositions_MeV=np.array([]),
                                nu_id=nu_id,
                                voxel_indices=voxel_indices,
                                depositions=depositions,
                                volume=volume_id,
                                is_primary=is_primary,
                                pid=pdg)
    particle.p = np.array([p.px(), p.py(), p.pz()])
    # particle.fragments = []
    # particle.particle_asis = p
    # particle.nu_id = nu_id
    # particle.voxel_indices = voxel_indices

    particle.startpoint = np.array([p.first_step().x(),
                                    p.first_step().y(),
                                    p.first_step().z()])

    if semantic_type == 1:
        particle.endpoint = np.array([p.last_step().x(),
                                    p.last_step().y(),
                                    p.last_step().z()])
    return particle


def get_true_particle_labels(labels, mask, pid=-1, verbose=False):
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