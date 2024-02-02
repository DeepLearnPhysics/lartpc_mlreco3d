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
                                  VALUE_COL,
                                  INTER_COL,
                                  GROUP_COL,
                                  NU_COL,
                                  TRACK_SHP,
                                  SHAPE_COL,
                                  PGRP_COL,
                                  PID_COL)
from analysis.classes import (Particle,
                              TruthParticle,
                              Interaction,
                              TruthInteraction)
from analysis.classes.matching import group_particles_to_interactions_fn
from mlreco.utils.vertex import get_truth_vertex

# These attributes are computed based on the particles being loaded to
# each interaction, and they are computed at initialization.

SKIP_KEYS = [
    'is_principal_match', 'match', 'match_overlap',
    'num_particles', 'num_primaries', 'particle_counts', 'particle_ids',
    'primary_counts', 'size', 'topology',
    # TruthInteraction Attributes
    'truth_particle_counts', 'truth_primary_counts', 'truth_topology', 'children_counts'
]


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
    def __init__(self, convert_to_cm=False):
        self.convert_to_cm = convert_to_cm

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
            point_cloud = result['input_rescaled'][entry]
        elif 'input_data' in data:
            point_cloud = data['input_data'][entry]
        else:
            msg = "To build Particle objects from HDF5 data, need either "\
                "input_data inside data dictionary or input_rescaled inside"\
                " result dictionary."
            raise KeyError(msg)
        out  = []
        blueprints = result['particles'][entry]
        for i, bp in enumerate(blueprints):
            mask = bp['index']
            prepared_bp = copy.deepcopy(bp)

            match = prepared_bp.pop('match', [])
            match_overlap = prepared_bp.pop('match_overlap', [])

            assert len(match) == len(match_overlap)

            prepared_bp.pop('depositions_sum', None)
            group_id = prepared_bp.pop('id', -1)
            prepared_bp['group_id'] = group_id
            prepared_bp.update({
                'points': point_cloud[mask][:, COORD_COLS],
                'depositions': point_cloud[mask][:, VALUE_COL],
            })
            if 'input_rescaled_source' in result:
                prepared_bp['sources'] = result['input_rescaled_source'][entry][mask]

            particle = Particle(**prepared_bp)
            particle.pid = bp['pid']
            particle.is_primary = bp['is_primary']
            if len(match) > 0:
                particle.match_overlap = OrderedDict({
                    key : val for key, val in zip(match, match_overlap)})
            # assert particle.image_id == entry
            out.append(particle)

        return out


    def _load_truth(self, entry, data, result):
        out = []
        true_nonghost  = data['cluster_label'][0]
        pred_nonghost  = result['cluster_label_adapted'][0]
        blueprints     = result['truth_particles'][0]

        if 'energy_label' in data:
            energy_label = data['energy_label'][entry]
        else:
            energy_label = None

        if 'sed' in data:
            true_sed = data['sed'][0]
        else:
            true_sed = None
        for i, bp in enumerate(blueprints):
            mask      = bp['index']
            true_mask = bp['truth_index']
            sed_mask  = bp.get('sed_index', None)
            pasis_selected = None

            prepared_bp = copy.deepcopy(bp)

            group_id = prepared_bp.pop('id', -1)
            prepared_bp['group_id'] = group_id
            prepared_bp.pop('depositions_sum', None)
            if energy_label is not None:
                truth_depositions_MeV = energy_label[true_mask][:, VALUE_COL]
            else:
                truth_depositions_MeV = np.empty(0, dtype=np.float32)
            prepared_bp.update({
                'points': pred_nonghost[mask][:, COORD_COLS],
                'depositions': pred_nonghost[mask][:, VALUE_COL],
                'truth_points': true_nonghost[true_mask][:, COORD_COLS],
                'truth_depositions': true_nonghost[true_mask][:, VALUE_COL],
                'truth_depositions_MeV': truth_depositions_MeV,
                'particle_asis': pasis_selected
            })

            if (sed_mask is not None) and ('sed' in data):
                prepared_bp['sed_points'] = true_sed[sed_mask][:, COORD_COLS]
                prepared_bp['sed_depositions_MeV'] = true_sed[sed_mask][:, VALUE_COL]
            if 'input_rescaled_source' in result:
                prepared_bp['sources'] = result['input_rescaled_source'][0][mask]

            match = prepared_bp.pop('match', [])
            match_overlap = prepared_bp.pop('match_overlap', [])

            truth_particle = TruthParticle(**prepared_bp)
            truth_particle.load_larcv_attributes(prepared_bp)
            if len(match) > 0:
                truth_particle.match_overlap = OrderedDict({
                    key : val for key, val in zip(match, match_overlap)})
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
        if 'input_rescaled' in result:
            input_voxels     = result['input_rescaled'][entry]
        else:
            input_voxels     = data['input_data'][entry]

        if 'input_rescaled_source' in result:
            input_source     = result['input_rescaled_source'][entry]
        else:
            input_source     = np.empty((0,2), dtype=np.float32)

        volume_labels    = input_voxels[:, BATCH_COL]
        point_cloud      = input_voxels[:, COORD_COLS]
        depositions      = input_voxels[:, VALUE_COL]

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
                            points=np.ascontiguousarray(point_cloud[p]),
                            sources=input_source[p] if len(input_source) else input_source,
                            depositions=depositions[p],
                            volume_id=volume_id,
                            pid_scores=pid_scores[i],
                            primary_scores=primary_scores[i],
                            start_point = np.ascontiguousarray(particle_start_points[i], dtype=np.float32),
                            end_point = np.ascontiguousarray(particle_end_points[i], dtype=np.float32))
            out.append(part)

        return out

    def _build_truth(self,
                    entry: int,
                    data: dict,
                    result: dict, verbose=False) -> List[TruthParticle]:
        """
        Returns
        -------
        out : List[TruthParticle]
            list of true TruthParticle instances of length equal to the
            batch size.
        """

        out = []
        image_index     = data['index'][entry]
        if 'cluster_label_adapted' in result:
            labels = result['cluster_label_adapted'][entry]
        elif 'cluster_label' in data:
            labels = data['cluster_label'][entry]
        else:
            msg = "To build TruthParticle objects from HDF5 data, need either "\
                "cluster_label inside data dictionary or cluster_label_adapted inside"\
                " result dictionary."
            raise KeyError(msg)
        particle_ids    = set(list(np.unique(labels[:, GROUP_COL]).astype(int)))
        labels_nonghost = data['cluster_label'][entry]
        larcv_particles = data['particles_asis'][entry]
        if 'input_rescaled' in result:
            rescaled_charge = result['input_rescaled'][entry][:, VALUE_COL]
            coordinates     = result['input_rescaled'][entry][:, COORD_COLS]
        else:
            rescaled_charge = data['input_data'][entry][:, VALUE_COL]
            coordinates     = data['input_data'][entry][:, COORD_COLS]

        if 'input_rescaled_source' in result:
            input_source     = result['input_rescaled_source'][entry]
        else:
            input_source     = np.empty((0,2), dtype=np.float32)

        if 'energy_label' in data:
            energy_label = data['energy_label'][entry][:, VALUE_COL]
        else:
            energy_label = None

        meta            = data['meta'][0]
        if 'sed' in data:
            simE_deposits   = data['sed'][entry]
        else:
            simE_deposits   = None

        # point_labels   = data['point_labels'][entry]
        # unit_convert = lambda x: pixel_to_cm_1d(x, meta) if self.convert_to_cm == True else x

        # For debugging
        voxel_counts = 0
        accounted_indices = []
        orphans = np.ones(labels_nonghost.shape[0]).astype(bool)

        for i, lpart in enumerate(larcv_particles):
            id = int(lpart.id())
            if lpart.id() != lpart.group_id():
                continue
            mask_nonghost = labels_nonghost[:, GROUP_COL].astype(int) == id
            if simE_deposits is not None:
                mask_sed      = simE_deposits[:, GROUP_COL].astype(int) == id
                sed_index     = np.where(mask_sed)[0]
            else:
                mask_sed, sed_index = np.array([]), np.array([])
            if np.count_nonzero(mask_nonghost) <= 0:
                continue  # Skip larcv particles with no true depositions
            # 1. Check if current pid is one of the existing group ids
            if id not in particle_ids:
                particle = handle_empty_truth_particles(labels_nonghost,
                                                        mask_nonghost,
                                                        lpart,
                                                        image_index,
                                                        sed=simE_deposits,
                                                        mask_sed=mask_sed)
                particle.id = len(out)
                particle.start_point = particle.first_step
                if particle.semantic_type == TRACK_SHP:
                    particle.end_point = particle.last_step

                out.append(particle)
                continue

            # 1. Process voxels
            mask = labels[:, GROUP_COL].astype(int) == id

            coords              = coordinates[mask]
            voxel_indices       = np.where(mask)[0]
            # depositions_MeV     = labels[mask][:, VALUE_COL]
            depositions_MeV     = None # TODO: Fix to get MeVs from adapted energy labels?
            depositions         = rescaled_charge[mask] # Will be in ADC
            coords_noghost      = labels_nonghost[mask_nonghost][:, COORD_COLS]
            true_voxel_indices  = np.where(mask_nonghost)[0]
            voxel_counts        += true_voxel_indices.shape[0]
            accounted_indices.append(true_voxel_indices)
            orphans[true_voxel_indices] = False
            depositions_noghost = labels_nonghost[mask_nonghost][:, VALUE_COL].squeeze()

            volume_labels       = labels_nonghost[mask_nonghost][:, BATCH_COL]
            volume_id, cts      = np.unique(volume_labels, return_counts=True)
            volume_id           = int(volume_id[cts.argmax()])

            if simE_deposits is not None:
                sed_points          = simE_deposits[mask_sed][:, COORD_COLS].astype(np.float32)
                sed_depositions_MeV = simE_deposits[mask_sed][:, VALUE_COL].astype(np.float32)
            else:
                sed_points          = np.empty((0,3), dtype=np.float32)
                sed_depositions_MeV = np.empty(0, dtype=np.float32)

            # 2. Process particle-level labels
            truth_labels = get_truth_particle_labels(labels_nonghost,
                                                     mask_nonghost,
                                                     id=id)
            semantic_type  = int(truth_labels[0])
            interaction_id = int(truth_labels[1])
            nu_id          = int(truth_labels[2])
            pid            = int(truth_labels[3])
            primary_id     = int(truth_labels[4])
            is_primary     = int(primary_id) == 1

            # 3. Process particle start / end point labels

            truth_depositions_MeV = np.empty(0, dtype=np.float32)
            if energy_label is not None:
                truth_depositions_MeV = energy_label[mask_nonghost].squeeze()

            particle = TruthParticle(#group_id=id,
                                     group_id=len(out),
                                     interaction_id=interaction_id,
                                     nu_id=nu_id,
                                     pid=pid,
                                     image_id=image_index,
                                     volume_id=volume_id,
                                     semantic_type=semantic_type,
                                     index=voxel_indices,
                                     points=coords,
                                     sources=input_source[mask] if len(input_source) else input_source,
                                     depositions=depositions,
                                     depositions_MeV=np.empty(0, dtype=np.float32),
                                     truth_index=true_voxel_indices,
                                     truth_points=coords_noghost,
                                     truth_depositions=depositions_noghost, # TODO
                                     truth_depositions_MeV=truth_depositions_MeV,
                                     sed_index=sed_index.astype(np.int64),
                                     sed_points=sed_points,
                                     sed_depositions_MeV=sed_depositions_MeV,
                                     is_primary=bool(is_primary),
                                    #  pid=pdg,
                                     particle_asis=lpart)

            particle.start_point = particle.first_step
            if particle.semantic_type == TRACK_SHP:
                particle.end_point = particle.last_step

            out.append(particle)

        accounted_indices = np.hstack(accounted_indices).squeeze() if len(accounted_indices) else np.empty(0, dtype=np.int64)
        if verbose:
            print("All Voxels = {}, Accounted Voxels = {}".format(labels_nonghost.shape[0], voxel_counts))
            print("Orphaned Semantics = ", np.unique(labels_nonghost[orphans][:, SHAPE_COL], return_counts=True))
            print("Orphaned GroupIDs = ", np.unique(labels_nonghost[orphans][:, GROUP_COL], return_counts=True))

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
    def __init__(self, convert_to_cm=False):
        self.convert_to_cm = convert_to_cm

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
            
            info = copy.deepcopy(bp)
            info['interaction_id'] = info.pop('id', -1)
            
            for key in bp:
                if key in SKIP_KEYS:
                    info.pop(key)
            if use_particles:
                particles = []
                for p in result['particles'][0]:
                    if p.interaction_id == bp['id']:
                        particles.append(p)
                ia = Interaction.from_particles(particles,
                                                verbose=False, **info)
            else:
                mask = bp['index']
                info.update({
                    'index': mask,
                    'points': point_cloud[mask][:, COORD_COLS],
                    'depositions': point_cloud[mask][:, VALUE_COL]
                })
                if 'input_rescaled_source' in result:
                    info['sources'] = result['input_rescaled_source'][0][mask]
                ia = Interaction(**info)

            # Handle matches
            match_overlap = OrderedDict({i: val for i, val in zip(bp['match'], bp['match_overlap'])})
            ia._match_overlap = match_overlap
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
        true_nonghost = data['cluster_label'][entry]
        pred_nonghost = result['cluster_label_adapted'][entry]

        if 'energy_label' in data:
            energy_label = data['energy_label'][entry]
        else:
            energy_label = None

        out = []
        blueprints = result['truth_interactions'][entry]
        use_particles = 'truth_particles' in result

        if not use_particles:
            msg = "Loading TruthInteractions without building TruthParticles. "\
            "This means TruthInteraction.particles will be empty!"
            print(msg)

        for i, bp in enumerate(blueprints):

            info = copy.deepcopy(bp)
            info['interaction_id'] = info.pop('id', -1)
            for key in bp:
                if key in SKIP_KEYS:
                    info.pop(key)
            if use_particles:
                particles = []
                for p in result['truth_particles'][entry]:
                    if p.interaction_id == bp['id']:
                        particles.append(p)
                ia = TruthInteraction.from_particles(particles,
                                                     verbose=False,
                                                     **info)
                ia.load_larcv_neutrino(info)
            else:
                mask = bp['index']
                true_mask = bp['truth_index']
                if energy_label is not None:
                    truth_depositions_MeV = energy_label[true_mask][:, VALUE_COL]
                else:
                    truth_depositions_MeV = np.empty(0, dtype=np.float32)
                info.update({
                    'index': mask,
                    'truth_index': true_mask,
                    'points': pred_nonghost[mask][:, COORD_COLS],
                    'depositions': pred_nonghost[mask][:, VALUE_COL],
                    'truth_points': true_nonghost[true_mask][:, COORD_COLS],
                    'truth_depositions': true_nonghost[true_mask][:, VALUE_COL],
                    'truth_depositions_MeV': truth_depositions_MeV
                })
                if 'input_rescaled_source' in result:
                    info['sources'] = result['input_rescaled_source'][0][mask]
                ia = TruthInteraction(**info)
                ia.id = len(out)
                ia.load_larcv_neutrino(info)
                
            # Handle matches
            match_overlap = OrderedDict({i: val for i, val in zip(bp['match'], bp['match_overlap'])})
            ia._match_overlap = match_overlap

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
            if ia.truth_id in vertices:
                ia.truth_vertex = vertices[ia.truth_id]

            if 'neutrinos' in data and ia.nu_id > -1:
                neutrinos = data['neutrinos'][entry]
                nu = neutrinos[ia.nu_id]
                ia.is_neutrino = True
                ia.register_larcv_neutrino(nu)

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
            vtx = get_truth_vertex(data['cluster_label'],
                                   data_idx=entry,
                                   inter_idx=inter_idx)
            mask    = data['cluster_label'][entry][:, INTER_COL].astype(int) == inter_idx
            points  = data['cluster_label'][entry][:, COORD_COLS]
            new_vtx = points[mask][np.linalg.norm(points[mask] - vtx, axis=1).argmin()]
            out[inter_idx] = new_vtx
        return out


# --------------------------Helper functions---------------------------

def handle_empty_truth_particles(labels_noghost,
                                 mask_noghost,
                                 p,
                                 entry,
                                 verbose=False,
                                 sed=None,
                                 mask_sed=None):
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
    id = int(p.id())
    is_primary = False

    semantic_type, interaction_id, nu_id, primary_id, pid = -1, -1, -1, -1, -1
    coords, depositions, voxel_indices = np.empty((0,3)), np.array([]), np.array([])
    coords_noghost, depositions_noghost = np.empty((0,3)), np.array([])
    sed_index, sed_points, sed_depositions_MeV = np.array([]), np.empty((0,3)), np.array([])
    if np.count_nonzero(mask_noghost) > 0:
        if sed is not None:
            sed_points = sed[mask_sed][:, COORD_COLS]
            sed_index = np.where(mask_sed)[0]
            sed_depositions_MeV = sed[mask_sed][:, VALUE_COL]
        coords_noghost = labels_noghost[mask_noghost][:, COORD_COLS]
        true_voxel_indices = np.where(mask_noghost)[0]
        depositions_noghost = labels_noghost[mask_noghost][:, VALUE_COL].squeeze()
        truth_labels = get_truth_particle_labels(labels_noghost,
                                                 mask_noghost,
                                                 id=id,
                                                 verbose=verbose)

        semantic_type  = int(truth_labels[0])
        interaction_id = int(truth_labels[1])
        nu_id          = int(truth_labels[2])
        pid            = int(truth_labels[3])
        primary_id     = int(truth_labels[4])
        is_primary     = bool(int(primary_id) == 1)

        volume_id, cts = np.unique(labels_noghost[:, BATCH_COL][mask_noghost].astype(int),
                                    return_counts=True)
        volume_id = int(volume_id[cts.argmax()])

    particle = TruthParticle(group_id=id,
                             interaction_id=interaction_id,
                             nu_id=nu_id,
                             pid=pid,
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
                             sed_index=sed_index.astype(np.int64),
                             sed_points=sed_points.astype(np.float32),
                             sed_depositions_MeV=sed_depositions_MeV.astype(np.float32),
                             particle_asis=p,
                             start_point=-np.ones(3, dtype=np.float32),
                             end_point=-np.ones(3, dtype=np.float32))
    return particle


def get_truth_particle_labels(labels_nonghost, mask, id=-1, verbose=False):
    """
    Helper function for fetching true particle labels from
    voxel label array.

    Parameters
    ----------
    labels: np.ndarray
        Predicted nonghost voxel label information
    mask: np.ndarray
        Voxel index mask
    id: int, optional
        Unique id of this particle (for debugging)
    """

    # Semantic Type
    semantic_type, sem_counts = np.unique(labels_nonghost[mask][:, SHAPE_COL].astype(int),
                                            return_counts=True)
    if semantic_type.shape[0] > 1:
        if verbose:
            print("Semantic Type of TruthParticle {} is not "\
                "unique: {}, {}".format(id,
                                        str(semantic_type),
                                        str(sem_counts)))
        perm = sem_counts.argmax()
        semantic_type = semantic_type[perm]
    else:
        semantic_type = semantic_type[0]

    # Interaction ID
    interaction_id, int_counts = np.unique(labels_nonghost[mask][:, INTER_COL].astype(int),
                                        return_counts=True)
    if interaction_id.shape[0] > 1:
        if verbose:
            print("Interaction ID of TruthParticle {} is not "\
                "unique: {}".format(id, str(interaction_id)))
        perm = int_counts.argmax()
        interaction_id = interaction_id[perm]
    else:
        interaction_id = interaction_id[0]

    # Neutrino ID
    nu_id, nu_counts = np.unique(labels_nonghost[mask][:, NU_COL].astype(int),
                                return_counts=True)
    if nu_id.shape[0] > 1:
        if verbose:
            print("Neutrino ID of TruthParticle {} is not "\
                "unique: {}".format(id, str(nu_id)))
        perm = nu_counts.argmax()
        nu_id = nu_id[perm]
    else:
        nu_id = nu_id[0]

    # Primary ID
    primary_id, primary_counts = np.unique(labels_nonghost[mask][:, PGRP_COL].astype(int),
                                return_counts=True)
    if primary_id.shape[0] > 1:
        if verbose:
            print("Primary ID of TruthParticle {} is not "\
                "unique: {}".format(id, str(primary_id)))
        perm = primary_counts.argmax()
        primary_id = primary_id[perm]
    else:
        primary_id = primary_id[0]

    # PID
    pid, pid_counts = np.unique(labels_nonghost[mask][:, PID_COL].astype(int),
                                return_counts=True)
    if pid.shape[0] > 1:
        if verbose:
            print("Primary ID of TruthParticle {} is not "\
                "unique: {}".format(id, str(pid)))
        perm = pid_counts.argmax()
        pid = pid[perm]
    else:
        pid = pid[0]

    return semantic_type, interaction_id, nu_id, pid, primary_id


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

def pixel_to_cm_1d(vec, meta):
    out = np.zeros_like(vec)
    out[0] = meta[0] + meta[6] * vec[0]
    out[1] = meta[1] + meta[7] * vec[1]
    out[2] = meta[2] + meta[8] * vec[2]
    return out
