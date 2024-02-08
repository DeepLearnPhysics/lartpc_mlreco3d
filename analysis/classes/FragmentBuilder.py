import numpy as np
from typing import List
from larcv import larcv

from mlreco.utils.globals import (BATCH_COL,
                                  COORD_COLS,
                                  VALUE_COL,
                                  CLUST_COL,
                                  GROUP_COL,
                                  PART_COL,
                                  TRACK_SHP,
                                  SHAPE_COL,
                                  )
from .builders import DataBuilder
from analysis.classes import ParticleFragment, TruthParticleFragment
from analysis.classes.builders import get_truth_particle_labels

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
    def __init__(self, builder_cfg={}, convert_to_cm=False):
        self.cfg = builder_cfg
        self.allow_nodes         = self.cfg.get('allow_nodes', [0,2,3])
        self.min_voxel_cut       = self.cfg.get('min_voxel_cut', 10)
        self.only_primaries      = self.cfg.get('only_primaries', False)
        self.include_semantics   = self.cfg.get('include_semantics', None)
        self.attaching_threshold = self.cfg.get('attaching_threshold', 5.0)
        self.verbose             = self.cfg.get('verbose', False)
        self.convert_to_cm       = convert_to_cm

    def _build_reco(self, entry,
                    data: dict,
                    result: dict):

        if 'input_rescaled' in result:
            point_cloud = result['input_rescaled'][entry]
        elif 'input_data' in data:
            point_cloud = data['input_data'][entry]
        else:
            msg = "To build ParticleFragment objects from HDF5 data, need either "\
                "input_data inside data dictionary or input_rescaled inside"\
                " result dictionary."
            raise KeyError(msg)
        volume_labels = point_cloud[:, BATCH_COL]
        depositions = point_cloud[:, VALUE_COL]
        fragments = result['fragment_clusts'][entry]
        fragments_seg = result['fragment_seg'][entry]

        assert len(fragments_seg) == len(fragments)

        temp = []

        for i, p in enumerate(fragments):
            voxels = point_cloud[p][:, COORD_COLS]
            seg_label = fragments_seg[i]
            volume_id, cts = np.unique(volume_labels[p], return_counts=True)
            volume_id = int(volume_id[cts.argmax()])

            part = ParticleFragment(fragment_id=i,
                                    group_id=-1,
                                    interaction_id=-1,
                                    image_id=entry,
                                    volume_id=volume_id,
                                    semantic_type=seg_label,
                                    index=p,
                                    points=voxels,
                                    depositions=depositions[p],
                                    is_primary=False)
            temp.append(part)
            
        # Decorate Showers
        
        dec_showers = {}
            
        shower_mask = np.isin(fragments_seg, self.allow_nodes)
        
        if 'shower_fragment_node_pred' in result:
            dec_showers['shower_frag_primary'] = np.argmax(
                result['shower_fragment_node_pred'][entry], axis=1)

        if 'shower_fragment_start_points' in result:
            dec_showers['shower_start_points'] = result['shower_fragment_start_points'][entry][:, COORD_COLS]
        
        if 'shower_fragment_group_pred' in result:
            dec_showers['shower_group'] = result['shower_fragment_group_pred'][entry]

        # Label shower fragments as primaries and attach start_point
        shower_counter = 0
        for p in np.array(temp)[shower_mask]:
            if 'shower_frag_primary' in dec_showers:
                is_primary = dec_showers['shower_frag_primary'][shower_counter]
                p.is_primary = bool(is_primary)
            if 'shower_start_points' in dec_showers:
                p.start_point = dec_showers['shower_start_points'][shower_counter]
            if 'shower_group' in dec_showers:
                p.group_id = int(dec_showers['shower_group'][shower_counter])
            shower_counter += 1
        # assert shower_counter == dec_showers['shower_frag_primary'].shape[0]
        
        # Decorate Tracks
        
        dec_tracks = {}
        
        if 'track_fragment_start_points' in result:
            dec_tracks['track_start_points'] = result['track_fragment_start_points'][entry][:, COORD_COLS]
        if 'track_fragment_end_points' in result:
            dec_tracks['track_end_points'] = result['track_fragment_end_points'][entry][:, COORD_COLS]

        # Attach end_point to track fragments
        track_counter = 0
        for p in temp:
            if p.semantic_type == 1:
                if 'track_fragment_group_pred' in result:
                    p.group_id = int(result['track_fragment_group_pred'][entry][track_counter])
                if 'track_fragment_start_points' in result:
                    p.start_point = dec_tracks['track_start_points'][track_counter]
                if 'track_fragment_end_points' in result:
                    p.end_point = dec_tracks['track_end_points'][track_counter]
                track_counter += 1
        # assert track_counter == track_group_pred.shape[0]

        # Apply fragment voxel cut
        out = []
        for p in temp:
            if p.size < self.min_voxel_cut:
                continue
            out.append(p)

        # Check primaries
        if self.only_primaries:
            out = [p for p in out if p.is_primary]

        if self.include_semantics is not None:
            out = [p for p in out if p.semantic_type in self.include_semantics]

        return out
    
    
    def _build_truth(self,
                    entry: int,
                    data: dict,
                    result: dict, verbose=False) -> List[TruthParticleFragment]:
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

        if 'input_rescaled' in result:
            rescaled_charge = result['input_rescaled'][entry][:, VALUE_COL]
            coordinates     = result['input_rescaled'][entry][:, COORD_COLS]
        else:
            rescaled_charge = data['input_data'][entry][:, VALUE_COL]
            coordinates     = data['input_data'][entry][:, COORD_COLS]

        if 'energy_label' in data:
            energy_label = data['energy_label'][entry][:, VALUE_COL]
        else:
            energy_label = None

        meta            = data['meta'][0]
        if 'sed' in data:
            simE_deposits   = data['sed'][entry]
        else:
            simE_deposits   = None

        # For debugging
        voxel_counts = 0
        accounted_indices = []
        
        labels_nonghost = data['cluster_label'][entry]
        larcv_particles = data['particles_asis'][entry]
        particle_dict = {}
        for p in larcv_particles:
            particle_dict[p.id()] = p
        
        orphans = np.ones(labels_nonghost.shape[0]).astype(bool)
        
        fragment_ids = np.unique(labels_nonghost[:, CLUST_COL].astype(int))
        particle_ids = np.array([p.id() for p in data['particles_asis'][entry]]).astype(int)
    
        
        for i, fid in enumerate(fragment_ids):
            
            if fid == -1:
                continue
            
            mask_nonghost = labels_nonghost[:, CLUST_COL].astype(int) == fid
            # part_id = np.unique(labels_nonghost[mask_nonghost][:, PART_COL].astype(int))[0]

            if simE_deposits is not None:
                mask_sed      = simE_deposits[:, CLUST_COL].astype(int) == fid
                sed_index     = np.where(mask_sed)[0]
            else:
                mask_sed, sed_index = np.array([]), np.array([])
            if np.count_nonzero(mask_nonghost) <= 0:
                continue
            
            mask = labels[:, CLUST_COL].astype(int) == fid

            # 1. Check if current pid is one of the existing group ids
            # if id not in particle_ids:
            #     particle = handle_empty_truth_fragments(labels_nonghost,
            #                                             mask_nonghost,
            #                                             lpart,
            #                                             image_index,
            #                                             sed=simE_deposits,
            #                                             mask_sed=mask_sed)
            #     particle.id = len(out)
            #     particle.start_point = particle.first_step
            #     if particle.semantic_type == TRACK_SHP:
            #         particle.end_point = particle.last_step

            #     out.append(particle)
            #     continue

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
                                                     id=fid)
            semantic_type  = int(truth_labels[0])
            interaction_id = int(truth_labels[1])
            nu_id          = int(truth_labels[2])
            pid            = int(truth_labels[3])
            primary_id     = int(truth_labels[4])
            is_primary     = int(primary_id) == 1
            
            # Group ID
            group_id, group_counts = np.unique(labels_nonghost[mask][:, GROUP_COL].astype(int),
                                                return_counts=True)
            if group_id.shape[0] > 1:
                if verbose:
                    print("Group ID of TruthParticleFragment {} is not "\
                        "unique: {}".format(fid, str(group_id)))
                perm = group_counts.argmax()
                group_id = group_id[perm]
            else:
                group_id = group_id[0]

            # 3. Process particle start / end point labels

            truth_depositions_MeV = np.empty(0, dtype=np.float32)
            if energy_label is not None:
                truth_depositions_MeV = energy_label[mask_nonghost].squeeze()
                
            lpart = particle_dict.get(fid, larcv.Particle())

            particle = TruthParticleFragment(fragment_id=len(out),
                                     group_id=group_id,
                                     interaction_id=interaction_id,
                                     nu_id=nu_id,
                                    #  pid=pid,
                                     image_id=image_index,
                                     volume_id=volume_id,
                                     semantic_type=semantic_type,
                                     index=voxel_indices,
                                     points=coords,
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
                                     particle_asis=lpart)

            particle.start_point = particle.first_step
            if particle.semantic_type == TRACK_SHP:
                particle.end_point = particle.last_step

            out.append(particle)

        accounted_indices = np.hstack(accounted_indices).squeeze() if len(accounted_indices) else np.empty(0, dtype=np.int64)
        if verbose:
            print("All Voxels = {}, Accounted Voxels = {}".format(labels_nonghost.shape[0], voxel_counts))
            print("Orphaned Semantics = ", np.unique(labels_nonghost[orphans][:, SHAPE_COL], return_counts=True))
            print("Orphaned ClustIDs = ", np.unique(labels_nonghost[orphans][:, CLUST_COL], return_counts=True))
        
        return out
    
    
def handle_empty_truth_fragments(labels_noghost,
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

    particle = TruthParticleFragment(group_id=id,
                             interaction_id=interaction_id,
                             volume_id=volume_id,
                             image_id=entry,
                             nu_id=nu_id,
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
