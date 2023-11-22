import numpy as np
from scipy.spatial.distance import cdist

from mlreco.utils.globals import COORD_COLS, TRACK_SHP
from mlreco.utils.geometry import Geometry
from mlreco.utils.numba_local import farthest_pair
from mlreco.utils.gnn.cluster import cluster_direction

from analysis.classes import TruthParticle, Interaction, TruthInteraction
from analysis.post_processing import PostProcessor


class CathodeCrosserProcessor(PostProcessor):
    '''
    Find particles that cross the cathode of a LArTPC module that is divided
    into two TPCs. It might manifest itself into two forms:
    - If the particle is ~in-time, it will be a single particle, with
      potentially a small break/offset in the center
    - If the particle is sigificantly out-of-time, a cathode crosser will
      be composed of two distinct reconstructed particle objects
    '''
    name = 'find_cathode_crossers'
    data_cap_opt = ['input_data']
    result_cap = ['particles', 'interactions']
    result_cap_opt = ['input_rescaled', 'cluster_label_adapted',
            'truth_particles', 'truth_interactions']

    def __init__(self,
                 crossing_point_tolerance,
                 offset_tolerance,
                 angle_tolerance,
                 adjust_crossers=True,
                 merge_crossers=True,
                 detector=None,
                 boundary_file=None,
                 source_file=None,
                 truth_point_mode='points',
                 run_mode='both'):
        '''
        Initialize the cathode crosser finder algorithm

        Parameters
        ----------
        crossing_point_tolerance : float
            Maximum allowed distance in the cathode plane (in cm) between two
            fragments of a cathode crosser to be considered compatible
        offset_tolerance
            Maximum allowed discrepancy between end-point to cathode offsets of
            two fragments of a cathode crosser to be considered compatible
        angle_tolerance : float
            Maximum allowed angle (in radians) between the directions of two
            fragments of a cathode crosser to be considered compatible
        adjust_crossers : bool, default True
            If True, shifts existing cathode crossers to fix the small breaks
            that may exist at the level of the cathode
        merge_crossers : bool, default True
            If True, look for tracks that have been broken up at the cathode
            and merge them into one particle
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        source_file : str, optional
            Path to a detector source file. Supersedes `detector` if set
        '''
        # Initialize the parent class
        super().__init__(run_mode, truth_point_mode)

        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file, source_file)

        # Store the matching parameters
        self.crossing_point_tolerance = crossing_point_tolerance
        self.offset_tolerance = offset_tolerance
        self.angle_tolerance = angle_tolerance
        self.adjust_crossers = adjust_crossers
        self.merge_crossers = merge_crossers

    def process(self, data_dict, result_dict):
        '''
        Find cathode crossing particles in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over particle types
        update_dict = {}
        for k in self.part_keys:
            # Find crossing particles already merged by the reconstruction
            truth = 'truth' in k
            prefix = 'truth_' if truth else ''
            candidate_mask = np.zeros(len(result_dict[k]), dtype=bool)
            for i, p in enumerate(result_dict[k]):
                # Only bother to look for tracks that cross the cathode
                if p.semantic_type != TRACK_SHP:
                    continue

                # Make sure the particle coordinates are expressed in cm
                self.check_units(p)

                # Get point coordinates
                points = self.get_points(p)
                if not len(points):
                    continue
                assert len(p.sources), \
                        'Cannot identify cathode crossers without sources'

                # If the particle is composed of points from multiple
                # contributing TPCs in the same module, it is a cathode crosser
                modules, tpcs = self.geo.get_contributors(p.sources)
                p.is_ccrosser = len(np.unique(tpcs)) > len(np.unique(modules))
                candidate_mask[i] = not p.is_ccrosser

                # Now measure the gap at the cathode, correct if requested
                # TODO: handle particles that cross a cathode in at least one
                # module but not all of them or cross multiple cathodes
                if p.is_ccrosser and self.adjust_crossers and len(tpcs) == 2:
                    # Adjust positions
                    self.adjust_positions(result_dict, i, truth=truth)

            # If we do not want to merge broken crossers, our job here is done
            if not self.merge_crossers:
                continue

            # Try to find compatible tracks
            candidate_ids = np.where(candidate_mask)[0]
            i = 0
            while i < len(candidate_ids):
                # Get the first particle and its properties
                ci = candidate_ids[i]
                pi = result_dict[k][ci]
                end_points_i = np.vstack([pi.start_point, pi.end_point])
                end_dirs_i = np.vstack([pi.start_dir, pi.end_dir])

                # Check that the particle lives in one TPC
                modules_i, tpcs_i = self.geo.get_contributors(pi.sources)
                if len(tpcs_i) != 1:
                    i += 1
                    continue

                # Get the cathode position, drift axis and cathode plane axes
                daxis, cpos = self.geo.cathodes[modules_i[0]]
                caxes = np.array([i for i in range(3) if i != daxis])

                # Store the distance of the particle to the cathode
                tpc_offset = self.geo.get_min_tpc_offset(end_points_i, \
                        modules_i[0], tpcs_i[0])[daxis]
                cdists = end_points_i[:,daxis] - tpc_offset - cpos

                # Loop over other tracks
                j = i + 1
                while j < len(candidate_ids):

                    # Get the second particle object and its properties
                    cj = candidate_ids[j]
                    pj = result_dict[k][cj]
                    end_points_j = np.vstack([pj.start_point, pj.end_point])
                    end_dirs_j = np.vstack([pj.start_dir, pj.end_dir])

                    # Check that the particles live in TPCs of one module
                    modules_j, tpcs_j = self.geo.get_contributors(pj.sources)
                    if len(tpcs_j) != 1 or modules_i[0] != modules_j[0] \
                            or tpcs_i[0] == tpcs_j[0]:
                        j += 1
                        continue

                    # Check if the two particles stop at roughly the same
                    # position in the plane of the cathode
                    compat = True
                    dist_mat = cdist(end_points_i[:, caxes],
                            end_points_j[:, caxes])
                    argmin = np.argmin(dist_mat)
                    pair_i, pair_j = np.unravel_index(argmin, (2, 2))
                    compat &= dist_mat[pair_i, pair_j] \
                            < self.crossing_point_tolerance

                    # Check if the offset of the two particles w.r.t. to the
                    # cathode is compatible
                    offset_i = end_points_i[pair_i, daxis] - cpos
                    offset_j = end_points_j[pair_j, daxis] - cpos
                    compat &= np.abs(offset_i + offset_j) \
                            < self.offset_tolerance

                    # Check that the two directions where the two fragment
                    # meet is consistent between the two
                    cosang = np.dot(end_dirs_i[pair_i], -end_dirs_j[pair_j])
                    compat &= np.arccos(cosang) < self.angle_tolerance

                    # If compatible, merge
                    if compat:
                        # Merge particle and adjust positions
                        self.adjust_positions(result_dict, ci, cj, truth=truth)

                        # Update the candidate list to remove matched particle
                        candidate_ids[j:-1] = candidate_ids[j+1:] - 1
                        candidate_ids = candidate_ids[:-1]
                    else:
                        j += 1

                # Increment
                i += 1

            # Update crossing interactions information
            int_k = f'{prefix}interactions'
            for ia in result_dict[int_k]:
                crosser, offsets = False, []
                parts = [p for p in result_dict[k] \
                        if p.interaction_id == ia.id]
                for p in parts:
                    if p.interaction_id != ia.id:
                        continue
                    crosser |= p.is_ccrosser
                    if p.is_ccrosser:
                        offsets.append(p.coffset)

                if crosser:
                    ia.is_ccrosser = crosser
                    ia.coffset = np.mean(offsets)

            # Update
            update_dict.update({k: result_dict[k]})
            update_dict.update({int_k: result_dict[int_k]})

        return {}, update_dict


    def adjust_positions(self, result_dict, idx_i, idx_j=None, truth=False):
        '''
        Given a cathode crosser (either in one or two pieces), apply the
        necessary position offsets to match it at the cathode.

        Parameters
        ----------
        result_dict : dict
            Chain output dictionary
        idx_i : int
            Index of a cathode crosser (or a cathode crosser fragment)
        idx_j : int, optional
            Index of a matched cathode crosser fragment
        truth : bool, default False
            If True, adjust truth object positions

        Results
        -------
        np.ndarray
           (N, 3) Point coordinates
        '''
        # If there are two indexes, create a new merged particle object
        prefix = 'truth_' if truth else ''
        k, int_k = f'{prefix}particles', f'{prefix}interactions'
        input_k = 'input_rescaled' \
                if 'input_rescaled' in result_dict else 'input_data'
        particles = result_dict[k]
        if idx_j is not None:
            # Merge particles
            int_id_i = particles[idx_i].interaction_id
            int_id_j = particles[idx_j].interaction_id
            particles[idx_i].merge(particles.pop(idx_j))

            # Update the particle IDs and interaction IDs
            assert idx_j > idx_i
            for i, p in enumerate(particles):
                p.id = i
                if p.interaction_id == int_id_j:
                    p.interaction_id = int_id_i
        
        # Get TPCs that contributed to this particle
        particle = particles[idx_i]
        modules, tpcs = self.geo.get_contributors(particle.sources)
        assert len(tpcs) == 2 and modules[0] == modules[1], \
                'Can only handle particles crossing a single cathode'

        # Get the particle's sisters
        int_id = particle.interaction_id
        sisters = [p for p in particles if p.interaction_id == int_id]

        # Get the cathode position
        m = modules[0]
        daxis, cpos = self.geo.cathodes[m]
        dcol = COORD_COLS[daxis]

        # Loop over contributing TPCs, shift the points in each independently
        offsets, global_offset = \
                self.get_cathode_offsets(particle, m, tpcs)
        for i, t in enumerate(tpcs):
            # Move each of the sister particles by the same amount
            for sister in sisters:
                part_index = self.geo.get_tpc_index(sister.sources, m, t)
                index = sister.index[part_index]
                if not len(index):
                    continue

                sister.points[part_index, daxis] -= offsets[i]
                result_dict[input_k][index, dcol] -= offsets[i]
                if truth:
                    sister.truth_points[part_index] -= offset
                    result_dict['cluster_label_adapted'][index, dcol] \
                            -= offsets[i]

        # Store crosser information
        particle.is_ccrosser = True
        particle.coffset = global_offset

        # Update interactions
        if idx_j is None:
            # In this case, just need to update the positions
            interactions = result_dict[int_k]
            points = [sister.points for sister in sisters]
            interactions[int_id].points = np.vstack(points)
            if truth:
                truth_points = [sister.truth_points for sister in sisters]
                interactions[int_id].truth_point = np.vstack(truth_points)
        else:
            interactions = []
            interaction_ids = np.array([p.interaction_id for p in particles])
            for i, int_id in enumerate(np.unique(interaction_ids)):
                # Get particles in interaction int_id
                particle_ids = np.where(interaction_ids == int_id)[0]
                parts = [particles[i] for i in particle_ids]

                # Build interactions
                if not truth:
                    interaction = Interaction.from_particles(parts)
                    interaction.id = i
                else:
                    interaction = TruthInteraction.from_particles(parts)
                    interaction.id = i
                    interaction.truth_id = int_id

                # Reset the interaction ID of the constiuent particles
                for j in particle_ids:
                    particles[j].interaction_id = i

                # Append
                interactions.append(interaction)

            result_dict[int_k] = interactions


    def get_cathode_offsets(self, particle, module, tpcs):
        '''
        Find the distance one must shift a particle points by to make
        both TPC contributions align at the cathode.

        Parameters
        ----------
        particle : Union[Particle, TruthParticle]
            Particle object
        module : int
            Module ID
        tpcs : List[int]
            List of TPC IDs

        Returns
        -------
        np.ndarray
            Offsets to apply to the each TPC contributions
        float
            General offset for this particle (proxy of out-of-time displacement)
        '''
        # Get the cathode position
        daxis, cpos = self.geo.cathodes[module]
        dvector = (np.arange(3) == daxis).astype(float)

        # Check which side of the cathode each TPC lives
        flip = (-1) ** (self.geo.boundaries[module, tpcs[0], daxis].mean() \
                > self.geo.boundaries[module, tpcs[1], daxis].mean())

        # Loop over the contributing TPCs
        closest_points = np.empty((2, 3))
        offsets = np.empty(2)
        for i, t in enumerate(tpcs):
            # Get the end points of the track segment
            index  = self.geo.get_tpc_index(particle.sources, module, t)
            points = self.get_points(particle)[index]
            idx0, idx1, _ = farthest_pair(points, 'recursive')
            end_points = points[[idx0, idx1]]

            # Find the point closest to the cathode
            tpc_offset = self.geo.get_min_tpc_offset(end_points,
                    module, t)[daxis]
            cdists = end_points[:, daxis] - tpc_offset - cpos
            argmin = np.argmin(np.abs(cdists))
            closest_points[i] = end_points[argmin]

            # Compute the offset to bring it to the cathode
            offsets[i] = cdists[argmin] + tpc_offset

        # Now optimize the offsets based on angular matching
        # xing_point = np.mean(closest_points, axis=0)
        # xing_point[daxis] = cpos
        # for i, t in enumerate(tpcs):
        #     end_dir = -cluster_direction(points, closest_points[i])
        #     factor = (cpos - closest_points[i, daxis])/end_dir[daxis]
        #     intersection = closest_points[i] + factor * end_dir
        #     vplane = dvector - end_dir/end_dir[daxis] if end_dir[daxis] else -end_dir
        #     dplane = intersection - xing_point
        #     disp = np.dot(dplane, vplane)/np.dot(vplane, vplane)
        #     offsets[i] = [disp, offsets[i]][np.argmin(np.abs([disp, offsets[i]]))]

        # Take the average offset as the value to use
        global_offset = flip * (offsets[1] - offsets[0])/2.

        return offsets, global_offset
