import numpy as np
import torch

from mlreco.utils.globals import SHOW_SHP, TRACK_SHP, LOWE_SHP, INVAL_TID, PDG_TO_PID


def get_interaction_ids(particles):
    '''
    A function which gets the interaction ID of each of the
    particle in the input particle list. It leverages shared
    ancestor position as a basis for interaction building and
    sets the interaction ID to -1 for parrticles with invalid
    ancestor track IDs.

    Parameters
    ----------
    particles : list(larcv.Particle)
        List of true particle instances

    Results
    -------
    np.ndarray
        List of interaction IDs, one per true particle instance
    '''
    # Define the interaction ID on the basis of sharing an ancestor vertex position
    anc_pos   = np.vstack([[getattr(p, f'ancestor_{a}')() for a in ['x', 'y', 'z']] for p in particles])
    inter_ids = np.unique(anc_pos, axis=0, return_inverse=True)[-1]

    # Now set the interaction ID of particles with an undefined ancestor to -1
    if len(particles):
        anc_ids = np.array([p.ancestor_track_id() for p in particles])
        inter_ids[anc_ids == INVAL_TID] = -1

    return inter_ids


def get_nu_ids(inter_ids, particles, particles_mpv=None, neutrinos=None):
    '''
    A function which gets the neutrino-like ID (0 for cosmic, 1 for
    neutrino) of each of the particle in the input particle list.

    If `particles_mpv` and `neutrinos` are not specified, it assumes that
    there is only one neutrino-like interaction, the first valid one, and
    it enforces that it must contain at least two true primaries.

    If a list of multi-particle vertex (MPV) particles or neutrinos is
    provided,  that information is leveraged to identify which interaction
    is  neutrino-like and which is not.

    Parameters
    ----------
    inter_ids : np.ndarray
        Array of interaction ID values, one per true particle instance
    particles : list(larcv.Particle)
        List of true particle instances
    particles_mpv : list(larcv.Particle), optional
        List of true MPV particle instances
    neutrinos : list(larcv.Neutrino), optional
        List of true neutrino instances

    Results
    -------
    np.ndarray
        List of neutrino IDs, one per true particle instance
    '''
    # Make sure there is only either MPV particles or neutrinos specified, not both
    assert particles_mpv is None or neutrinos is None,\
            'Do not specify both particle_mpv_event and neutrino_event in parse_cluster3d'

    # Initialize neutrino IDs
    nu_ids = np.zeros(len(inter_ids), dtype=inter_ids.dtype)
    nu_ids[inter_ids == -1] = -1
    if particles_mpv is None and neutrinos is None:
        # Find the first particle with a valid interaction ID
        valid_mask = np.where(inter_ids > -1)[0]
        if not len(valid_mask):
            return nu_ids

        # Identify the interaction ID of that particle
        inter_id = inter_ids[valid_mask[0]]
        inter_index = np.where(inter_ids == inter_id)[0]

        # If there are at least two primaries, the interaction is nu-like
        primary_ids = get_group_primary_id(particles)
        num_primary = np.sum(primary_ids[inter_index])
        if num_primary > 1:
            nu_ids[inter_index] = 1
    else:
        # Find the reference positions gauge if a particle comes from a neutrino-like interaction
        ref_pos = None
        if particles_mpv:
            ref_pos = np.vstack([[getattr(p, f'{a}')() for a in ['x', 'y', 'z']] for p in particles_mpv])
        elif neutrinos:
            ref_pos = np.vstack([[getattr(n, f'{a}')() for a in ['x', 'y', 'z']] for n in neutrinos])

        # If a particle shares its ancestor position with an MPV particle
        # or a neutrino, it belongs to a neutrino-like interaction
        if ref_pos is not None and len(ref_pos):
            anc_pos = np.vstack([[getattr(p, f'ancestor_{a}')() for a in ['x', 'y', 'z']] for p in particles])
            for pos in ref_pos:
                nu_ids[(anc_pos == pos).all(axis=1)] = 1

    return nu_ids


def get_particle_id(particles_v, nu_ids, include_mpr=False, include_secondary=False):
    '''
    Function that gives one of five labels to particles of
    particle species predictions. This function ensures:
    - Particles that do not originate from an MPV are labeled -1,
      unless the include_mpr flag is set to true
    - Secondary particles (includes Michel/delta and neutron activity) are
      labeled -1, unless the include_secondary flag is true
    - All shower daughters are labeled the same as their primary. This
      makes sense as otherwise an electron primary gets overruled by
      its many photon daughters (voxel-wise majority vote). This can
      lead to problems as, if an electron daughter is not clustered with
      the primary, it is labeled electron, which is counter-intuitive.
      This is handled downstream with the high_purity flag.
    - Particles that are not in the list target are labeled -1

    Inputs:
        - particles_v (array of larcv::Particle)    : (N) LArCV Particle objects
        - nu_ids: a numpy array with shape (n, 1) where 1 is neutrino id (0 if not an MPV)
        - include_mpr: include MPR (cosmic-like) particles to PID target
        - include_secondary: include secondary particles into the PID target
    Outputs:
        - array: (N) list of group ids
    '''
    particle_ids = np.empty(len(nu_ids))
    primary_ids  = get_group_primary_id(particles_v, nu_ids, include_mpr)
    for i in range(len(particle_ids)):
        # If the primary ID is invalid, assign invalid
        if primary_ids[i] < 0:
            particle_ids[i] = -1
            continue

        # If secondary particles are not included and primary_id < 1, assign invalid
        if not include_secondary and primary_ids[i] < 1:
            particle_ids[i] = -1
            continue

        # If the particle type exists in the predefined list, assign
        group_id = int(particles_v[i].group_id())
        t = int(particles_v[group_id].pdg_code())
        if t in PDG_TO_PID.keys():
            particle_ids[i] = PDG_TO_PID[t]
        else:
            particle_ids[i] = -1

    return particle_ids


def get_shower_primary_id(cluster_event, particles_v):
    '''
    Function that assigns valid primary tags to shower fragments.
    This could be handled somewhere else (e.g. SUPERA)

    Inputs:
        - cluster_event (larcv::EventClusterVoxel3D): (N) Array of cluster tensors
        - particles_v (array of larcv::Particle)    : (N) LArCV Particle objects
    Outputs:
        - array: (N) list of group ids
    '''
    # Loop over the list of particles
    group_ids   = np.array([p.group_id() for p in particles_v])
    primary_ids = np.empty(particles_v.size(), dtype=np.int32)
    for i, p in enumerate(particles_v):
        # If the particle is a track or a low energy cluster, it is not a primary shower fragment
        if p.shape() == 1 or p.shape() == 4:
            primary_ids[i] = 0
            continue

        # If a particle is a Delta or a Michel, it is a primary shower fragment
        if p.shape() == 2 or p.shape() == 3:
            primary_ids[i] = 1
            continue

        # If the shower fragment originates from nuclear activity, it is not a primary
        process = p.creation_process()
        parent_pdg_code = abs(p.parent_pdg_code())
        if 'Inelastic' in process or 'Capture' in process or parent_pdg_code == 2112:
            primary_ids[i] = 0
            continue

        # If a shower group's parent fragment has size zero, there is no valid primary in the group
        gid = int(p.group_id())
        parent_size = cluster_event.as_vector()[gid].as_vector().size()
        if not parent_size:
            primary_ids[i] = 0
            continue

        # If a shower group's parent fragment is not the first in time, there is no valid primary in the group
        idxs = np.where(group_ids == gid)[0]
        clust_times = np.array([particles_v[int(j)].first_step().t() for j in idxs])
        min_id = np.argmin(clust_times)
        if idxs[min_id] != gid :
            primary_ids[i] = 0
            continue

        # If all conditions are met, label shower fragments which have identical ID and group ID as primary
        primary_ids[i] = int(gid == i)

    return primary_ids


def get_group_primary_id(particles_v, nu_ids=None, include_mpr=True):
    '''
    Function that assigns valid primary tags to particle groups.
    This could be handled somewhere else (e.g. SUPERA)

    Inputs:
        - particles_v (array of larcv::Particle)    : (N) LArCV Particle objects
        - nu_ids: a numpy array with shape (n, 1) where 1 is neutrino id (0 if not an MPV)
        - include_mpr: include MPR (cosmic-like) particles to primary target
    Outputs:
        - array: (N) list of group ids
    '''
    # Loop over the list of particles
    primary_ids = np.empty(particles_v.size(), dtype=np.int32)
    for i, p in enumerate(particles_v):
        # If MPR particles are not included and the nu_id < 1, assign invalid
        if not include_mpr and nu_ids[i] < 1:
            primary_ids[i] = -1
            continue

        # If the ancestor particle is unknown (no creation process), assign invalid (TODO: fix in supera)
        if not p.ancestor_creation_process():
            primary_ids[i] = -1
            continue

        # If the particle is not a shower or a track, it is not a primary
        if p.shape() != SHOW_SHP and p.shape() != TRACK_SHP and p.shape() != LOWE_SHP:
            primary_ids[i] = 0
            continue

        # If the particle group originates from nuclear activity, it is not a primary
        gid = int(p.group_id())
        process = particles_v[gid].creation_process()
        parent_pdg_code = abs(particles_v[gid].parent_pdg_code())
        ancestor_pdg_code = abs(particles_v[gid].ancestor_pdg_code())
        if 'Inelastic' in process or 'Capture' in process or parent_pdg_code == 2112 or ancestor_pdg_code == 2112:
            primary_ids[i] = 0
            continue

        # If the parent is a pi0, make sure that it is a primary pi0 (pi0s are not stored in particle list)
        if parent_pdg_code == 111 and ancestor_pdg_code != 111:
            primary_ids[i] = 0
            continue

        # If the parent ID of the primary particle in the group is the same as the group ID, it is a primary
        primary_ids[i] = int(particles_v[gid].parent_id() == gid)

    return primary_ids
