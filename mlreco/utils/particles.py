import numpy as np

from .globals import (TRACK_SHP, MICHL_SHP, DELTA_SHP, INVAL_ID,
                      INVAL_TID, PDG_TO_PID)


def get_valid_mask(particles):
    '''
    A function which checks that the particle labels have been filled properly
    at the SUPERA level. It the `interaction_id` attribute is filled, it uses
    its validity to check. Otherwise, it checks that the track ID of each
    particle is not an invalid number and that the ancestor creation process
    is filled.

    Parameters
    ----------
    particles : list(larcv.Particle)
        (P) List of true particle instances

    Results
    -------
    np.ndarray
        (P) Boolean list of validity, one per true particle instance
    '''
    # If there are no particles, nothing to do here
    if not len(particles):
        return np.empty(0, dtype=bool)

    # If the interaction IDs are set in the particle tree, simply use that
    inter_ids = np.array([p.interaction_id() for p in particles], dtype=np.int32)
    if np.any(inter_ids != INVAL_ID):
        return inter_ids != INVAL_ID

    # Otherwise, check that the ancestor track ID and creation process are valid
    mask  = np.array([p.ancestor_track_id() != INVAL_TID for p in particles])
    mask &= np.array([bool(len(p.ancestor_creation_process())) for p in particles])

    return mask


def get_interaction_ids(particles):
    '''
    A function which gets the interaction ID of each of the particle in
    the input particle list. If the `interaction_id` member of the
    larcv.Particle class is filled, it simply uses that quantity.

    Otherwise, it leverages shared ancestor position as a
    basis for interaction building and sets the interaction
    ID to -1 for particles with invalid ancestor track IDs.

    Parameters
    ----------
    particles : list(larcv.Particle)
        (P) List of true particle instances

    Results
    -------
    np.ndarray
        (P) List of interaction IDs, one per true particle instance
    '''
    # If there are no particles, nothing to do here
    if not len(particles):
        return np.empty(0, dtype=np.int32)

    # Get the mask of valid particle labels
    valid_mask = get_valid_mask(particles)

    # If the interaction IDs are set in the particle tree, simply use that
    inter_ids = np.array([p.interaction_id() for p in particles], dtype=np.int32)
    if np.any(inter_ids != INVAL_ID):
        inter_ids[~valid_mask] = -1
        return inter_ids

    # Otherwise, define interaction IDs on the basis of sharing an ancestor vertex position
    anc_pos   = np.vstack([[getattr(p, f'ancestor_{a}')() for a in ['x', 'y', 'z']] for p in particles])
    inter_ids = np.unique(anc_pos, axis=0, return_inverse=True)[-1]

    # Now set the interaction ID of particles with an undefined ancestor to -1
    inter_ids[~valid_mask] = -1

    return inter_ids


def get_nu_ids(particles, inter_ids, particles_mpv=None, neutrinos=None):
    '''
    A function which gets the neutrino-like ID (-1 for cosmics, index for
    neutrino) of each of the particle in the input particle list.

    If `particles_mpv` and `neutrinos` are not specified, it assumes that
    only neutrino-like interactions have more than one true primary
    particle in a single interaction.

    If a list of multi-particle vertex (MPV) particles or neutrinos is
    provided,  that information is leveraged to identify which interactions
    are neutrino-like and which are not.

    Parameters
    ----------
    particles : list(larcv.Particle)
        (P) List of true particle instances
    inter_ids : np.ndarray
        (P) Array of interaction ID values, one per true particle instance
    particles_mpv : list(larcv.Particle), optional
        (M) List of true MPV particle instances
    neutrinos : list(larcv.Neutrino), optional
        (N) List of true neutrino instances

    Results
    -------
    np.ndarray
        List of neutrino IDs, one per true particle instance
    '''
    # Make sure there is only either MPV particles or neutrinos specified, not both
    assert particles_mpv is None or neutrinos is None, \
            'Do not specify both particle_mpv_event and neutrino_event in parse_cluster3d'

    # Initialize neutrino IDs
    nu_ids = -np.ones(len(inter_ids), dtype=inter_ids.dtype)
    if particles_mpv is None and neutrinos is None:
        # Loop over the interactions
        # TODO: Warn that this is dangerous
        primary_ids = get_group_primary_ids(particles)
        nu_id = 0
        for i in np.unique(inter_ids):
            # If the interaction ID is invalid, skip
            if i < 0: continue

            # If there are at least two primaries, the interaction is neutrino-like
            inter_index = np.where(inter_ids == i)[0]
            if np.sum(primary_ids[inter_index] == 1) > 1:
                nu_ids[inter_index] = nu_id
                nu_id += 1
    else:
        # Find the reference positions to gauge if a particle comes from a neutrino-like interaction
        ref_pos = None
        if particles_mpv:
            ref_pos = np.vstack([[getattr(p, a)() for a in ['x', 'y', 'z']] for p in particles_mpv])
            ref_pos = np.unique(ref_pos, axis=0)
        elif neutrinos:
            ref_pos = np.vstack([[getattr(n, a)() for a in ['x', 'y', 'z']] for n in neutrinos])

        # If any particle in an interaciton shares its ancestor position with an MPV particle
        # or a neutrino, the whole interaction is a neutrino-like interaction.
        if ref_pos is not None and len(ref_pos):
            anc_pos = np.vstack([[getattr(p, f'ancestor_{a}')() for a in ['x', 'y', 'z']] for p in particles])
            for i in np.unique(inter_ids):
                if i < 0: continue
                inter_index = np.where(inter_ids == i)[0]
                for ref_id, pos in enumerate(ref_pos):
                    if np.any((anc_pos[inter_index] == pos).all(axis=1)):
                        nu_ids[inter_index] = ref_id
                        break

    return nu_ids


def get_particle_ids(particles, nu_ids, include_mpr=False, include_secondary=False):
    '''
    Function which gets a particle ID (PID) for each of the particle in
    the input particle list. This function ensures:
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

    Parameters
    ----------
    particles : list(larcv.Particle)
        (P) List of true particle instances
    nu_ids : np.ndarray
        (P) Array of neutrino ID values, one per true particle instance
    include_mpr : bool, default False
        Include cosmic-like particles (MPR or cosmics) to valid PID labels
    include_secondary : bool, default False
        Inlcude secondary particles to valid PID labels

    Returns
    -------
    np.ndarray
        (P) List of particle IDs, one per true particle instance
    '''
    particle_ids = -np.ones(len(nu_ids), dtype=np.int32)
    primary_ids  = get_group_primary_ids(particles, nu_ids, include_mpr)
    for i in range(len(particle_ids)):
        # If the primary ID is invalid, skip
        if primary_ids[i] < 0: continue

        # If secondary particles are not included and primary_id < 1, skip
        if not include_secondary and primary_ids[i] < 1: continue

        # If the particle type exists in the predefined list, assign
        group_id = particles[i].group_id()
        t = particles[group_id].pdg_code()
        if t in PDG_TO_PID.keys():
            particle_ids[i] = PDG_TO_PID[t]

    return particle_ids


def get_shower_primary_ids(particles):
    '''
    Function which gets primary labels for shower fragments.
    This could be handled somewhere else (e.g. SUPERA)

    Parameters
    ----------
    particles : list(larcv.Particle)
        (P) List of true particle instances

    Results
    -------
    np.ndarray
        (P) List of particle shower primary IDs, one per true particle instance
    '''
    # Loop over the list of particle groups
    primary_ids = np.zeros(len(particles), dtype=np.int32)
    group_ids   = np.array([p.group_id() for p in particles], dtype=np.int32)
    valid_mask  = get_valid_mask(particles)
    for g in np.unique(group_ids):
        # If the particle group has invalid labeling or if it is a track
        # group, the concept of shower primary is ill-defined
        if (g == INVAL_ID or
            not valid_mask[g] or
            particles[g].shape() == TRACK_SHP):
            group_index = np.where(group_ids == g)[0]
            primary_ids[group_index] = -1
            continue

        # If a group originates from a Delta or a Michel, it has a primary
        p = particles[g]
        if p.shape() == MICHL_SHP or p.shape() == DELTA_SHP:
            primary_ids[g] = 1
            continue

        # If a shower group's parent fragment the first in time, it is a valid primary
        group_index = np.where(group_ids == g)[0]
        clust_times = np.array([particles[i].first_step().t() for i in group_index])
        min_id = np.argmin(clust_times)
        if group_index[min_id] == g:
            primary_ids[g] = 1

    return primary_ids


def get_group_primary_ids(particles, nu_ids=None, include_mpr=True):
    '''
    Parameters
    ----------
    particles : list(larcv.Particle)
        (P) List of true particle instances
    nu_ids : np.ndarray, optional
        (P) List of neutrino IDs, one per particle instance
    include_mpr : bool, default False
        Include cosmic-like particles (MPR or cosmics) to valid primary labels

    Results
    -------
    np.ndarray
        (P) List of particle primary IDs, one per true particle instance
    '''
    # Loop over the list of particles
    primary_ids = -np.ones(len(particles), dtype=np.int32)
    valid_mask  = get_valid_mask(particles)
    for i, p in enumerate(particles):
        # If the particle has invalid labeling, it has invalid primary status
        if p.group_id() == INVAL_ID or not valid_mask[i]:
            continue

        # If MPR particles are not included and the nu_id < 0, assign invalid
        if not include_mpr and nu_ids is not None and nu_ids[i] < 0:
            continue

        # If the particle originates from a primary pi0, label as primary
        # Small issue with photo-nuclear activity here, but very rare
        group_p = particles[p.group_id()]
        if group_p.ancestor_pdg_code() == 111:
            primary_ids[i] = 1
            continue

        # If the origin of a particle agrees with the origin of its ancestor, label as primary
        group_pos = np.array([getattr(group_p, a)() for a in ['x', 'y', 'z']])
        anc_pos   = np.array([getattr(p, f'ancestor_{a}')() for a in ['x', 'y', 'z']])
        primary_ids[i] = (group_pos == anc_pos).all()

    return primary_ids
