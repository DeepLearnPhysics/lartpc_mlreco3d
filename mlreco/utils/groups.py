# utility function to reconcile groups data with energy deposition and 5-types data:
# problem: parse_cluster3d and parse_sparse3d will not output voxels in same order
# additionally, some voxels in groups data do not deposit energy, so do not appear in images
# also, some voxels have more than one group.
# plan is to put in a function to:
# 1) lexicographically sort group data (images are lexicographically sorted)
# 2) remove voxels from group data that are not in image
# 3) choose only one group per voxel (by lexicographic order)
# WARNING: (3) is certainly not a canonical choice

import numpy as np
import torch
from larcv import larcv

def get_group_types(particle_v, meta, point_type="3d"):
    """
    Gets particle classes for voxel groups
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    gt_types = []
    for particle in particle_v:
        pdg_code = abs(particle.pdg_code())
        prc = particle.creation_process()

        # Determine point type
        if (pdg_code == 2212):
            gt_type = 0 # proton
        elif pdg_code != 22 and pdg_code != 11:
            gt_type = 1
        elif pdg_code == 22:
            gt_type = 2
        else:
            if prc == "primary" or prc == "nCapture" or prc == "conv":
                gt_type = 2 # em shower
            elif prc == "muIoni" or prc == "hIoni":
                gt_type = 3 # delta
            elif prc == "muMinusCaptureAtRest" or prc == "muPlusCaptureAtRest" or prc == "Decay":
                gt_type = 4 # michel
            else:
                gt_type = -1 # not well defined

        gt_types.append(gt_type)

    return np.array(gt_types)


def filter_duplicate_voxels(data, usebatch=True):
    """
    return array that will filter out duplicate voxels
    Only first instance of voxel will appear
    Assume data[:4] = [x,y,z,batchid]
    Assumes data is lexicographically sorted in x,y,z,batch order
    """
    # set number of cols to look at
    if usebatch:
        k = 4
    else:
        k = 3
    n = data.shape[0]
    ret = np.empty(n, dtype=bool)
    for i in range(1,n):
        if np.all(data[i-1,:k] == data[i,:k]):
            # duplicate voxel
            ret[i-1] = False
        else:
            # new voxel
            ret[i-1] = True
    ret[n-1] = True
    # ret[0] = True
    # for i in range(n-1):
    #     if np.all(data[i,:k] == data[i+1,:k]):
    #         # duplicate voxel
    #         ret[i+1] = False
    #     else:
    #         # new voxel
    #         ret[i+1] = True
    return ret


def filter_duplicate_voxels_ref(data, reference, meta, usebatch=True, precedence=[1,2,0,3,4]):
    """
    return array that will filter out duplicate voxels
    Sort with respect to a reference and following the specified precedence order
    Assume data[:4] = [x,y,z,batchid]
    Assumes data is lexicographically sorted in x,y,z,batch order
    """
    # set number of cols to look at
    if usebatch:
        k = 4
    else:
        k = 3
    n = data.shape[0]
    ret = np.full(n, True, dtype=bool)
    duplicates = {}
    for i in range(1,n):
        if np.all(data[i-1,:k] == data[i,:k]):
            x, y, z = int(data[i,0]), int(data[i,1]), int(data[i,2])
            id = meta.index(x, y, z)
            if id in duplicates:
                duplicates[id].append(i)
            else:
                duplicates[id] = [i-1, i]
    for d in duplicates.values():
        ref = np.array([precedence.index(r) for r in reference[d]])
        args = np.argsort(-ref, kind='mergesort') # Must preserve of order of duplicates
        ret[np.array(d)[args[:-1]]] = False

    return ret


def filter_nonimg_voxels(data_grp, data_img, usebatch=True):
    """
    return array that will filter out voxels in data_grp that are not in data_img
    ASSUME: data_grp and data_img are lexicographically sorted in x,y,z,batch order
    ASSUME: all points in data_img are also in data_grp
    ASSUME: all voxels in data are unique
    """
    # set number of cols to look at
    if usebatch:
        k = 4
    else:
        k = 3
    ngrp = data_grp.shape[0]
    nimg = data_img.shape[0]
    igrp = 0
    iimg = 0
    ret = np.empty(ngrp, dtype=bool) # return array
    while igrp < ngrp and iimg < nimg:
        if np.all(data_grp[igrp,:k] == data_img[iimg,:k]):
            # voxel is in both data
            ret[igrp] = True
            igrp += 1
            iimg += 1
        else:
            # voxel is in data_grp, but not data_img
            ret[igrp] = False
            igrp += 1
    # need to go through rest of data_grp if any left
    while igrp < ngrp:
        ret[igrp] = False
        igrp += 1
    return ret


def filter_group_data(data_grp, data_img):
    """
    return return array that will permute and filter out voxels so that data_grp and data_img have same voxel locations
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)
    WARNING: (3) is certainly not a canonical choice
    """
    # step 1: lexicographically sort group data
    perm = np.lexsort(data_grp[:,:-1:].T)
    data_grp = data_grp[perm,:]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels(data_grp)
    inds1 = np.where(sel1)[0]
    data_grp = data_grp[inds1,:]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(data_grp, data_img)
    inds2 = np.where(sel2)[0]

    return perm[inds1[inds2]]


def process_group_data(data_grp, data_img):
    """
    return processed group data
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)
    WARNING: (3) is certainly not a canonical choice
    """
    data_grp_np = data_grp.cpu().detach().numpy()
    data_img_np = data_img.cpu().detach().numpy()

    inds = filter_group_data(data_grp_np, data_img_np)

    return data_grp[inds,:]


def get_interaction_id(particle_v, num_ancestor_loop=1):
    '''
    A function to sort out interaction ids.
    Note that this assumes cluster_id==particle_id.
    Inputs:
        - particle_v (array)     : larcv::EventParticle.as_vector()
        - num_ancestor_loop (int): number of ancestor loops (default 1)
    Outputs:
        - interaction_ids: a numpy array with the shape (n,)
    '''
    ##########################################################################
    # sort out the interaction ids using the information of ancestor vtx info
    # then loop over to make sure the ancestor particles having the same interaction ids
    ##########################################################################
    # get the particle ancestor vtx array first
    # and the track ids
    # and the ancestor track ids
    ancestor_vtxs = []
    track_ids = []
    ancestor_track_ids = np.empty(0, dtype=int)
    for particle in particle_v:
        ancestor_vtx = [
            particle.ancestor_x(),
            particle.ancestor_y(),
            particle.ancestor_z(),
        ]
        ancestor_vtxs.append(ancestor_vtx)
        track_ids.append(particle.track_id())
        ancestor_track_ids = np.append(ancestor_track_ids, [particle.ancestor_track_id()])
    ancestor_vtxs = np.asarray(ancestor_vtxs)
    # get the list of unique interaction vertexes
    interaction_vtx_list = np.unique(
        ancestor_vtxs,
        axis=0,
    ).tolist()
    # loop over each cluster to assign interaction ids
    interaction_ids = np.ones(particle_v.size(), dtype=int)*(-1)
    for clust_id in range(particle_v.size()):
        # get the interaction id from the unique list (index is the id)
        interaction_ids[clust_id] = interaction_vtx_list.index(
            ancestor_vtxs[clust_id].tolist()
        )
    # Loop over ancestor, making sure particle having the same interaction id as ancestor
    for _ in range(num_ancestor_loop):
        for clust_id, ancestor_track_id in enumerate(ancestor_track_ids):
            if ancestor_track_id in track_ids:
                ancestor_clust_index = track_ids.index(ancestor_track_id)
                interaction_ids[clust_id] = interaction_ids[ancestor_clust_index]

    return interaction_ids


def get_nu_id(cluster_event, particle_v, interaction_ids, particle_mpv=None):
    '''
    A function to sorts interactions into nu or not nu (0 for cosmic, 1 for nu).
    CAVEAT: Dirty way to sort out nu_ids
            Assuming only one nu interaction is generated and first group/cluster belongs to such interaction
    Inputs:
        - cluster_event (larcv::EventClusterVoxel3D): (N) Array of cluster tensors
        - particle_v vector: larcv::EventParticle.as_vector()
        - interaction_id: a numpy array with shape (n, 1) where 1 is interaction id
        - (optional) particle_mpv: vector of particles from mpv generator, used to work around
        the lack of proper interaction id for the time being.
    Outputs:
        - nu_id: a numpy array with the shape (n,1)
    '''
    # initiate the nu_id
    nu_id = np.zeros(len(particle_v))

    if particle_mpv is None:
        # find the first cluster that has nonzero size
        sizes = np.array([cluster_event.as_vector()[i].as_vector().size() for i in range(len(particle_v))])
        nonzero = np.where(sizes > 0)[0]
        if not len(nonzero):
            return nu_id
        first_clust_id = nonzero[0]
        # the corresponding interaction id
        nu_interaction_id = interaction_ids[first_clust_id]
        # Get clust indexes for interaction_id = nu_interaction_id
        inds = np.where(interaction_ids == nu_interaction_id)[0]
        # Check whether there're at least two clusts coming from 'primary' process
        num_primary = 0
        for i, part in enumerate(particle_v):
            if i not in inds:
                continue
            create_prc = part.creation_process()
            parent_pdg = part.parent_pdg_code()
            if create_prc == 'primary' or parent_pdg == 111:
                num_primary += 1
        # if there is nu interaction
        if num_primary > 1:
            nu_id[inds] = 1
    elif len(particle_mpv) > 0:
        # Find mpv particles
        is_mpv = np.zeros((len(particle_v),))
        # mpv_ids = [p.id() for p in particle_mpv]
        mpv_pdg = np.array([p.pdg_code() for p in particle_mpv]).reshape((-1,))
        mpv_energy = np.array([p.energy_init() for p in particle_mpv]).reshape((-1,))
        for idx, part in enumerate(particle_v):
            # track_id - 1 in `particle_pcluster_tree` corresponds to id (or track_id) in `particle_mpv_tree`
            # if (part.track_id()-1) in mpv_ids or (part.ancestor_track_id()-1) in mpv_ids:
            # FIXME the above was wrong I think.
            close = np.isclose(part.energy_init()*1e-3, mpv_energy)
            pdg = part.pdg_code() == mpv_pdg
            if (close & pdg).any():
                is_mpv[idx] = 1.
            # else:
            #     print("fake cosmic", part.pdg_code(), part.shape(), part.creation_process(), part.track_id(), part.ancestor_track_id(), mpv_ids)
        is_mpv = is_mpv.astype(bool)
        nu_interaction_ids = np.unique(interaction_ids[is_mpv])
        for idx, x in enumerate(nu_interaction_ids):
            # # Check whether there're at least two clusts coming from 'primary' process
            # num_primary = 0
            # for part in particle_v[interaction_ids == x]:
            #     if part.creation_process() == 'primary':
            #         num_primary += 1
            # if num_primary > 1:
            nu_id[interaction_ids == x] = 1 # Only tells whether neutrino or not
            # nu_id[interaction_ids == x] = idx

    return nu_id


type_labels = {
    22: 0,  # photon
    11: 1,  # e-
    -11: 1, # e+
    13: 2,  # mu-
    -13: 2, # mu+
    211: 3, # pi+
    -211: 3, # pi-
    2212: 4, # protons
}


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
        if t in type_labels.keys():
            particle_ids[i] = type_labels[t]
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
        if p.shape() != larcv.kShapeShower and p.shape() != larcv.kShapeTrack:
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
