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

def get_group_types(particle_v, meta, point_type="3d"):
    """
    Gets particle classes for voxel groups
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
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
    ret = np.empty(n, dtype=np.bool)
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


def filter_duplicate_voxels_ref(data, reference, meta, usebatch=True, precedence=[2,1,0,3,4]):
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
    ret = np.full(n, True, dtype=np.bool)
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
    ret = np.empty(ngrp, dtype=np.bool) # return array
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


def get_valid_group_id(cluster_event, particles_v):
    '''
    Function that makes sure that the particle for which id = group_id (primary)
    is a valid group ID. This should be handled somewhere else (e.g. SUPERA)

    Inputs:
        - cluster_event (larcv::EventClusterVoxel3D): (N) Array of cluster tensors
        - particles_v (array of larcv::Particle)    : (N) LArCV Particle objects
    Outputs:
        - array: (N) list of group ids
    '''
    # Only shower fragments that come first in time and deposit energy can be primaries
    num_clusters = cluster_event.as_vector().size()
    group_ids = np.array([particles_v[i].group_id() for i in range(particles_v.size())])
    new_group = num_clusters + 1
    for i, gid in enumerate(np.unique(group_ids)):
        # If the group's parent is not EM or LE, nothing to do
        if particles_v[int(gid)].shape() != 0 and particles_v[int(gid)].shape() != 4:
            continue

        # If the group's parent is nuclear activity, Delta or Michel, make it non primary
        process = particles_v[int(gid)].creation_process()
        parent_pdg_code = abs(particles_v[int(gid)].parent_pdg_code())
        idxs = np.where(group_ids == gid)[0]
        if 'Inelastic' in process or 'Capture' in process or parent_pdg_code == 13:
            group_ids[idxs] = new_group
            new_group += 1
            continue

        # If a group's parent fragment has size zero, make it non primary
        parent_size = cluster_event.as_vector()[int(gid)].as_vector().size()
        if not parent_size:
            idxs = np.where(group_ids == gid)[0]
            group_ids[idxs] = new_group
            new_group += 1
            continue

        # If a group's parent is not the first in time, make it non primary
        idxs = np.where(group_ids == gid)[0]
        clust_times = np.array([particles_v[int(j)].first_step().t() for j in idxs])
        min_id = np.argmin(clust_times)
        if idxs[min_id] != gid :
            group_ids[idxs] = new_group
            new_group += 1
            continue

    return group_ids


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
    ancestor_track_ids = np.empty(0, dtype=np.int)
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
    interaction_ids = np.ones(particle_v.size(), dtype=np.int)*(-1)
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


def get_nu_id(cluster_event, particle_v, interaction_ids):
    '''
    A function to sort out nu ids (0 for cosmic, 1 to n_nu for nu).
    CAVEAT: Dirty way to sort out nu_ids
            Assuming only one nu interaction is generated and first group/cluster belongs to such interaction
    Inputs:
        - cluster_event (larcv::EventClusterVoxel3D): (N) Array of cluster tensors
        - particle_v vector: larcv::EventParticle.as_vector()
        - interaction_id: a numpy array with shape (n, 1) where 1 is interaction id
    Outputs:
        - nu_id: a numpy array with the shape (n,1)
    '''
    # initiate the nu_id
    nu_id = np.ones(len(particle_v))*(-1)
    # find the first cluster that has nonzero size
    sizes = np.array([cluster_event.as_vector()[i].as_vector().size() for i in range(len(particle_v))])
    nonzero = np.where(sizes > 0)[0]
    first_clust_id = nonzero[0]
    # the corresponding interaction id
    nu_interaction_id = interaction_ids[first_clust_id]
    # Get clust indexes for interaction_id = nu_interaction_id
    inds = np.where(interaction_ids==nu_interaction_id)[0]
    # Check whether there're at least two clusts coming from 'primary' process
    num_primary = 0
    for i, part in enumerate(particle_v):
        if i not in inds:
            continue
        create_prc = part.creation_process()
        if create_prc=='primary':
            num_primary += 1
    # if there is nu interaction
    if num_primary>1:
        nu_id[inds]=1
    return nu_id
