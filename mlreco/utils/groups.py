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


def get_interaction_id(particle_v, np_features, num_ancestor_loop=1):
    '''
    A function to sort out interaction ids.
    Note that this assumes cluster_id==particle_id.
    Inputs:
        - particle_v vector: larcv::EventParticle.as_vector()
        - np_features: a numpy array with the shape (n,4) where 4 is voxel value,
        cluster id, group id, and semantic type respectively
        - number of ancestor loops (default 1)
    Outputs:
        - interaction_ids: a numpy array with the shape (n,)
    '''
    # initiate the interaction_ids, setting all ids to -1 (as unknown) by default
    interaction_ids = (-1.)*np.ones(np_features.shape[0])
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
    interaction_ids_cluster_wise = np.ones(particle_v.size(), dtype=np.int)*(-1)
    for clust_id in range(particle_v.size()):
        # get the interaction id from the unique list (index is the id)
        interaction_ids_cluster_wise[clust_id] = interaction_vtx_list.index(
            ancestor_vtxs[clust_id].tolist()
        )
    # Loop over ancestor, making sure particle having the same interaction id as ancestor
    for _ in range(num_ancestor_loop):
        for clust_id, ancestor_track_id in enumerate(ancestor_track_ids):
            if ancestor_track_id in track_ids:
                ancestor_clust_index = track_ids.index(ancestor_track_id)
                interaction_ids_cluster_wise[clust_id] = interaction_ids_cluster_wise[ancestor_clust_index]
    # loop over clusters to assign interaction to voxel wise array
    for clust_id, interaction_id in enumerate(interaction_ids_cluster_wise):
        # update the interaction_ids array
        clust_inds = np.where(np_features[:,1]==clust_id)[0]
        interaction_ids[clust_inds] = interaction_id
    return interaction_ids