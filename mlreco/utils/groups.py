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

def get_nu_id(particle_v, np_features, interaction_ids):
    '''
    A function to sort out nu ids (0 for cosmic, 1 for nu).
    CAVEAT: Dirty way to sort out nu_ids
            Assuming only one nu interaction is generated and first group/cluster belongs to such interaction
    Inputs:
        - particle_v vector: larcv::EventParticle.as_vector()
        - np_features: a numpy array with the shape (n,4) where 4 is voxel value,
        cluster id, group id, and semantic type respectively
        - interaction_id: a numpy array with shape (n, 1) where 1 is interaction id
    Outputs:
        - nu_id: a numpy array with the shape (n,1)
    '''
    # initiate the nu_id
    nu_id = np.zeros((np_features.shape[0], 1))
    # find the first cluster
    first_clust_id = np.min(np.unique(np_features[:,1]))
    # the corresponding interaction id
    nu_interaction_id = np.unique(
        interaction_ids[
            np.where(np_features[:,1]==first_clust_id)[0]
        ]
    )[0]
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



def reassign_id(data, id_index=6, batch_index=3):
    '''
    A function to reassign group id when merging all the batches,
    make sure that there is no overlap of group id
    :param data: torch tensor (N,8) -> [x,y,z,batchid,value,id,group id,sem. type]
    :return: torch tensor (N) -> reassigned group ids
    Or the input and output can be both numpy array
    '''
    group_ids = None
    batch_ids = None
    if type(data)==torch.Tensor:
        group_ids = data[:,id_index].detach().cpu().numpy()
        batch_ids = data[:,batch_index].detach().cpu().numpy()
    else:
        group_ids = data[:,id_index]
        batch_ids = data[:,batch_index]
    max_shift = 0
    for batch_id in np.unique(batch_ids):
        inds = np.where(batch_ids==batch_id)[0]
        group_ids[inds] += max_shift
        max_shift = np.max(group_ids[inds],axis=-1)+1
    if type(data)==torch.Tensor:
        return torch.tensor(group_ids, device=data.device, dtype=data.dtype)
    return group_ids.astype(dtype=data.dtype)

def form_merging_batches(batch_ids, mean_merge_size):
    """
    Function for returning a list of batch_ids for merging
    """
    num_of_batch_ids = len(batch_ids)
    # generate random numbers based on mean merge size
    nums_merging_batch = np.random.poisson(
        mean_merge_size,
        size = int(num_of_batch_ids / mean_merge_size * 2)
    )
    # remove zeros
    nums_merging_batch = nums_merging_batch[
        np.where(nums_merging_batch!=0)[0]
    ]
    # cumsum it
    nums_merging_batch = np.cumsum(nums_merging_batch)
    # cut it where it exceeds total batch number
    nums_merging_batch = nums_merging_batch[
        np.where(nums_merging_batch<num_of_batch_ids)[0]
    ]
    # complete it
    nums_merging_batch = np.append(
        np.append(0,nums_merging_batch),
        num_of_batch_ids
    )
    # loop over
    merging_batches_list = []
    for lower_index, upper_index in zip(
        nums_merging_batch[:-1],
        nums_merging_batch[1:]
    ):
        merging_batches_list.append(batch_ids[lower_index:upper_index])
    return merging_batches_list


def merge_batch(data, merge_size=2, whether_fluctuate=False, data_type='cluster'):
    """
    Merge events in same batch
    For ex., if batch size = 16 and merge_size = 2
    output data has a batch size of 8 with each adjacent 2 batches in input data merged.
    Input:
        data - (N, 8) tensor or numpy array -> [x,y,z,batch_id,value, id, group_id, sem. type]
               or can be [start_x, start_y, start_z, end_x, end_y, end_z, batch_id, group_id] if it is "particle" type
        merge_size: how many batches to be merged if whether_fluctuate=False,
                    otherwise sample the number of merged batches using Poisson with mean of merge_size
        whether_fluctuate: whether not using a constant merging size
        type:       'cluster" or "particle"
    Output:
        output_data - (N, 8) tensor or numpy array
    """
    # specify batch index
    index_for_batch = 3
    if data_type=='particle':
        index_for_batch = 6
    # specify resign id index
    reassign_id_indexes = [5,6]
    if data_type=='particle':
        reassign_id_indexes = [7]
    # Get the unique batch ids
    batch_ids = None
    if type(data)==torch.Tensor:
        batch_ids = data[:,index_for_batch].unique()
    elif type(data)==np.ndarray:
        batch_ids = np.unique(data[:,index_for_batch])
    else:
        return data
    # Get the list of arrays
    if whether_fluctuate:
        merging_batch_id_list = form_merging_batches(batch_ids, merge_size)
    else:
        if type(batch_ids)==torch.Tensor:
            # to be sure
            batch_ids = batch_ids.cpu().detach().numpy()
        if len(batch_ids)%merge_size==0:
            merging_batch_id_list = np.reshape(batch_ids,(-1,merge_size))
        else:
            # it will be a bit more complicated
            # if length of batch ids is indivisible by merge size
            # first reshape the divisible part
            merging_batch_id_list = np.reshape(
                batch_ids[:-int(len(batch_ids)%merge_size)],
                (-1, merge_size)
            ).tolist()
            # then append the rest
            merging_batch_id_list.append(batch_ids[-int(len(batch_ids)%merge_size):].tolist())
    # Loop over
    output_data = data
    for i, merging_batch_ids in enumerate(merging_batch_id_list):
        selection = None
        for j, batch_id in enumerate(merging_batch_ids):
            if j==0:
                selection = (data[:,index_for_batch]==batch_id)
            else:
                selection = selection | (data[:,index_for_batch]==batch_id)
        inds = None
        if type(data)==torch.Tensor:
            inds = selection.nonzero().view(-1)
        else:
            inds = np.where(selection)[0]
        # Merge batch
        for reassign_id_index in reassign_id_indexes:
            output_data[inds,reassign_id_index] = reassign_id(
                data[inds,:],
                reassign_id_index,
                index_for_batch
            )
        output_data[inds,index_for_batch] = i
    return output_data, merging_batch_id_list

def merge_batch_based_on_list(data, merging_batch_id_list, data_type='cluster'):
    """
    Similar to merge_batch
    but this function merge batches according to a list
    """
    # specify batch index
    index_for_batch = 3
    if data_type == 'particle':
        index_for_batch = 6
    # specify resign id index
    reassign_id_indexes = [5, 6]
    if data_type == 'particle':
        reassign_id_indexes = [7]
    # check if data is tensor or numpy
    if type(data)!=torch.Tensor and type(data)!=np.ndarray:
        return data, merging_batch_id_list
    # Loop over
    output_data = data
    for i, merging_batch_ids in enumerate(merging_batch_id_list):
        selection = None
        for j, batch_id in enumerate(merging_batch_ids):
            if j == 0:
                selection = (data[:, index_for_batch] == batch_id)
            else:
                selection = selection | (data[:, index_for_batch] == batch_id)
        inds = None
        if type(data) == torch.Tensor:
            inds = selection.nonzero().view(-1)
        else:
            inds = np.where(selection)[0]
        # Merge batch
        for reassign_id_index in reassign_id_indexes:
            output_data[inds, reassign_id_index] = reassign_id(
                data[inds, :],
                reassign_id_index,
                index_for_batch
            )
        output_data[inds, index_for_batch] = i
    return output_data, merging_batch_id_list

