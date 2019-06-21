# Utility to determine if a cluster is primary ionization of EM shower
# based on NeutrinoGNN/neutrino_gnn/util/cluster/data.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
from mlreco.utils.ppn import contains
from mlreco.utils.gnn.cluster import get_cluster_label, get_cluster_batch
from mlreco.utils.gnn.compton import filter_compton


def get_em_primary_info(particle_v, meta, point_type="3d", min_voxel_count=5, min_energy_deposit=0.05):
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_positions = []
    for pid, particle in enumerate(particle_v):
        pdg_code = particle.pdg_code()
        prc = particle.creation_process()
        # Skip particle under some conditions
        if (particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count):
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue
        if pdg_code == 11 or pdg_code == 22 or pdg_code == -11:  # Shower
            # we are now in EM primary
            if not contains(meta, particle.first_step(), point_type=point_type):
                continue
            
            # TODO deal with different 2d projections
            # Register start point
            x = particle.first_step().x()
            y = particle.first_step().y()
            z = particle.first_step().z()
            px = particle.px()
            py = particle.py()
            pz = particle.pz()
            if point_type == '3d':
                x = (x - meta.min_x()) / meta.size_voxel_x()
                y = (y - meta.min_y()) / meta.size_voxel_y()
                z = (z - meta.min_z()) / meta.size_voxel_z()
                gt_positions.append([x, y, z, px, py, pz, pid])
            else:
                x = (x - meta.min_x()) / meta.pixel_width()
                y = (y - meta.min_y()) / meta.pixel_height()
                gt_positions.append([x, y, px, py, pz, pid])

    return np.array(gt_positions)


###
# Assignment of primaries to clusters
# note that these functions should become obsolete when we include this explicitly in data
###


def score_cluster_primary(clust, data, clabel, primary):
    """
    score how far off cluster is from primary trajectory
    * whether label matches
    * distance from primary start
    """
    # l = primary[-1]
    # # check if primary label and cluster label agree
    # if l != clabel:
    #    return np.inf
    # cluster voxel positions
    cx = data[clust, :3]
    # primary position
    x = primary[:3]
    
    d = np.min(sp.spatial.distance.cdist([x], cx, 'euclidean'), axis=1)[0]
    return d


def score_clusters_primary(clusts, data, labels, primary):
    """
    return cluster scores for an EM primary
    """
    n = len(clusts)
    scores = np.zeros(n)
    for i, c in enumerate(clusts):
        scores[i] = score_cluster_primary(c, data, labels[i], primary)
    return scores


def assign_primaries(primaries, clusts, data):
    """
    for each EM primary assign closest cluster that matches batch and group
    data should contain groups of voxels
    """
    
    primaries = primaries.cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    
    #first remove compton-like clusters from list
    selection = filter_compton(clusts) # non-compton looking clusters
    selinds = np.where(selection)[0] # selected indices
    cs2 = clusts[selinds]
    # if everything looks compton, say no primaries
    if len(cs2) < 1:
        return []
    
    labels = get_cluster_label(data, cs2)
    batches = get_cluster_batch(data, cs2)
    
    assn = []
    for primary in primaries:
        # get list of indices that match label and batch
        pbatch = primary[-2]
        # plabel = primary[-1]
        # pselection = np.logical_and(labels == plabel, batches == pbatch)
        pselection = batches == pbatch
        pinds = np.where(pselection)[0] # indices to compare against
        if len(pinds) < 1:
            continue
        
        scores = score_clusters_primary(cs2[pinds], data, labels[pinds], primary)
        ind = np.argmin(scores)
        # print(scores[ind])
        assn.append(selinds[pinds[ind]])
    return assn


def assign_primaries2(primaries, clusts, data):
    """
    for each EM primary assign closest cluster that matches batch and group
    data should contain groups of voxels
    
    this version does not filter out compton clusters first
    """
    
    primaries = primaries.cpu()
    data = data.cpu()
    
    labels = get_cluster_label(data, clusts)
    batches = get_cluster_batch(data, clusts)
    
    assn = []
    for primary in primaries:
        # get list of indices that match label and batch
        pbatch = primary[-2]
        # plabel = primary[-1]
        # pselection = np.logical_and(labels == plabel, batches == pbatch)
        pselection = batches == pbatch
        pinds = np.where(pselection)[0] # indices to compare against
        if len(pinds) < 1:
            continue
        
        scores = score_clusters_primary(clusts[pinds], data, labels[pinds], primary)
        ind = np.argmin(scores)
        # print(scores[ind])
        assn.append(pinds[ind])
    return assn

def assign_primaries3(primaries, clusts, data):
    """
    for each EM primary assign closest cluster that matches batch and group
    data should contain groups of voxels
    """
    
    #first remove compton-like clusters from list
    cs2 = clusts
#     selection = filter_compton(clusts) # non-compton looking clusters
#     selinds = np.where(selection)[0] # selected indices
#     cs2 = clusts[selinds]
    # if everything looks compton, say no primaries
    if len(cs2) < 1:
        return []
    
    labels = get_cluster_label(data, cs2)
    batches = get_cluster_batch(data, cs2)
    
    assn = []
    for primary in primaries:
        # get list of indices that match label and batch
        pbatch = primary[-2]
        plabel = primary[-1]
        pselection = np.logical_and(labels == plabel, batches == pbatch)
        pinds = np.where(pselection)[0] # indices to compare against
        if len(pinds) < 1:
            assn.append(-1)
            continue
        
        scores = score_clusters_primary(cs2[pinds], data, labels[pinds], primary)
        ind = np.argmin(scores)
        # print(scores[ind])
#         assn.append(selinds[pinds[ind]])
        assn.append(pinds[ind])
    return assn