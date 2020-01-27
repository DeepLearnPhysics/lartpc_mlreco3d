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


def get_em_primary_info(particle_v, meta, point_type="3d", min_voxel_count=7, min_energy_deposit=10):
    """
    Gets EM particle information for training GNN
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_positions = []
    for pid, particle in enumerate(particle_v):
        pdg_code = abs(particle.pdg_code())
        prc = particle.creation_process()
        # Skip particle under some conditions
        if (particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count):
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue
        if pdg_code == 11 or pdg_code == 22:  # Shower
            # we are now in EM primary
            if not contains(meta, particle.first_step(), point_type=point_type):
                continue

            # check that the particle is not an EM daughter
            parent_pdg_code = abs(particle.parent_pdg_code())
            if parent_pdg_code == 11 or parent_pdg_code == 22:
                continue
            cp = particle.creation_process()
            if cp != 'Decay' and cp != 'primary':
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


def score_cluster_primary(clust, data, primary):
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


def score_clusters_primary(clusts, data, primary):
    """
    return cluster scores for an EM primary
    """
    n = len(clusts)
    scores = np.zeros(n)
    for i, c in enumerate(clusts):
        scores[i] = score_cluster_primary(c, data, primary)
    return scores


def assign_primaries(primaries, clusts, data, use_labels=False, max_dist=None, compton_thresh=0):
    """
    for each EM primary assign closest cluster that matches batch and group
    data should contain groups of voxels
    """
    
    primaries = primaries.cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    
    #first remove compton-like clusters from list
    selection = filter_compton(clusts, compton_thresh) # non-compton looking clusters
    selinds = np.where(selection)[0] # selected indices
    cs2 = clusts[selinds]
    # if everything looks compton, say no primaries
    if len(cs2) < 1:
        return []
    
    if use_labels:
        labels = get_cluster_label(data, cs2)
    batches = get_cluster_batch(data, cs2)
    
    assn = []
    for primary in primaries:
        # get list of indices that match label and batch
        pbatch = primary[-2]
        if use_labels:
            plabel = primary[-1]
            pselection = np.logical_and(labels == plabel, batches == pbatch)
        else:
            pselection = batches == pbatch
        pinds = np.where(pselection)[0] # indices to compare against
        if len(pinds) < 1:
            continue
        
        scores = score_clusters_primary(cs2[pinds], data, primary)
        ind = np.argmin(scores)
        if max_dist and scores[ind] > max_dist:
            continue
        # print(scores[ind])
        assn.append(selinds[pinds[ind]])
        
    # assignments may not be unique
    assn = np.unique(assn)
    return assn


def assign_primaries_unique(primaries, clusts, data, use_labels=False):
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
    
    assn = -1*np.ones(len(primaries))
    assn_scores = -1*np.ones(len(primaries))
    for i in range(len(primaries)):
        primary = primaries[i]
        # get list of indices that match label and batch
        pbatch = primary[-2]
        if use_labels:
            plabel = primary[-1]
            pselection = np.logical_and(labels == plabel, batches == pbatch)
        else:
            pselection = batches == pbatch
        pinds = np.where(pselection)[0] # indices to compare against
        if len(pinds) < 1:
            continue
        
        scores = score_clusters_primary(cs2[pinds], data, primary)
        ind = np.argmin(scores)
        pind = pinds[ind]
        score = scores[ind]
        
        already_assigned = np.where(assn == pind)[0]
        if len(already_assigned) > 0:
            current_low = assn_scores[already_assigned][0]
            if score < current_low:
                assn_scores[already_assigned] = -1.0
                assn[already_assigned] = -1.0
            else:
                continue
        assn_scores[i] = score
        assn[i] = pind
    return assn


def analyze_primaries(p_est, p_true):
    """
    Return some statistics on primary assignment
    INPUTS:
        p_est  - list of assigned primaries
        p_true - list of true primaries
    OUTPUTS:
        fdr - false discovery rate
        tdr - true discovery rate
        acc - proportion of all primaries found
    """
    n_est = len(p_est)
    n_true = len(p_true)
    n_int = len(np.intersect1d(p_est, p_true))
    n_out = n_est - n_int
    tdr = n_int * 1.0 / n_est
    fdr = n_out * 1.0 / n_est
    acc = n_int * 1.0 / n_true
    return fdr, tdr, acc
    
def get_true_primaries(clust_ids, batch_ids, points):
    # For each cluster, check that it is in the list of primary points
    primaries = []
    for i, idx in enumerate(zip(batch_ids.cpu().numpy(), clust_ids.cpu().numpy())):
        for p in points:
            if (np.array(idx) == p[-2:].cpu().numpy()).all():
                primaries.append(i)
                break

    return np.array(primaries)
