import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from analysis.classes.Particle import Particle
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from analysis.post_processing import post_processing
from mlreco.utils.globals import *


@post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
                                                              'particle_clusts',
                                                              'particle_start_points',
                                                              'particle_end_points'])
def order_end_points(data_dict,
                     result_dict,
                     method='local_dedx',
                     neighborhood_radius=5):

    assert method == 'local_dedx', 'Only method currently supported'

    input_data   = data_dict['input_data'] if 'input_rescaled' not in result_dict else result_dict['input_rescaled']
    particles    = result_dict['particle_clusts']
    start_points = result_dict['particle_start_points']
    end_points   = result_dict['particle_end_points']

    start_dedxs, end_dedxs = np.empty(len(particles)), np.empty(len(particles))
    for i, p in enumerate(particles):
        dist_mat = cdist(start_points[i, COORD_COLS][None,:], input_data[p][:, COORD_COLS]).flatten()
        de = np.sum(input_data[p][dist_mat < neighborhood_radius, VALUE_COL])
        start_dedxs[i] = de/neighborhood_radius

        dist_mat = cdist(end_points[i, COORD_COLS][None,:], input_data[p][:, COORD_COLS]).flatten()
        de = np.sum(input_data[p][dist_mat < neighborhood_radius, VALUE_COL])
        end_dedxs[i] = de/neighborhood_radius

    switch_mask = start_dedxs > end_dedxs
    temp_start_points = deepcopy(start_points)
    start_points[switch_mask] = end_points[switch_mask]
    end_points[switch_mask] = temp_start_points[switch_mask]

    update_dict = {
        'particle_start_points': start_points,
        'particle_end_points': end_points
    }
            
    return update_dict

def order_clusters(points, eps=1.1, min_samples=1):
    """
    Order clusters along the principal axis of the track.

    Args:
    points (numpy.ndarray): array of 3D points

    Returns:
    numpy.ndarray: array of ordered cluster indices
    """
    #DBSCAN with chebyshev clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='chebyshev').fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    centroids = np.array([points[labels == label].mean(axis=0) for label in unique_labels if label != -1])

    #PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    projected_centroids = pca.transform(centroids)
    sorted_indices = np.argsort(projected_centroids[:, 0])

    return unique_labels[sorted_indices],labels

def inter_cluster_distance(points, labels, cluster_indices):
    """
    Compute inter-cluster distance between consecutive clusters.

    Args:
    points (numpy.ndarray): array of 3D points
    labels (numpy.ndarray): array of cluster labels
    cluster_indices (numpy.ndarray): array of ordered cluster indices

    Returns:
    List[float]: list of inter-cluster distances
    """
    distances = []

    for i in range(1, len(cluster_indices)):
        curr_cluster = points[labels == cluster_indices[i]]
        prev_cluster = points[labels == cluster_indices[i - 1]]
        #min_dist = np.min([np.linalg.norm(curr - prev, ord=np.inf) for curr in curr_cluster for prev in prev_cluster])
        min_dist = np.min(cdist(curr_cluster,prev_cluster))
        distances.append(min_dist)

    return distances

def gap_length_calc_cheb(points,start_point,end_point,norm_to_track_length=True,
                         eps=1.1, min_samples=1):
    """
    Computes chebyshev intercluster gap lengths, with option to normalize to total length

    Args:
        particle (analysis.classes.Particle): single particle
        norm_to_track_length (bool, optional): Divide gap length by track length. Defaults to True.
        eps (float, optional): DBSCAN epsilon distance parameter. Defaults to 1.1.
        min_samples (int, optional): Minimum number of clusters DBSCAN. Defaults to 1.

    Returns:
        float: gap length
    """
    #Get direction information
    direction = (start_point-end_point)/np.linalg.norm(start_point - end_point)
    gamma = 1/np.max(abs(direction))

    #Get cluster distances
    cluster_indices,labels = order_clusters(points,eps=eps, min_samples=min_samples)
    distances = inter_cluster_distance(points,labels,cluster_indices)
    g = np.sum(distances-gamma)
    if norm_to_track_length:
        g /= np.linalg.norm(start_point-end_point)
    return g

@post_processing(data_capture=[], result_capture=[#'input_rescaled',
                                                              #'particle_clusts',
                                                              #'particle_start_points',
                                                              #'particle_end_points',
                                                              'particles',
                                                              'truth_particles'])
def compute_gap_lengths(data_dict,
                     result_dict,
                     norm_to_track_length=True,
                     eps=1.1,
                     min_samples=1):
    """
    Adds gap length attribute to particle

    Args:
        data_dict (dictionary): _description_
        result_dict (_type): _description_
        norm_to_track_length (bool, optional): _description_. Defaults to True.
        eps (float, optional): _description_. Defaults to 1.1.
        min_samples (int, optional): _description_. Defaults to 1.
    """
    # if 'input_rescaled' not in result_dict:
    #     input_data = data_dict['input_data']
    # else:
    #     input_data = result_dict['input_rescaled']
    # particles    = result_dict['particle_clusts']
    # start_points = result_dict['particle_start_points']
    # end_points   = result_dict['particle_end_points']
    
    invalid_particle = Particle() #Get dummy start and end positions from class
    # for i,p in enumerate(particles):
    #     points = input_data[p][:, COORD_COLS]
    #     start_point = start_points[i, COORD_COLS][None,:]
    #     end_point = end_points[i, COORD_COLS][None,:]
    #     if start_point != invalid_particle.start_point and end_point != invalid_particle.end_point: #Valid start and end point
    #         p.gap_length = gap_length_calc_cheb(points,
    #                                             start_point,
    #                                             end_point,
    #                                             norm_to_track_length=norm_to_track_length,
    #                                             eps=eps,
    #                                             min_samples=min_samples) #calc intercluster distance of points
    #     else:
    #         p.gap_length = -1 #dummy value assignment
    for key in ['particles','truth_particles']:
        for i,p in enumerate(result_dict[key]):
            points = p.points
            start_point = p.start_point
            end_point = p.end_point
            if list(start_point) != list(invalid_particle.start_point) and list(end_point) != list(invalid_particle.end_point) and list(start_point) != list(end_point) and len(points) > min_samples: #Valid start and end point, use list as they compare the entire object at once
                p.gap_length = gap_length_calc_cheb(points,
                                                    start_point,
                                                    end_point,
                                                    norm_to_track_length=norm_to_track_length,
                                                    eps=eps,
                                                    min_samples=min_samples) #calc intercluster distance of points
            else:
                p.gap_length = -1 #dummy value assignment
        
    return {} #In place function
