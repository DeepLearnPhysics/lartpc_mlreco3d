from mlreco.utils.gnn.features.utils import *
import numpy as np
from mlreco.utils.gnn.cluster import form_clusters
from mlreco.utils.gnn.primary import assign_primaries_unique
from sklearn.neighbors import KNeighborsClassifier

def find_shower_cone(dbscan, em_primaries, energy_data, types, length_factor=14.107334041, slope_percentile=52.94032412, slope_factor=5.86322059):
    """
    dbscan: data parsed from "dbscan_label": ["parse_dbscan", "sparse3d_fivetypes"]
    em_primaries: data parsed from "em_primaries" : ["parse_em_primaries", "sparse3d_data", "particle_mcst"]
    energy_data: data parsed from "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    
    returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary
    """
    clusts = form_clusters(dbscan)
    selected_voxels = []
    true_voxels = []
    
    if len(clusts) == 0:
        # assignn everything to first primary
        selected_voxels.append(np.arange(len(dbscan)))
        print('all clusters identified as Compton')
        return selected_voxels
    assigned_primaries = assign_primaries_unique(em_primaries, clusts, types).astype(int)
    for i in range(len(assigned_primaries)):
        if assigned_primaries[i] != -1:
            c = clusts[assigned_primaries[i]]
            
            p = em_primaries[i]
            em_point = p[:3]

            # find primary cluster axis
            primary_points = dbscan[c][:, :3]
            primary_energies = energy_data[c][:, -1]
            if np.sum(primary_energies) == 0:
                selected_voxels.append(np.array([]))
                continue
            primary_center = np.average(primary_points.T, axis=1, weights=primary_energies)
            primary_axis = primary_center - em_point

            # find furthest particle from cone axis
            primary_length = np.linalg.norm(primary_axis)
            direction = primary_axis / primary_length
            axis_distances = np.linalg.norm(np.cross(primary_points-primary_center, primary_points-em_point), axis=1)/primary_length
            axis_projections = np.dot(primary_points - em_point, direction)
            primary_slope = np.percentile(axis_distances/axis_projections, slope_percentile)
            
            # define a cone around the primary axis
            cone_length = length_factor * primary_length
            cone_slope = slope_factor * primary_slope
            cone_vertex = em_point
            cone_axis = direction

            classified_indices = []
            for j in range(len(dbscan)):
                point = types[j]
                if point[-1] < 2:
                    continue
                coord = point[:3]
                axis_dist = np.dot(coord - em_point, cone_axis)
                if 0 <= axis_dist and axis_dist <= cone_length:
                    cone_radius = axis_dist * cone_slope
                    point_radius = np.linalg.norm(np.cross(coord-(em_point + cone_axis), coord-em_point))
                    if point_radius < cone_radius:
                        # point inside cone
                        classified_indices.append(j)
            classified_indices = np.array(classified_indices)
            selected_voxels.append(classified_indices)
        else:
            selected_voxels.append(np.array([]))
    
    return selected_voxels

# node features: [# voxels in cluster]
# edge features: [edge labels]
    # label = 1 if in cone, 0.5 if knn-assigned to cone
def cone_features(data, em_filter, edges):
    dbscan = data['dbscan_label'][em_filter]
    energy_data = data['input_data'][em_filter]
    em_primaries = data['em_primaries']
    types = data['segment_label'][em_filter]
    cone_assignments = find_shower_cone(dbscan, em_primaries, energy_data, types)
    node_labels = -1*np.ones(len(dbscan))
    for l in range(len(cone_assignments)):
        if len(cone_assignments[l]) > 0:
            node_labels[cone_assignments[l]] = l
    
    # TODO test two approaches for unlabeled nodes: randomization and knn
    unlabeled = np.where(node_labels == -1)
    labeled = np.where(node_labels != -1)
    
    # randomize labels so no edges are drawn between unlabeled nodes
#     node_labels[unlabeled] = -1 * np.random.uniform(size=len(unlabeled[0]))
    
    # classify unlabeled points with nearest neighbors to cone classifications
    if len(labeled[0]) == 0:
        node_labels = np.ones(len(dbscan))
        edge_labels = node_labels_to_edge_labels(edges, node_labels).astype(np.float64)
    elif len(unlabeled[0]) > 0:
        positions = types[:, :3]
        classified_positions = positions[labeled]
        unclassified_positions = positions[unlabeled]
        cl = KNeighborsClassifier(n_neighbors=2)
        cl.fit(classified_positions, node_labels[labeled])
        node_labels[unlabeled] = cl.predict(unclassified_positions)
        edge_labels = node_labels_to_edge_labels(edges, node_labels).astype(np.float64)
        edge_labels[np.where(np.isin(edges, unlabeled[0]))[0]] *= 0.5
    else:
        edge_labels = node_labels_to_edge_labels(edges, node_labels).astype(np.float64)
    edge_features = np.reshape(edge_labels, (-1, 1))
    node_features = np.reshape(node_labels_to_cluster_sizes(node_labels), (-1, 1))
    return node_features, edge_features
