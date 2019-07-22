from sklearn.neighbors import KNeighborsClassifier
from mlreco.utils.gnn.features.utils import *
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_centers, get_cluster_voxels
from sklearn.cluster import DBSCAN

# node features: [# voxels in cluster, cluster center, cluster "orientation", unit vector of cluster direction]*len(eps_values)
# edge features: [labels based on DBSCAN clusters for eps_values[0], ..., labels based on DBSCAN clusters for eps_values[-1]]
def dbscan_features(data, em_filter, edges):
    eps = [2, 15, 30, 60]
    delta = 0.0   # regularization
    num_node_features = 7
    orientation = False
    if orientation:
        num_node_features += 9
    
    positions = data['segment_label'][em_filter][:, :3]
    nf = []
    ef = []
    for e in eps:
        node_labels = DBSCAN(eps=e, min_samples=10).fit(positions).labels_
        node_features = np.zeros((len(positions), num_node_features))
        # create node features for truly clustered nodes only (not unlabeled)
        clusters, counts = np.unique(node_labels, return_counts=True)
        for i in range(len(clusters)):
            if clusters[i] == -1:
                continue
            x_uncentered = positions[np.where(node_labels == clusters[i])]
            center = np.mean(x_uncentered, axis=0)
            # center data
            x = x_uncentered - center
            
            # get orientation matrix
            A = x.T.dot(x)
            # get eigenvectors - convention with eigh is that eigenvalues are ascending
            w, v = np.linalg.eigh(A)
            dirwt = 0.0 if w[2] == 0 else 1.0 - w[1] / w[2] # weight for direction
            
            # get direction - look at direction of spread orthogonal to v[:,2]
            v0 = v[:,2]
            # projection of x along v0 
            x0 = x.dot(v0)
            # projection orthogonal to v0
            xp0 = x - np.outer(x0, v0)
            np0 = np.linalg.norm(xp0, axis=1)
            # spread coefficient
            sc = np.dot(x0, np0)
            if sc < 0:
                # reverse 
                v0 = -v0
            # weight direction
            v0 = dirwt*v0
            
            if orientation:
                w = w + delta # regularization
                w = w / w[2] # normalize top eigenvalue to be 1
                # orientation matrix
                B = v.dot(np.diag(w)).dot(v.T)
                cluster_feature = np.concatenate(([len(x)], center, B.flatten(), v0))
            else:
                cluster_feature = np.concatenate(([len(x)], center, v0))
            node_features[np.where(node_labels == clusters[i])] = cluster_feature
        node_features[np.where(node_labels == -1)] = np.array([0]*num_node_features)
        nf.append(node_features)
        
        # create edge features
        unlabeled = np.where(node_labels == -1)
        labeled = np.where(node_labels != -1)
        if len(labeled[0]) == 0:
            node_labels = np.ones(len(node_labels))
            edge_labels = node_labels_to_edge_labels(edges, node_labels).astype(np.float64)
        elif len(unlabeled[0]) > 0:
            classified_positions = positions[labeled]
            unclassified_positions = positions[unlabeled]
            cl = KNeighborsClassifier(n_neighbors=2)
            cl.fit(classified_positions, node_labels[labeled])
            node_labels[unlabeled] = cl.predict(unclassified_positions)
            edge_labels = node_labels_to_edge_labels(edges, node_labels).astype(np.float64)
            edge_labels[np.where(np.isin(edges, unlabeled[0]))[0]] *= 0.5
        else:
            edge_labels = node_labels_to_edge_labels(edges, node_labels).astype(np.float64)
        ef.append(np.reshape(edge_labels, (-1, 1)))
        
    nf = np.concatenate(tuple(nf), axis=1)
    ef = np.concatenate(tuple(ef), axis=1)
    return nf, ef 