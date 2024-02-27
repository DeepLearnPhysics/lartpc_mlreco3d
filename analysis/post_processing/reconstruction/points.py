from mlreco.utils.globals import TRACK_SHP
from mlreco.utils.tracking import check_track_orientation
from mlreco.utils.ppn import check_track_orientation_ppn

#imports for gap length calculation
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np

from analysis.post_processing import PostProcessor


class ParticleExtremaProcessor(PostProcessor):
    '''
    Assigns track start point and end point.
    '''
    name = 'assign_particle_extrema'
    result_cap = ['particles']
    result_cap_opt = ['ppn_candidates']

    def __init__(self,
                 method='local',
                 **kwargs):
        '''
        Parameters
        ----------
        method : algorithm to correct track startpoint/endpoint misplacement.
            The following modes are available:
            - local: computes local energy deposition density only at
            the extrema and chooses the higher one as the endpoint.
            - gradient: computes local energy deposition density throughout
            the track, computes the overall slope (linear fit) of the energy
            density variation to estimate the direction.
            - ppn: uses ppn candidate predictions (classify_endpoints) to
            assign start and endpoints.
        kwargs : dict
            Extra arguments to pass to the `check_track_orientation` or the
            `check_track_orientation_ppn' functions
        '''
        # Store the orientation method and its arguments
        self.method = method
        self.kwargs = kwargs

    def process(self, data_dict, result_dict):
        '''
        Orient all particles in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        for p in result_dict['particles']:
            if p.semantic_type == TRACK_SHP:
                # Check if the end points need to be flipped
                if self.method in ['local', 'gradient']:
                    flip = not check_track_orientation(p.points,
                            p.depositions, p.start_point, p.end_point,
                            self.method, **self.kwargs)
                elif self.method == 'ppn':
                    assert 'ppn_candidates' in result_dict, \
                            'Must run the get_ppn_predictions post-processor '\
                            'before using PPN predictions to assign  extrema'
                    flip = not check_track_orientation_ppn(p.start_point,
                            p.end_point, result_dict['ppn_candidates'])
                else:
                    raise ValueError('Point assignment method not ' \
                            f'recognized: {self.method}')

                # If needed, flip en end points
                if flip:
                    start_point, end_point = p.end_point, p.start_point
                    p.start_point = start_point
                    p.end_point   = end_point

        return {}, {}
class GapLengthProcessor(PostProcessor):
    '''
    computes gap length between adjacent clusters for tracks using chebyshev clustering
    '''
    name = 'compute_gap_length'
    result_cap = ['particles']

    def __init__(self,**kwargs):
        '''
        Parameters
        ----------
        kwargs : dict
            Extra arguments to pass to gap length function. Includes epsilon,
                min_samples, and norm_to_track_length
        '''
        # Store epsilon, min_samples, and norm_to_track_length arguments
        self.kwargs = kwargs
    def process(self, data_dict, result_dict):
        '''
        Orient all particles in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        for p in result_dict['particles']:
            if p.semantic_type == TRACK_SHP: #only compute gap length for tracks
                # Extract points, start_point, and end_point
                points = p.points
                start_point = p.start_point
                end_point = p.end_point
                gap_length = self.gap_length_calc_cheb(points,start_point,end_point,**self.kwargs)
                p.gap_length = gap_length
            else:
                p.gap_length = 0
        return {}, {}
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