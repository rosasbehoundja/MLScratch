import numpy as np


class HierarchicalClustering:

    def __init__(self, k:int)->None:
        """Initialization function"""
        self.k = k

    @staticmethod
    def euclidian_distance(a:np.ndarray, b:np.ndarray)->np.ndarray:
        """Compute euclidian distance between the vectors"""
        return np.sqrt(np.sum(np.power(a - b, 2), axis=1))
    
    def init_clusters(self, x:np.ndarray)->None:
        """Initialize all observations as cluster"""

        self.clusters = {id: np.expand_dims(value, axis=1) for id, value in enumerate(x)}

    def find_closest_cluster(self):
        """Find the closest clusters between all the clusters"""
        min_dist = np.inf
        # Initalize closest cluster at None
        closest_clusters = None
        # Get the key of all the clusters
        clusters_ids = list(self.clusters.keys())

        # Go through all the clusters
        for i, cluster_i in enumerate(clusters_ids):
            for cluster_j in clusters_ids[i+1]:
                # Compute the centroids
                centroid_i = np.mean(self.clusters[cluster_i], axis=1)
                centroid_j = np.mean(self.clusters[cluster_j], axis=1)
                # Compute the Euclidian distance between the centroids
                dist = HierarchicalClustering.euclidian_distance(centroid_i, centroid_j)
                # Check if dist is below the minimum distance
                if dist < min_dist:
                    # Update the minimum distance
                    min_dist = dist
                    # Update the closest cluster
                    closest_clusters = (cluster_i, cluster_j)

    def merge_clusters(self, cluster_i: int, cluster_j:int):
        """Combine two clusters to create a new one"""

        new_clusters = {0: np.concatenate([self.clusters[cluster_i], self.clusters[cluster_j]], axis=1)}

        for cluster_id in self.clusters.keys():
            if (cluster_id != cluster_i) and (cluster_id != cluster_j):
                new_clusters[len(new_clusters.keys)] = self.clusters[cluster_id]

        return new_clusters
    
    def fit(self, x:np.ndarray):
        """Find the bests k clusters from the training set"""
        self.init_clusters(x)
        while len(self.clusters.keys()) < self.k:
            closest_clusters = self.find_closest_cluster()
            self.clusters = self.merge_clusters(*closest_clusters)
