import numpy as np

class KMeans:
    '''The constructor initializes the KMeans object with ratings data and movie names. It
     converts the ratings data into a numpy array if it's not already one.'''
    def __init__(self, ratings_data, movie_names):
        if isinstance(ratings_data, np.ndarray):
            self.ratings = ratings_data
        else:
            self.ratings = np.array(ratings_data)

        self.movie_names = movie_names
        self.k = None
        self.centroids = None
        self.clusters = None
        self.iteration_centroids = []  # Store centroids after each iteration


    # Randomly selects k data points as initial centroids.
    def _initialize_centroids(self):
        np.random.seed(42)
        random_indices = np.random.choice(len(self.ratings), size=self.k, replace=False)
        self.centroids = self.ratings[random_indices]

    # Computes the Euclidean distance between two points.
    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Assigns each data point to the nearest centroid based on Euclidean distance.
    def _assign_clusters(self):
        distances = np.zeros((len(self.ratings), self.k)) # Initialize distance with zero
        for i, point in enumerate(self.ratings):
            for j, centroid in enumerate(self.centroids):
                distances[i, j] = self._euclidean_distance(point, centroid)
        self.clusters = np.argmin(distances, axis=1) # selecting the minimum distance


    # Updates centroids by computing the mean of all points in each cluster.
    def _update_centroids(self):
        for i in range(self.k):
            cluster_points = self.ratings[self.clusters == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                self.centroids[i] = new_centroid


    # Performs KMeans clustering with k clusters.
    def cluster_movies(self, k):
        self.k = k
        self._initialize_centroids()
        prev_clusters = np.zeros(len(self.ratings), dtype=int) # Store the prev. cluster as stopping condition if clusters content unchanged
        while True:
            self._assign_clusters()
            self._update_centroids()
            # Append current centroids to the list after each iteration
            self.iteration_centroids.append(self.centroids.copy())
            if np.array_equal(self.clusters, prev_clusters): # Stop the loop when the clusters content doesn't change
                print(prev_clusters)
                break
            prev_clusters = np.copy(self.clusters)

        # Identify outliers within each cluster
        cluster_outliers = []
        for i in range(self.k):
            cluster_indices = np.where(self.clusters == i)[0]
            cluster_ratings = self.ratings[cluster_indices]
            print(f"Cluster {i+1}:")
            print(len(cluster_ratings))
            outliers = self._detect_outliers(cluster_ratings, i)
            cluster_outliers.append(outliers)

        # Remove outliers from clusters
        for i in range(self.k):
            self.clusters[np.where(self.clusters == i)[0][cluster_outliers[i]]] = -1  # Mark outliers in cluster as -1

    # Returns clustered movies for each cluster.
    def get_clustered_movies(self):
        clustered_movies = [[] for _ in range(self.k)]
        for i, cluster_id in enumerate(self.clusters):
            if cluster_id != -1:  # Exclude outliers
                clustered_movies[int(cluster_id)].append(i)
        return clustered_movies

    def _detect_outliers(self, data, cluster_index):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((data <= lower_bound) | (data >= upper_bound))























    ''' output_text += "Centroids After Each Iteration:-\n"
        for iteration, centroids in enumerate(centroids_history):
            output_text += f"Iteration {iteration + 1}:\n"
            for i, centroid in enumerate(centroids):
                output_text += f"Centroid {i + 1}: {centroid}\n" '''