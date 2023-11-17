import numpy as np
import utils

class KMeans:

    def __init__(self, X, K=5, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        """
        Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        if len(X.shape) != 2:
            self.X = 1 * np.reshape(X, (-1, X.shape[2]))
        else:
            self.X = np.copy(1 * X)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'optimum'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.2
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'

        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'custom':
            self.centroids = np.empty((self.K, self.X.shape[1]))
            for k in range(self.K):
                self.centroids[k, :] = k * 255 / (self.K - 1)
        elif self.options['km_init'].lower() == 'first':
            self.centroids = np.zeros([self.K, self.X.shape[1]])
            arrayDeZeros = np.zeros([self.X.shape[-1]])
            pos_X = 0
            pos_C_A_Add = 0
            while pos_C_A_Add < self.K and pos_X < len(self.X):
                isInCentroids = 0
                if (pos_C_A_Add == 0) or (np.array_equal(arrayDeZeros, self.X[pos_X])):
                    pass
                else:
                    for value in self.centroids:
                        if np.array_equal(value, self.X[pos_X]):
                            isInCentroids = 1
                            break
                if isInCentroids == 0:
                    self.centroids[pos_C_A_Add] = np.copy(self.X[pos_X])
                    pos_C_A_Add += 1
                pos_X += 1
        elif self.options['km_init'].lower() == 'optimum':
            self.centroids = np.zeros([self.K, self.X.shape[1]])
            MAX_VALUE = np.max(self.X, axis=0)
            MIN_VALUE = np.min(self.X, axis=0)
            if self.K % 2 == 0:
                parity = 0
            else:
                parity = 1
                self.centroids[int((self.K - 1) / 2)] = np.mean(self.X, axis=0)
            for i in range(self.X.shape[-1]):
                self.centroids[:int((self.K - parity) / 2), i] = np.linspace(MIN_VALUE[i],
                                                                             self.centroids[int((self.K - 1) / 2)][i],
                                                                             num=int((self.K - parity) / 2),
                                                                             endpoint=False)
            for i in range(self.X.shape[-1]):
                self.centroids[int((self.K - parity) / 2 + parity):, i] = np.linspace(MAX_VALUE[i], self.centroids[
                    int((self.K - parity) / 2)][i], num=int((self.K - parity) / 2), endpoint=False)
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X
         and assigns each point to the closest centroid
        """
        X = self.X
        K = self.K
        C = self.centroids
        dist = distance(X, C)

        for i in range(K):
            self.labels = np.argmin(dist, 1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        X = self.X
        K = self.K
        L = self.labels
        C = self.centroids
        self.old_centroids = np.copy(C)

        aux = np.zeros([K, X.shape[-1]])

        for i in range(len(X)):
            aux[L[i]] += X[i]

        for j in range(K):
            if np.sum(L == j) == 0:
                C[j] = np.copy(C[j - 1])
            else:
                C[j] = aux[j] / np.sum(L == j)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        C = self.centroids
        OC = self.old_centroids

        if np.amax(np.absolute(C - OC)) > self.options['tolerance']:
            return False
        else:
            return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """

        K = self.K
        self._init_centroids()
        if K == 0:
            self.get_labels()
            self.get_centroids()

        if self.options['max_iter'] > self.num_iter:
            self.get_labels()
            self.get_centroids()
            while not self.converges() and self.options['max_iter'] > self.num_iter:
                self.num_iter += 1
                self.get_labels()
                self.get_centroids()

    def withinClassDistance(self):

        """
        returns the within class distance of the current clustering
        """
        WCD = 0
        for centroid in range(self.centroids.shape[0]):
            if (self.labels == centroid).any():
                WCD += np.linalg.norm(self.X[self.labels == centroid] - self.centroids[centroid]) ** 2
        for i, pixel in enumerate(self.X):
            aux = pixel - self.centroids[self.labels[i]]
            WCD += np.matmul(aux, aux.transpose())

        return WCD / self.X.shape[0]

    def find_bestK(self, max_K):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        tolerance = self.options['tolerance']
        last_dist = -1

        for k in range(2, max_K + 1):

            self.K = k
            self._init_centroids()
            self.num_iter = 0
            self.fit()
            res = self.withinClassDistance()
            if last_dist != -1:
                aux = res / last_dist
                if (1 - aux) < tolerance:
                    self.K = k - 1
                    break

            last_dist = res

        return 1 - aux


def distance(X, C):
    """
   Calculates the distance between each pixel and each centroid
   Args:
       X (numpy array): PxD 1st set of data points (usually data points)
       C (numpy array): KxD 2nd set of data points (usually cluster centroids points)
   Returns:
       dist: PxK numpy array position ij is the distance between the
       i-th point of the first set an the j-th point of the second set
   """
    rows = C.shape[0]
    cols = X.shape[0]
    dist = np.empty([rows, cols])
    for pos in range(rows):
        dist[pos] = np.sqrt(np.sum((X - C[pos]) ** 2, axis=1))
    return dist.T


def get_colors(centroids):
    """
        for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
        Args:
           centroids (numpy array): KxD 1st set of data points (usually centroid points)
        Returns:
           labels: list of K labels corresponding to one of the 11 basic colors
    """

    color_labels = []
    matrix = np.array(utils.get_color_prob(centroids))
    for row in matrix:
        max_value = np.argmax(row)
        row_color = utils.colors[max_value]
        color_labels.append(row_color)

    return color_labels
