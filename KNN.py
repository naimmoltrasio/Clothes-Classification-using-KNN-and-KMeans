import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.neighbors = None
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        train = 1 * np.reshape(train_data, (train_data.shape[0], -1))
        self.train_data = train.astype(float)
        return self.train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        test = 1 * np.reshape(test_data, (test_data.shape[0], -1))
        test_data = test.astype(float)
        distances = cdist(test_data, self.train_data, 'euclidean')
        neighbors = []
        x = []
        y = []

        for i in range(len(distances)):
            neighbors.append(distances[i].argsort()[:k])

        for array in range(len(neighbors)):
            for j in range(len(neighbors[array])):
                index = neighbors[array][j]
                x.append(self.labels[index])
            y.append(x)
            x = []
        self.neighbors = y
        return self.neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        uniqueArr = []
        matrix = self.neighbors

        for row in matrix:
            uniqueEl = []
            uniqueEl, counts = np.unique(row, return_counts=True)
            if len(set(counts)) == 1:
                mostV = row[0]
            else:
                indexMostV = np.argmax(counts)
                a = counts[indexMostV]
                if np.count_nonzero(counts == a) > 1:
                    indices = []
                    maxValues = []
                    for i in range(len(counts)):
                        if counts[i] == a:
                            indices.append(i)
                    for x in range(len(indices)):
                        maxValues.append(uniqueEl[indices[x]])
                    for valor in row:
                        if valor in maxValues:
                            mostV = valor
                            break
                else:
                    mostV = uniqueEl[indexMostV]
            uniqueArr.append(mostV)

        return uniqueArr

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        test_data = 1 * np.reshape(test_data, (test_data.shape[0], -1))
        self.get_k_neighbours(test_data, k)
        return self.get_class()
