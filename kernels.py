import numpy as np
from distances import euclidean_distance


def linear_kernel(X, Y):
    K = np.dot(X, Y.T)
    return K

def rbf_kernel(X, Y, gamma):
    """Computes the Radial Basis Function (RBF) kernel between two data matrices."""
    dists = np.square(euclidean_distance(X[:, np.newaxis], Y, axis=2))
    K = np.exp(-gamma * dists)
    return K

class LinearKernel:

    def transform(self, X, Y):
        return linear_kernel(X, Y)


class RBFKernel:

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def transform(self, X, Y):
        """Transforms data using the RBF kernel."""
        return rbf_kernel(X, Y, self.gamma)
