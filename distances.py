import numpy as np

def euclidean_distance(x1, x2, axis = None):
    return np.sqrt(np.sum((x1 - x2) ** 2  , axis= axis))