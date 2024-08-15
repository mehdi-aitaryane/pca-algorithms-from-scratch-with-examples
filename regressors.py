
import numpy as np
from kernels import LinearKernel

class RidgeRegression:
  """
  Ridge regression implementation from scratch using numpy.
  """
  def __init__(self, alpha = 1.0):
    self.alpha = alpha

  def fit(self, X, y):

    self.X = X
    self.y = y

    # Add a column of ones for the bias term
    self.X = np.c_[np.ones(X.shape[0]), X]

    # Calculate the ridge regression coefficients
    I = np.eye(self.X.shape[1])
    self.w = np.linalg.inv(self.X.T @ self.X + self.alpha * I) @ self.X.T @ y

  def predict(self, X):

    X = np.c_[np.ones(X.shape[0]), X]
    return X @ self.w

class KernelRidgeRegression(RidgeRegression):
  """
  Kernel Ridge regression implementation from scratch using numpy.
  """
  def __init__(self, alpha = 1.0, kernel = LinearKernel()):
    super().__init__(alpha=alpha)
    self.kernel = kernel

  def fit(self, X, y):
    
    self.X = X
    K = self.kernel.transform(X, self.X)

    super().fit(X, y)
    
  def predict(self, X):

    return  super().predict(X)
