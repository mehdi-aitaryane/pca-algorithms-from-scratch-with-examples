import numpy as np
from kernels import LinearKernel
from regressors import KernelRidgeRegression

class PCA:

    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        centered_X = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_X, rowvar=False)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices]

        if self.n_components is not None:
            if self.n_components < 1:
                # Interpret n_components as the desired variance retained
                explained_variance_ratio = eigenvalues[sorted_indices] / np.sum(eigenvalues)
                cumulative_variance = np.cumsum(explained_variance_ratio)
                self.n_components = np.argmax(cumulative_variance >= self.n_components) + 1
                self.components = self.components[:, :self.n_components]
            else:
                # If n_components is a positive integer, use it directly
                self.components = self.components[:, :self.n_components]

    def transform(self, X):
        # Mean centering
        centered_X = X - self.mean

        # Project data onto the new feature space
        transformed_data = np.dot(centered_X, self.components)

        return transformed_data.real

    def fit_transform(self, X):
        # Call fit
        self.fit(X)

        # Call transform
        return self.transform(X)

    def inverse_transform(self, X):
        # Project data back to the original feature space
        original_data = np.dot(X, self.components.T) + self.mean

        return original_data.real

class KernelPCA(PCA):

    def __init__(self, n_components, kernel = LinearKernel(), alpha = 1.0):
        super().__init__(n_components = n_components)
        self.kernel = kernel
        self.X = None
        self.alpha = alpha
        self.krr = KernelRidgeRegression(alpha = alpha, kernel = kernel)

    def fit(self, X):
        self.X = X
        K = self.kernel.transform(X, self.X)
        super().fit(K)
        self.krr.fit(super().transform(K), X)  # Train KernelRidgeRegression

    def transform(self, X):
        K = self.kernel.transform(X, self.X)
        return super().transform(K)

    def inverse_transform(self, X):
        return self.krr.predict(X)  # Use KernelRidgeRegression to inverse transform
