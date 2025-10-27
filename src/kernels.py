"""
Kernel functions for the optimal augmentation framework.
Linear, RBF, and Polynomial kernels.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    """
    Base class for kernels.
    """
    
    @abstractmethod
    def compute(self, X, Y=None):
        """
        Compute kernel matrix K(X, Y).
        X is (n, d), Y is (m, d). If Y is None, computes K(X, X).
        """
        pass
    
    def __call__(self, X, Y=None):
        return self.compute(X, Y)


class LinearKernel(BaseKernel):
    """
    Linear kernel: K(x, y) = x^T y
    Just the dot product.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute(self, X, Y=None):
        """
        Compute linear kernel matrix.
        
        Args:
            X: (n, d) array
            Y: (m, d) array or None
            
        Returns:
            K: (n, m) or (n, n) kernel matrix
        """
        if Y is None:
            Y = X
        return X @ Y.T
    
    def __repr__(self):
        return "LinearKernel()"


class RBFKernel(BaseKernel):
    """
    RBF (Gaussian) Kernel: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    Universal kernel, always full rank for distinct samples.
    """
    
    def __init__(self, sigma=1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma ** 2)
        
    def compute(self, X, Y=None):
        """
        Compute RBF kernel matrix.
        
        Args:
            X: (n, d) array
            Y: (m, d) array or None
            
        Returns:
            K: (n, m) or (n, n) kernel matrix
        """
        if Y is None:
            Y = X
            
        # Compute pairwise squared Euclidean distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (n, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)  # (1, m)
        distances_sq = X_norm + Y_norm - 2 * (X @ Y.T)
        
        # Avoid negative values due to numerical errors
        distances_sq = np.maximum(distances_sq, 0)
        
        # Compute kernel
        K = np.exp(-self.gamma * distances_sq)
        return K
    
    def __repr__(self):
        return f"RBFKernel(sigma={self.sigma})"


class PolynomialKernel(BaseKernel):
    """
    Polynomial Kernel: K(x, y) = (x^T y + coef0)^degree
    """
    
    def __init__(self, degree=3, coef0=1.0):
        super().__init__()
        if degree <= 0:
            raise ValueError(f"degree must be positive, got {degree}")
        self.degree = degree
        self.coef0 = coef0
        
    def compute(self, X, Y=None):
        """
        Compute polynomial kernel matrix.
        
        Args:
            X: (n, d) array
            Y: (m, d) array or None
            
        Returns:
            K: (n, m) or (n, n) kernel matrix
        """
        if Y is None:
            Y = X
        
        # Compute dot product
        dot_product = X @ Y.T
        
        # Add coef0 and raise to power
        K = (dot_product + self.coef0) ** self.degree
        return K
    
    def __repr__(self):
        return f"PolynomialKernel(degree={self.degree}, coef0={self.coef0})"


def get_kernel(kernel_type, **kwargs):
    """
    Get a kernel by name.
    kernel_type: "linear", "rbf", or "polynomial"
    """
    kernel_type = kernel_type.lower()
    
    if kernel_type == "linear":
        return LinearKernel()
    elif kernel_type == "rbf":
        sigma = kwargs.get("sigma", 1.0)
        return RBFKernel(sigma=sigma)
    elif kernel_type == "polynomial":
        degree = kwargs.get("degree", 3)
        coef0 = kwargs.get("coef0", 1.0)
        return PolynomialKernel(degree=degree, coef0=coef0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                        f"Choose from: linear, rbf, polynomial")


def check_kernel_conditions(K, tol=1e-10):
    """
    Check if kernel matrix satisfies Condition 3.3 (full rank).
    Returns diagnostics about the kernel matrix.
    """
    n = K.shape[0]
    
    # Check symmetry
    is_symmetric = np.allclose(K, K.T, atol=tol)
    
    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(K)
    min_eigenvalue = np.min(eigenvalues)
    is_positive_definite = min_eigenvalue > tol
    
    # Check rank
    rank = np.linalg.matrix_rank(K, tol=tol)
    is_full_rank = (rank == n)
    
    return {
        "symmetric": is_symmetric,
        "positive_definite": is_positive_definite,
        "full_rank": is_full_rank,
        "rank": rank,
        "dimension": n,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": np.max(eigenvalues),
        "condition_number": np.max(eigenvalues) / max(min_eigenvalue, tol),
    }
