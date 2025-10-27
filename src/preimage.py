"""
Pre-image problem solver using closed-form solution.
Based on Honeine & Richard (2011).
"""

import numpy as np
from scipy.linalg import lstsq


class PreImageSolver:
    """
    Solves the kernel pre-image problem.
    Given φ'  in Hilbert space, find x' such that φ(x') ≈ φ'.
    """
    
    def __init__(self, mu_p=1.0, method="closed_form", clip_range=None):
        self.mu_p = mu_p
        self.method = method
        self.clip_range = clip_range
        
        if method not in ["closed_form", "iterative"]:
            raise ValueError(f"Unknown method: {method}")
    
    def solve(self, X, K, theta):
        """
        Solve pre-image problem for a single point.
        X is (m, n), K is (n, n), theta is (n,).
        """
        if self.method == "closed_form":
            return self._solve_closed_form(X, K, theta)
        else:
            raise NotImplementedError("Iterative method not implemented")
    
    def _solve_closed_form(self, X, K, theta):
        """
        Closed-form solution from Honeine & Richard (2011).
        Solves: argmin ||X^T x' - (X^T X - μ_P K^{-1}) θ||^2
        """
        m, n = X.shape
        
        XTX = X.T @ X
        
        K_reg = K + 1e-10 * np.eye(n)
        K_inv = np.linalg.inv(K_reg)
        
        b = (XTX - self.mu_p * K_inv) @ theta
        
        result = lstsq(X.T, b)
        x_prime = result[0]
        
        if self.clip_range is not None:
            x_prime = np.clip(x_prime, self.clip_range[0], self.clip_range[1])
        
        return x_prime
    
    def solve_batch(
        self,
        X,
        K,
        Theta,
    ) -> np.ndarray:
        """
        Solve pre-image problem for multiple points.
        
        Args:
            X: (m, n) training data matrix
            K: (n, n) kernel Gram matrix
            Theta: (n, k) coefficient matrix, each column is a θ
            
        Returns:
            X_prime: (m, k) pre-images
        """
        m, n = X.shape
        k = Theta.shape[1]
        
        X_prime = np.zeros((m, k))
        
        for i in range(k):
            X_prime[:, i] = self.solve(X, K, Theta[:, i])
        
        return X_prime
    
    def solve_augmentation(
        self,
        X,
        K,
        M,
        x_indices = None,
    ) -> np.ndarray:
        """
        Solve pre-image for augmented points.
        
        Given transformation T_H in Hilbert space represented by matrix M,
        compute augmented points in input space.
        
        For each x_i, compute T(x_i) by:
        1. Apply transformation in Hilbert space: φ' = M φ(x_i)
        2. Solve pre-image: T(x_i) = φ^{-1}(φ')
        
        Args:
            X: (m, n) training data matrix
            K: (n, n) kernel Gram matrix
            M: (n, n) transformation matrix (e.g., for Barlow Twins)
            x_indices: Indices of points to augment (default: all)
            
        Returns:
            X_aug: (m, k) augmented points
        """
        m, n = X.shape
        
        if x_indices is None:
            x_indices = np.arange(n)
        
        k = len(x_indices)
        X_aug = np.zeros((m, k))
        
        # For each point to augment
        for i, idx in enumerate(x_indices):
            # Get coefficients for this point's transformation
            # T_H(φ(x_i)) = Φ M e_i = Φ M[:, i]
            theta = M[:, idx]
            
            # Solve pre-image
            X_aug[:, i] = self.solve(X, K, theta)
        
        return X_aug
    
    def __repr__(self):
        return f"PreImageSolver(mu_p={self.mu_p}, method={self.method})"


def validate_preimage_quality(
    X_original,
    X_preimage,
    K_original,
    K_preimage,
    kernel_func,
) -> dict:
    """
    Validate quality of pre-image solution.
    
    The pre-image x' should have φ(x') ≈ φ' in the Hilbert space.
    We can check this by comparing kernel evaluations.
    
    Args:
        X_original: (m, n) original training data
        X_preimage: (m, k) computed pre-images
        K_original: (n, n) kernel matrix on original data
        K_preimage: (k, k) kernel matrix on pre-images
        kernel_func: Kernel function to evaluate
        
    Returns:
        Dictionary with quality metrics
    """
    # Compute cross-kernel between original and pre-images
    K_cross = kernel_func(X_original.T, X_preimage.T)
    
    # The pre-image quality can be assessed by kernel alignment
    # or by checking if the kernel values are preserved
    
    # Average kernel value preservation
    diag_original = np.diagonal(K_original).mean()
    diag_preimage = np.diagonal(K_preimage).mean()
    diag_diff = abs(diag_original - diag_preimage)
    
    # Frobenius norm of difference (for same-size matrices)
    if K_original.shape == K_preimage.shape:
        frobenius_diff = np.linalg.norm(K_original - K_preimage, 'fro')
    else:
        frobenius_diff = None
    
    return {
        'diagonal_original': float(diag_original),
        'diagonal_preimage': float(diag_preimage),
        'diagonal_difference': float(diag_diff),
        'frobenius_difference': frobenius_diff,
    }
