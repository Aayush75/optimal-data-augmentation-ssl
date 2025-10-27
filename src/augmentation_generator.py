"""
Algorithm 1 from the paper.
Generates optimal augmentations for Barlow Twins using kernel methods.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from .kernels import BaseKernel
from .preimage import PreImageSolver


class BarlowTwinsAugmentationGenerator:
    """
    Generates optimal augmentations using kernel methods and Lyapunov solver.
    """
    
    def __init__(self, kernel, lambda_ridge=1.0, mu_p=1.0, check_conditions=True):
        self.kernel = kernel
        self.lambda_ridge = lambda_ridge
        self.mu_p = mu_p
        self.check_conditions = check_conditions
        
        self.X_train = None
        self.K = None
        self.C = None
        self.B = None
        self.T_H_matrix = None
        
        self.preimage_solver = PreImageSolver(mu_p=mu_p)
    
    def fit(self, X, F_target):
        """
        Fit the generator. Implements lines 1-9 of Algorithm 1.
        X is (m, n) - flattened images as columns.
        F_target is (d, n) - target feature vectors.
        """
        m, n = X.shape
        d = F_target.shape[0]
        
        print(f"Fitting augmentation generator...")
        print(f"  Data shape: {X.shape}")
        print(f"  Target representations shape: {F_target.shape}")
        
        self.X_train = X
        
        print("  Computing kernel matrix...")
        self.K = self.kernel(X.T, X.T)
        
        if self.check_conditions:
            rank = np.linalg.matrix_rank(self.K)
            print(f"  Kernel matrix rank: {rank}/{n}")
            if rank < n:
                print("  WARNING: Kernel matrix is not full rank!")
        
        print("  Solving kernel ridge regression...")
        K_reg = self.K + self.lambda_ridge * np.eye(n)
        K_reg_inv = np.linalg.inv(K_reg)
        self.C = F_target @ K_reg_inv
        
        print(f"  Coefficient matrix C shape: {self.C.shape}")
        
        if self.check_conditions:
            rank_C = np.linalg.matrix_rank(self.C)
            print(f"  Coefficient matrix rank: {rank_C}/{d}")
            if rank_C < d:
                print("  WARNING: C matrix is not full rank!")
        
        print("  Setting up Lyapunov equation...")
        
        CKC_T = self.C @ self.K @ self.C.T
        CKC_T_inv = np.linalg.inv(CKC_T)
        CKC_T_inv2 = CKC_T_inv @ CKC_T_inv
        
        G = self.K @ self.C.T @ CKC_T_inv2 @ self.C @ self.K
        
        eigvals, eigvecs = np.linalg.eigh(self.K)
        eigvals = np.maximum(eigvals, 1e-10)
        K_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        K_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        print("  Solving Lyapunov equation...")
        
        RHS = 2 * n * K_sqrt @ self.C.T @ CKC_T_inv2 @ self.C @ K_sqrt
        
        self.B = solve_continuous_lyapunov(self.K, RHS)
        
        if self.check_conditions:
            symmetry_error = np.linalg.norm(self.B - self.B.T, 'fro')
            print(f"  Lyapunov solution symmetry error: {symmetry_error:.2e}")
        
        print("  Constructing transformation matrix...")
        self.T_H_matrix = K_sqrt_inv @ self.B @ K_sqrt_inv
        
        print("  Fit complete!")
        return self
    
    def transform(self, X_aug=None, indices=None):
        """
        Generate augmented data. Implements lines 10-14 of Algorithm 1.
        """
        if self.X_train is None:
            raise RuntimeError("Generator not fitted. Call fit() first.")
        
        if X_aug is not None:
            raise NotImplementedError("Out-of-sample augmentation not yet implemented.")
        else:
            if indices is None:
                indices = np.arange(self.X_train.shape[1])
        
        print(f"Generating augmentations for {len(indices)} samples...")
        
        X_augmented = self.preimage_solver.solve_augmentation(
            self.X_train,
            self.K,
            self.T_H_matrix,
            x_indices=indices,
        )
        
        print("  Augmentation generation complete!")
        return X_augmented
    
    def fit_transform(self, X, F_target, indices=None):
        """
        Fit and transform in one step.
        """
        self.fit(X, F_target)
        return self.transform(indices=indices)
    
    def get_augmentation_distribution(self):
        """
        Get info about the learned augmentation distribution.
        Returns transformation matrix properties.
        """
        if self.T_H_matrix is None:
            raise RuntimeError("Generator not fitted.")
        
        eigvals = np.linalg.eigvalsh(self.T_H_matrix)
        
        return {
            'transformation_matrix': self.T_H_matrix,
            'eigenvalues': eigvals,
            'min_eigenvalue': eigvals.min(),
            'max_eigenvalue': eigvals.max(),
            'condition_number': eigvals.max() / max(eigvals.min(), 1e-10),
        }
    
    def verify_optimality(self, F_learned, F_target, tolerance=1e-3):
        """
        Verify that learned representations match targets up to rotation (Theorem 4.4).
        """
        from scipy.linalg import orthogonal_procrustes
        
        # Solve Procrustes problem: Find Q s.t. ||F_learned - Q F_target||_F is minimized
        Q, scale = orthogonal_procrustes(F_target.T, F_learned.T)
        
        # Check if Q is orthogonal
        Q_orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]), 'fro')
        
        # Compute residual
        F_aligned = Q.T @ F_learned
        residual = np.linalg.norm(F_aligned - F_target, 'fro') / np.linalg.norm(F_target, 'fro')
        
        is_optimal = (residual < tolerance) and (Q_orthogonality_error < tolerance)
        
        return {
            'is_optimal': is_optimal,
            'residual': float(residual),
            'orthogonality_error': float(Q_orthogonality_error),
            'optimal_rotation': Q,
            'tolerance': tolerance,
        }
    
    def __repr__(self):
        status = "fitted" if self.X_train is not None else "not fitted"
        return (f"BarlowTwinsAugmentationGenerator("
                f"kernel={self.kernel}, "
                f"lambda_ridge={self.lambda_ridge}, "
                f"status={status})")
