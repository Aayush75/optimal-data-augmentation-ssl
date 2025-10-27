"""
Barlow Twins loss - symmetrized unnormalized cross-correlation.
Based on Zbontar et al. (2021) and Simon et al. (2023).
"""

import torch
import torch.nn as nn
import numpy as np


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss: L_BT = sum(1-C_ii)^2 + lambda*sum(C_ij^2)
    C is the symmetrized cross-correlation matrix.
    """
    
    def __init__(self, lambda_param=0.005, normalize=False):
        super().__init__()
        self.lambda_param = lambda_param
        self.normalize = normalize
        
    def forward(self, Z, Z_prime):
        d, n = Z.shape
        
        # Normalize if requested (standard Barlow Twins does this)
        if self.normalize:
            Z = (Z - Z.mean(dim=1, keepdim=True)) / (Z.std(dim=1, keepdim=True) + 1e-8)
            Z_prime = (Z_prime - Z_prime.mean(dim=1, keepdim=True)) / (Z_prime.std(dim=1, keepdim=True) + 1e-8)
        
        # Compute symmetrized unnormalized cross-correlation
        # C = 1/(2n) * (Z @ Z'^T + Z' @ Z^T)
        C = (Z @ Z_prime.T + Z_prime @ Z.T) / (2 * n)
        
        # On-diagonal terms: Σ_i (1 - C_ii)^2
        on_diag = torch.diagonal(C)
        on_diag_loss = torch.sum((1 - on_diag) ** 2)
        
        # Off-diagonal terms: Σ_{i≠j} C_ij^2
        # We can compute this as: sum(C^2) - sum(diag(C)^2)
        off_diag_loss = torch.sum(C ** 2) - torch.sum(on_diag ** 2)
        
        # Total loss
        loss = on_diag_loss + self.lambda_param * off_diag_loss
        
        # Information for logging
        info = {
            'loss': loss.item(),
            'on_diag_loss': on_diag_loss.item(),
            'off_diag_loss': off_diag_loss.item(),
            'on_diag_mean': on_diag.mean().item(),
            'off_diag_mean': (C.sum() - on_diag.sum()).item() / (d * (d - 1)),
        }
        
        return loss, info
    
    def compute_cross_correlation(self, Z, Z_prime):
        """
        Compute the symmetrized cross-correlation matrix C.
        """
        n = Z.shape[1]
        C = (Z @ Z_prime.T + Z_prime @ Z.T) / (2 * n)
        return C


def barlow_twins_loss_numpy(Z, Z_prime, lambda_param=0.005):
    """
    NumPy version of Barlow Twins loss.
    Useful for verification.
    """
    d, n = Z.shape
    
    # Compute cross-correlation
    C = (Z @ Z_prime.T + Z_prime @ Z.T) / (2 * n)
    
    # On-diagonal terms
    on_diag = np.diagonal(C)
    on_diag_loss = np.sum((1 - on_diag) ** 2)
    
    # Off-diagonal terms
    off_diag_loss = np.sum(C ** 2) - np.sum(on_diag ** 2)
    
    # Total loss
    loss = on_diag_loss + lambda_param * off_diag_loss
    
    info = {
        'loss': float(loss),
        'on_diag_loss': float(on_diag_loss),
        'off_diag_loss': float(off_diag_loss),
        'on_diag_mean': float(on_diag.mean()),
        'off_diag_mean': float((C.sum() - on_diag.sum()) / (d * (d - 1))),
    }
    
    return loss, info


def verify_barlow_twins_optimality(Z, Z_prime, tolerance=1e-6):
    """
    Check if representations achieve zero loss (C approximately equals I).
    """
    d, n = Z.shape
    
    # Compute cross-correlation
    C = (Z @ Z_prime.T + Z_prime @ Z.T) / (2 * n)
    
    # Check if C ≈ I
    I = np.eye(d)
    error = np.linalg.norm(C - I, 'fro')
    
    # Check diagonal and off-diagonal separately
    diag_error = np.abs(np.diagonal(C) - 1).max()
    off_diag_error = np.abs(C - np.diag(np.diagonal(C))).max()
    
    is_optimal = error < tolerance
    
    return {
        'is_optimal': is_optimal,
        'frobenius_error': float(error),
        'max_diagonal_error': float(diag_error),
        'max_off_diagonal_error': float(off_diag_error),
        'tolerance': tolerance,
    }
