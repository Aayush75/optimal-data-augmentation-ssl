"""
Optimal Data Augmentations for Self-Supervised Learning
Based on: arXiv:2411.01767v3

Implementation of optimal augmentation generation for Barlow Twins
using kernel methods and Lyapunov equation solvers.
"""

from .kernels import LinearKernel, RBFKernel, PolynomialKernel, get_kernel
from .target_models import TargetModel
from .augmentation_generator import BarlowTwinsAugmentationGenerator
from .losses import BarlowTwinsLoss
from .preimage import PreImageSolver
from .utils import (
    load_cifar100,
    set_seed,
    compute_procrustes_distance,
    save_image_grid,
)

__version__ = "1.0.0"
__author__ = "Based on Feigin et al., 2024"

__all__ = [
    # Kernels
    "LinearKernel",
    "RBFKernel",
    "PolynomialKernel",
    "get_kernel",
    # Models
    "TargetModel",
    "BarlowTwinsAugmentationGenerator",
    # Loss
    "BarlowTwinsLoss",
    # Pre-image
    "PreImageSolver",
    # Utils
    "load_cifar100",
    "set_seed",
    "compute_procrustes_distance",
    "save_image_grid",
]
