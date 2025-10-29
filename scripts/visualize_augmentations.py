"""
Visualize optimal augmentations with different kernels.
Compares Linear, RBF, and Polynomial kernels side-by-side.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.kernels import LinearKernel, RBFKernel, PolynomialKernel
from src.target_models import TargetModel
from src.augmentation_generator import BarlowTwinsAugmentationGenerator
from src.utils import (
    load_cifar100,
    images_to_matrix,
    matrix_to_images,
    save_image_grid,
    create_comparison_grid,
    set_seed
)

set_seed(42)

print("Loading CIFAR-100 data...")
images, labels = load_cifar100(num_samples=500)
X = images_to_matrix(images)
print(f"Loaded {X.shape[1]} images, each with {X.shape[0]} dimensions")

print("\nExtracting target representations with ResNet-18...")
target_model = TargetModel(pca_dim=64, device='cuda' if torch.cuda.is_available() else 'cpu')
F_target = target_model.get_target_representations(torch.from_numpy(images), fit_pca=True).T
print(f"Target representations shape: {F_target.shape}")

# Use only RBF kernels - LinearKernel causes numerical instability
# (rank deficit and Lyapunov solver issues)
kernels = {
    'RBF (sigma=0.5)': RBFKernel(sigma=0.5),
    'RBF (sigma=1.0)': RBFKernel(sigma=1.0),
    'RBF (sigma=2.0)': RBFKernel(sigma=2.0),
    'RBF (sigma=3.0)': RBFKernel(sigma=3.0),
}

print("\n" + "="*60)
print("Generating augmentations with different kernels")
print("="*60)

augmented_images = {}
indices_to_show = np.array(range(50))

for name, kernel in kernels.items():
    print(f"\n[{name}]")
    
    try:
        generator = BarlowTwinsAugmentationGenerator(
            kernel=kernel,
            lambda_ridge=1.0,
            mu_p=1.0,
            check_conditions=False
        )
        
        X_aug = generator.fit_transform(X, F_target, indices=indices_to_show)
        images_aug = matrix_to_images(X_aug, (3, 32, 32))
        augmented_images[name] = images_aug
        
        print(f"  Generated {len(images_aug)} augmented images")
        
        aug_dist = generator.get_augmentation_distribution()
        print(f"  T_H norm: {aug_dist['T_H_frobenius_norm']:.4f}")
        if aug_dist['lyapunov_residual'] is not None:
            print(f"  Lyapunov residual: {aug_dist['lyapunov_residual']:.6f}")
        
    except Exception as e:
        print(f"  Failed: {str(e)}")
        continue

os.makedirs('results/visualizations', exist_ok=True)

print("\n" + "="*60)
print("Saving comparison grids")
print("="*60)

original_subset = images[indices_to_show]

for name, aug_imgs in augmented_images.items():
    comparison = {
        'Original': original_subset,
        name: aug_imgs
    }
    
    filename = f"results/visualizations/{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    create_comparison_grid(comparison, filename, nrow=10)
    print(f"Saved {filename}")

all_methods = {'Original': original_subset}
all_methods.update(augmented_images)
create_comparison_grid(all_methods, 'results/visualizations/all_kernels_comparison.png', nrow=10)
print("\nSaved complete comparison to results/visualizations/all_kernels_comparison.png")

print("\n" + "="*60)
print("Analyzing kernel properties")
print("="*60)

for name, kernel in kernels.items():
    K = kernel(X[:, indices_to_show], X[:, indices_to_show])
    eigenvalues = np.linalg.eigvalsh(K)
    
    print(f"\n[{name}]")
    print(f"  Kernel matrix shape: {K.shape}")
    print(f"  Rank: {np.linalg.matrix_rank(K)}")
    print(f"  Condition number: {np.linalg.cond(K):.2e}")
    print(f"  Min eigenvalue: {eigenvalues[0]:.4f}")
    print(f"  Max eigenvalue: {eigenvalues[-1]:.4f}")

print("\nVisualization complete!")
