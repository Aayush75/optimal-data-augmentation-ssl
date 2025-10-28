# Usage Guide: Optimal Data Augmentations for Self-Supervised Learning

Complete guide for using this implementation of "A Theoretical Characterization of Optimal Data Augmentations in Self-Supervised Learning" (arXiv:2411.01767v3).

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Scripts Overview](#scripts-overview)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.6.9 or higher
- PyTorch with CUDA support (tested with torchvision 0.11.2+cu102)
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM

**Compatibility Note**: This codebase is specifically designed to work with Python 3.6.9 and torchvision 0.11.2. It uses the legacy `models.resnet18(pretrained=True)` API instead of the newer weights enum API.

### Setup

```bash
# Clone the repository
git clone https://github.com/Aayush75/optimal-data-augmentation-ssl.git
cd optimal-data-augmentation-ssl

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained ResNet-18
python scripts/download_models.py
```

### Verify Installation

```bash
python -c "import torch; import torchvision; import scipy; import sklearn; print('All packages installed successfully')"
```

## Quick Start

### Run Complete Experiment

```bash
python scripts/run_experiment.py
```

This runs the full pipeline (30-60 min on GPU):
- Loads CIFAR-100 data (10,000 train, 2,000 test)
- Extracts target representations from ResNet-18
- Solves Lyapunov equation for optimal augmentations
- Trains with Barlow Twins loss (3,000 epochs)
- Tracks Procrustes distance to verify optimality
- Saves results to `results/`

### Generate Augmented Images

```bash
python scripts/generate_images.py
```

Creates augmented image grids showing original and augmented versions.

Generates augmented images with different kernel types and saves to `results/generated_images/`.

### Visualize Different Kernels

```bash
python scripts/visualize_augmentations.py
```

Compares Linear, RBF, and Polynomial kernel augmentations.

### Analyze Training Progress

```bash
python scripts/analyze_procrustes.py
```

Plots Procrustes distance showing convergence to targets.

## Scripts Overview

### download_models.py
Downloads pretrained ResNet-18 and saves locally.

```bash
python scripts/download_models.py
```

### run_experiment.py
Complete experimental pipeline from the paper.

```bash
python scripts/run_experiment.py
```

Pipeline steps:
1. Load CIFAR-100 data
2. Extract ResNet-18 features
3. Solve Lyapunov equation
4. Train with Barlow Twins
5. Track Procrustes distance
6. Save results

### generate_images.py
Generate augmented image grids.

```bash
python scripts/generate_images.py
```

### visualize_augmentations.py
Compare kernels side-by-side.

```bash
python scripts/visualize_augmentations.py
```

### analyze_procrustes.py
Plot distance curves.

```bash
python scripts/analyze_procrustes.py
```

## Configuration

### Main Config

Edit `configs/barlow_twins_cifar100.yaml`:

```yaml
data:
  dataset: "cifar100"
  num_train_samples: 10000  # Adjust for speed
  num_test_samples: 2000

target_model:
  model_name: "resnet18"
  pretrained: true
  pca_dim: 64

barlow_twins:
  num_epochs: 3000
  batch_size: 512
  learning_rate: 0.001
  lambda_param: 0.005

kernel:
  type: "rbf"  # Options: linear, rbf, polynomial
  rbf:
    sigma: 1.0
```

### Kernel Options

Edit `configs/kernels.yaml`:

```yaml
linear:
  # No parameters

rbf:
  sigma: 1.0  # Try 0.5, 1.0, 2.0

polynomial:
  degree: 3  # Try 2, 3, 4
  coef0: 1.0
```

## API Reference

### Load Data

```python
from src.utils import load_cifar100, images_to_matrix, matrix_to_images

# Load CIFAR-100
images, labels = load_cifar100(num_samples=1000)

# Convert to matrix (n_samples, n_features)
X = images_to_matrix(images)

# Convert back to images
images_reconstructed = matrix_to_images(X, (3, 32, 32))
```

### Kernels

```python
from src.kernels import LinearKernel, RBFKernel, PolynomialKernel
import numpy as np

X = np.random.randn(100, 3072)

# Linear kernel
kernel = LinearKernel()
K = kernel.compute(X, X)

# RBF kernel
kernel = RBFKernel(sigma=1.0)
K = kernel.compute(X, X)

# Polynomial kernel
kernel = PolynomialKernel(degree=3, coef0=1.0)
K = kernel.compute(X, X)
```

### Target Model

```python
from src.target_models import TargetModel
import torch

# Initialize model
model = TargetModel(
    model_name="resnet18",
    pretrained=True,
    pca_dim=64
)

# Get representations
images_tensor = torch.randn(100, 3, 32, 32)
features = model.get_target_representations(images_tensor, fit_pca=True)
```

### Augmentation Generator

```python
from src.augmentation_generator import BarlowTwinsAugmentationGenerator
from src.kernels import RBFKernel

# Setup
kernel = RBFKernel(sigma=1.0)
generator = BarlowTwinsAugmentationGenerator(
    kernel=kernel,
    mu_k=0.01,
    mu_p=0.01
)

# Generate augmentations
X_aug = generator.fit_transform(X, F_target)
```

### Barlow Twins Loss

```python
from src.losses import BarlowTwinsLoss
import torch

# Initialize loss
loss_fn = BarlowTwinsLoss(lambda_param=0.005)

# Compute loss on two views
z1 = torch.randn(512, 64)
z2 = torch.randn(512, 64)
loss = loss_fn(z1, z2)
```

### Pre-image Solver

```python
from src.preimage import PreImageSolver

solver = PreImageSolver(kernel=kernel, mu_p=0.01, clip_range=(0, 1))
X_aug = solver.solve_batch(X, K, theta_aug.T)
```

### Utilities

```python
from src.utils import compute_procrustes_distance, save_image_grid

# Compute distance
distance = compute_procrustes_distance(F_learned, F_target)

# Save images
save_image_grid(images, "grid.png", nrow=10)
```

## Complete Example

```python
from src.utils import load_cifar100, images_to_matrix, matrix_to_images
from src.target_models import TargetModel
from src.kernels import RBFKernel
from src.augmentation_generator import BarlowTwinsAugmentationGenerator
import torch

# Load data
images, labels = load_cifar100(num_samples=1000)
X = images_to_matrix(images)

# Get target features
target_model = TargetModel(pca_dim=64)
F_target = target_model.get_target_representations(
    torch.from_numpy(images), fit_pca=True
).T

# Generate augmentations
kernel = RBFKernel(sigma=1.0)
generator = BarlowTwinsAugmentationGenerator(kernel=kernel)
X_aug = generator.fit_transform(X, F_target)

# Convert back to images
images_aug = matrix_to_images(X_aug, (3, 32, 32))
print(f"Generated {len(images_aug)} augmented images")
```

## Troubleshooting

### Common Issues

#### Out of Memory
Reduce batch size or number of samples:
```yaml
data:
  num_train_samples: 1000  # Down from 10000
```

#### Slow Training
- Use GPU instead of CPU
- Reduce PCA dimension: `pca_dim: 32`
- Use fewer epochs: `num_epochs: 1000`
- Use Linear kernel (faster than RBF)

#### Kernel Matrix Issues
If you see "Kernel matrix is not full rank":
- Ensure data points are distinct
- Increase regularization: `mu_k: 0.1`
- Try RBF kernel (always full rank for distinct points)

#### Poor Image Quality
If augmented images look bad:
- Adjust `mu_p` (try 0.1 to 10)
- Try different kernel
- Check target representations quality

### Performance Tips

1. Start with 1000 samples for testing
2. Use GPU for 10-100x speedup
3. Cache intermediate results
4. Profile code to find bottlenecks

### Getting Help

Open an issue on GitHub with:
- Configuration file
- Error message
- System info (OS, Python version, GPU)

## Advanced Topics

### Custom Target Model

Use your own pretrained model:

```python
class CustomTargetModel:
    def __init__(self, model):
        self.model = model
    
    def get_target_representations(self, images):
        with torch.no_grad():
            return self.model(images).numpy()
```

### Custom Kernel

Implement your own kernel:

```python
from src.kernels import BaseKernel
import numpy as np

class CosineSimilarityKernel(BaseKernel):
    def compute(self, X, Y=None):
        if Y is None:
            Y = X
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        return X_norm @ Y_norm.T
```

## References

- Paper: [arXiv:2411.01767v3](https://arxiv.org/abs/2411.01767v3)
- Barlow Twins: Zbontar et al. (2021)
- Pre-image method: Honeine & Richard (2011)

## License

MIT License
