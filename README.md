# Optimal Data Augmentations for Self-Supervised Learning

Implementation of "A Theoretical Characterization of Optimal Data Augmentations in Self-Supervised Learning" (arXiv:2411.01767v3).

## Overview

This repository implements Algorithm 1 from the paper, which generates optimal augmentations for Barlow Twins using kernel methods and Lyapunov equations.

### Key Features

- Barlow Twins optimal augmentation generation
- Multiple kernel functions (Linear, RBF, Polynomial)
- ResNet-18 target representations
- CIFAR-100 dataset support
- Procrustes distance tracking
- Pre-image problem solver

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py
```

## Quick Start

### Run Complete Experiment

```bash
python scripts/run_experiment.py
```

This runs the complete pipeline (30-60 min on GPU):
- Loads CIFAR-100 data
- Extracts target representations using ResNet-18
- Solves Lyapunov equation for optimal augmentations
- Trains with Barlow Twins loss
- Saves results and visualizations to `results/`

### Generate Custom Images

```bash
python scripts/generate_images.py
```

Creates augmented images with different kernels and saves visualization grids.

### Visualize Augmentations

```bash
python scripts/visualize_augmentations.py
```

Compare augmentations across Linear, RBF, and Polynomial kernels.

### Analyze Results

```bash
python scripts/analyze_procrustes.py
```

Plot Procrustes distance curves showing convergence to target representations.

## Project Structure

```
optimal-data-augmentation-ssl/
├── src/
│   ├── kernels.py                 # Kernel functions (Linear, RBF, Polynomial)
│   ├── target_models.py           # ResNet-18 feature extraction
│   ├── losses.py                  # Barlow Twins loss implementation
│   ├── preimage.py                # Pre-image problem solver
│   ├── augmentation_generator.py  # Algorithm 1 from paper
│   └── utils.py                   # Data loading and utilities
├── scripts/
│   ├── download_models.py         # Download pretrained ResNet-18
│   ├── run_experiment.py          # Full experiment pipeline
│   ├── generate_images.py         # Generate augmented images
│   ├── visualize_augmentations.py # Compare kernel augmentations
│   └── analyze_procrustes.py      # Plot distance curves
├── configs/
│   ├── barlow_twins_cifar100.yaml # Main experiment config
│   └── kernels.yaml               # Kernel parameters
└── requirements.txt               # Python dependencies
```

## Usage Examples

### Basic Usage

```python
from src.utils import load_cifar100, images_to_matrix, matrix_to_images
from src.target_models import TargetModel
from src.kernels import RBFKernel
from src.augmentation_generator import BarlowTwinsAugmentationGenerator
import torch

# Load CIFAR-100 data
images, labels = load_cifar100(num_samples=1000)
X = images_to_matrix(images)

# Get target representations from ResNet-18
target_model = TargetModel(pca_dim=64)
F_target = target_model.get_target_representations(
    torch.from_numpy(images), 
    fit_pca=True
).T

# Generate optimal augmentations
kernel = RBFKernel(sigma=1.0)
generator = BarlowTwinsAugmentationGenerator(kernel=kernel)
X_aug = generator.fit_transform(X, F_target)

# Convert back to images
images_aug = matrix_to_images(X_aug, (3, 32, 32))
```

# Convert back to images
images_aug = matrix_to_images(X_aug, (3, 32, 32))
```

### Custom Configuration

You can customize experiments by editing `configs/barlow_twins_cifar100.yaml`:

```yaml
data:
  num_train_samples: 5000  # Reduce for faster testing
  num_test_samples: 1000

target_model:
  pca_dim: 32  # Lower dimension for speed

barlow_twins:
  num_epochs: 1000  # Adjust training time
  learning_rate: 0.001

kernel:
  type: "rbf"  # or "linear", "polynomial"
  rbf:
    sigma: 2.0
```

Or use different kernels programmatically:

```python
from src.kernels import LinearKernel, RBFKernel, PolynomialKernel

# Linear kernel
kernel = LinearKernel()

# RBF kernel with custom sigma
kernel = RBFKernel(sigma=2.0)

# Polynomial kernel
kernel = PolynomialKernel(degree=3, coef0=1.0)
```

## How It Works

The implementation follows Algorithm 1 from the paper:

1. **Compute kernel matrix K** - Encodes similarity between images
2. **Solve ridge regression** - Find optimal coefficient matrix C
3. **Solve Lyapunov equation** - KB + BK = 2n*RHS for matrix B
4. **Construct transformation** - T_H = K^(-1/2) B K^(-1/2)
5. **Solve pre-image problem** - Map augmented features back to images

The optimal augmentations guarantee that learned representations match target representations up to an orthogonal transformation (Theorem 4.4).

## Results

After running experiments, you'll find:

- `results/augmented_images/` - Generated augmentation examples
- `results/procrustes_distances.npy` - Convergence metrics
- `results/procrustes_curve.png` - Visualization of training progress

The Procrustes distance should decrease over training, showing that learned representations converge to the target.

## Citation

```bibtex
@article{feigin2024theoretical,
  title={A Theoretical Characterization of Optimal Data Augmentations in Self-Supervised Learning},
  author={Feigin, Elad and Fleissner, Florian and Ghoshdastidar, Debarghya},
  journal={arXiv preprint arXiv:2411.01767},
  year={2024}
}
```

## License

MIT License
