"""
Utility Functions

Helper functions for data loading, visualization, metrics, and reproducibility.
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
import random
import os


def set_seed(seed = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cifar100(data_dir="data", train=True, download=True, normalize=True, num_samples=None):
    """
    Load CIFAR-100 dataset.
    Returns images as (n, c, h, w) and labels as (n,).
    """
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    transform = transforms.Compose(transform_list)
    
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=download,
        transform=transform,
    )
    
    if num_samples is not None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        images = torch.stack([dataset[i][0] for i in indices])
        labels = np.array([dataset[i][1] for i in indices])
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False
        )
        images, labels = next(iter(loader))
        labels = labels.numpy()
    
    return images.numpy(), labels


def images_to_matrix(images):
    """
    Convert images to matrix format for kernel methods.
    X is (m, n) where each column is a flattened image.
    """
    n, c, h, w = images.shape
    m = c * h * w
    X = images.reshape(n, m).T  # (m, n)
    return X


def matrix_to_images(X, image_shape):
    """
    Convert matrix back to images.
    """
    c, h, w = image_shape
    n = X.shape[1]
    images = X.T.reshape(n, c, h, w)
    return images


def denormalize_images(images, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]):
    """
    Denormalize images for visualization.
    """
    mean = np.array(mean).reshape(1, -1, 1, 1)
    std = np.array(std).reshape(1, -1, 1, 1)
    
    denorm = images * std + mean
    denorm = np.clip(denorm, 0, 1)
    
    return denorm


def compute_procrustes_distance(F1, F2):
    """
    Compute Procrustes distance between two representation matrices.
    Used in Section 6 of the paper: min_{Q in O(d)} ||F1 - Q F2||_F / n
    """
    # Solve orthogonal Procrustes problem
    Q, scale = orthogonal_procrustes(F2.T, F1.T)
    
    # Compute aligned distance
    F2_aligned = Q.T @ F2
    distance = np.linalg.norm(F1 - F2_aligned, 'fro') / F1.shape[1]
    
    return distance


def save_image_grid(images, save_path, nrow=10, padding=2, normalize=True, title=None):
    """
    Save a grid of images to file.
    """
    # Convert to torch tensor
    images_tensor = torch.from_numpy(images).float()
    
    # Make grid
    grid = torchvision.utils.make_grid(
        images_tensor,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
    )
    
    # Convert to numpy for plotting
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Plot
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_np)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved image grid to {save_path}")


def plot_procrustes_curve(distances, save_path, target_distances=None, labels=None, title="Procrustes Distance During Training"):
    """
    Plot Procrustes distance over training iterations.
    """
    plt.figure(figsize=(10, 6))
    
    if labels is None:
        labels = ["Procrustes Distance"]
    
    plt.plot(distances, linewidth=2, label=labels[0])
    
    if target_distances is not None:
        if len(labels) > 1:
            plt.plot(target_distances, linewidth=2, label=labels[1], linestyle='--')
        else:
            plt.plot(target_distances, linewidth=2, label="To Random", linestyle='--')
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Procrustes Distance", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Procrustes plot to {save_path}")


def generate_random_orthogonal(d, seed=None):
    """
    Generate a random orthogonal matrix via QR decomposition.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random matrix
    A = np.random.randn(d, d)
    
    # QR decomposition gives orthogonal matrix
    Q, _ = np.linalg.qr(A)
    
    return Q


def create_comparison_grid(images_dict, save_path, nrow=10, title="Augmentation Comparison"):
    """
    Create a comparison grid showing multiple augmentation methods.
    images_dict maps method name to (n, c, h, w) images.
    """
    n_methods = len(images_dict)
    
    fig, axes = plt.subplots(n_methods, 1, figsize=(15, 4 * n_methods))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, images) in enumerate(images_dict.items()):
        # Convert to tensor and make grid
        images_tensor = torch.from_numpy(images).float()
        grid = torchvision.utils.make_grid(
            images_tensor[:nrow**2],  # Limit to nrow^2 images
            nrow=nrow,
            padding=2,
            normalize=True,
        )
        grid_np = grid.permute(1, 2, 0).numpy()
        
        # Plot
        axes[idx].imshow(grid_np)
        axes[idx].set_title(method_name, fontsize=14)
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison grid to {save_path}")


class Logger:
    """Simple logger for experiment tracking."""
    
    def __init__(self, log_file=None, verbose=True):
        self.log_file = log_file
        self.verbose = verbose
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message):
        """Log a message."""
        if self.verbose:
            print(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def log_dict(self, d, prefix=""):
        """Log a dictionary."""
        for key, value in d.items():
            if isinstance(value, float):
                self.log(f"{prefix}{key}: {value:.6f}")
            else:
                self.log(f"{prefix}{key}: {value}")
