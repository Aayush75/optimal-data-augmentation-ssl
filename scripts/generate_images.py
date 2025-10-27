"""
Generate augmented images using different kernels for visualization.
"""

import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    load_cifar100,
    images_to_matrix,
    matrix_to_images,
    denormalize_images,
    save_image_grid,
    create_comparison_grid,
    set_seed,
)
from src.kernels import get_kernel
from src.target_models import TargetModel
from src.augmentation_generator import BarlowTwinsAugmentationGenerator


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_images_for_kernel(kernel_name, kernel_config, X_train, F_target, config, output_dir):
    """
    Generate augmented images for a specific kernel.
    """
    print(f"\n{'='*60}")
    print(f"Generating augmentations with {kernel_name} kernel")
    print(f"{'='*60}")
    
    kernel = get_kernel(**kernel_config)
    print(f"Kernel: {kernel}")
    
    generator = BarlowTwinsAugmentationGenerator(
        kernel=kernel,
        lambda_ridge=config['augmentation']['lambda_ridge'],
        mu_p=config['augmentation']['mu_p'],
    )
    
    num_vis = config['visualization']['num_samples']
    indices = np.arange(min(num_vis, X_train.shape[1]))
    
    X_augmented = generator.fit_transform(
        X_train,
        F_target,
        indices=indices,
    )
    
    image_shape = (3, 32, 32)
    images_aug = matrix_to_images(X_augmented, image_shape)
    images_orig = matrix_to_images(X_train[:, indices], image_shape)
    
    images_aug_vis = denormalize_images(images_aug)
    images_orig_vis = denormalize_images(images_orig)
    
    save_image_grid(
        images_orig_vis,
        os.path.join(output_dir, f"{kernel_name}_original.png"),
        nrow=10,
        title=f"Original Images",
    )
    
    save_image_grid(
        images_aug_vis,
        os.path.join(output_dir, f"{kernel_name}_augmented.png"),
        nrow=10,
        title=f"Augmented Images ({kernel_name})",
    )
    
    return {
        'original': images_orig_vis,
        'augmented': images_aug_vis,
        'kernel_name': kernel_name,
    }


def main(config_path="configs/barlow_twins_cifar100.yaml"):
    """
    Main function to generate augmented images.
    """
    config = load_config(config_path)
    
    set_seed(config['experiment']['seed'])
    
    output_dir = os.path.join(
        config['experiment']['output_dir'],
        'generated_images'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("AUGMENTED IMAGE GENERATION")
    print("="*60)
    print(f"Configuration: {config_path}")
    print(f"Output directory: {output_dir}")
    
    print("\nLoading CIFAR-100...")
    images, labels = load_cifar100(
        data_dir=config['data']['data_dir'],
        train=True,
        download=True,
        normalize=config['data']['normalize'],
        num_samples=config['data']['num_train_samples'],
    )
    print(f"Loaded {len(images)} images")
    
    X_train = images_to_matrix(images)
    print(f"Data matrix shape: {X_train.shape}")
    
    print("\nLoading target model...")
    target_model = TargetModel(
        architecture=config['target_model']['architecture'],
        pretrained=config['target_model']['pretrained'],
        weights=config['target_model']['weights'],
        pca_dim=config['target_model']['pca_dim'],
        device=config['experiment']['device'],
    )
    
    print("Extracting target representations...")
    F_target = target_model.get_target_representations(
        torch.from_numpy(images).float(),
        fit_pca=True,
    )
    print(f"Target representations shape: {F_target.shape}")
    F_target = F_target.T
    
    print("\n" + "="*60)
    print("GENERATING AUGMENTATIONS")
    print("="*60)
    
    with open("configs/kernels.yaml", 'r') as f:
        kernel_configs = yaml.safe_load(f)
    
    kernels_to_use = config['visualization'].get('compare_kernels', ['linear', 'rbf'])
    
    results = {}
    for kernel_name in kernels_to_use:
        if kernel_name in kernel_configs['kernels']:
            kernel_config = kernel_configs['kernels'][kernel_name]
        else:
            kernel_config = {
                'type': config['kernel']['type'],
                **config['kernel'].get(config['kernel']['type'], {})
            }
        
        result = generate_images_for_kernel(
            kernel_name,
            kernel_config,
            X_train,
            F_target,
            config,
            output_dir,
        )
        
        results[kernel_name] = result
    
    if len(results) > 1:
        print("\n" + "="*60)
        print("CREATING COMPARISON GRIDS")
        print("="*60)
        
        originals = {
            f"{k} (Original)": v['original']
            for k, v in results.items()
        }
        
        create_comparison_grid(
            originals,
            os.path.join(output_dir, "comparison_original.png"),
            nrow=10,
            title="Original Images - Different Kernels",
        )
        
        augmented = {
            f"{k} Kernel": v['augmented']
            for k, v in results.items()
        }
        
        create_comparison_grid(
            augmented,
            os.path.join(output_dir, "comparison_augmented.png"),
            nrow=10,
            title="Augmented Images - Different Kernels",
        )
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate augmented CIFAR-100 images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/barlow_twins_cifar100.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    
    main(args.config)
