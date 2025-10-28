"""
Run complete experiment: load data, extract targets, generate augmentations,
train with Barlow Twins, track Procrustes distance, save results.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    load_cifar100,
    images_to_matrix,
    set_seed,
    compute_procrustes_distance,
    plot_procrustes_curve,
    generate_random_orthogonal,
)
from src.kernels import get_kernel, check_kernel_conditions
from src.target_models import TargetModel, check_representation_conditions
from src.augmentation_generator import BarlowTwinsAugmentationGenerator
from src.losses import BarlowTwinsLoss, barlow_twins_loss_numpy


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(config_path="configs/barlow_twins_cifar100.yaml", resume=False):
    """
    Run the complete optimal augmentation experiment.
    
    Args:
        config_path: Path to configuration file
        resume: If True, try to resume from checkpoint
    """
    config = load_config(config_path)
    
    set_seed(config['experiment']['seed'])
    
    output_dir = config['experiment']['output_dir']
    plots_dir = os.path.join(output_dir, 'plots')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'generator_checkpoint.npz')
    has_checkpoint = os.path.exists(checkpoint_path)
    
    print("="*80)
    print("OPTIMAL DATA AUGMENTATION EXPERIMENT - BARLOW TWINS")
    print("="*80)
    print(f"Configuration: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Random seed: {config['experiment']['seed']}")
    
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    images, labels = load_cifar100(
        data_dir=config['data']['data_dir'],
        train=True,
        download=True,
        normalize=config['data']['normalize'],
        num_samples=config['data']['num_train_samples'],
    )
    print(f"Loaded {len(images)} CIFAR-100 images")
    print(f"Image shape: {images.shape}")
    print(f"Number of unique classes: {len(np.unique(labels))}")
    
    X_train = images_to_matrix(images)
    print(f"Data matrix shape: {X_train.shape}")
    
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING TARGET REPRESENTATIONS")
    print("="*80)
    
    target_model = TargetModel(
        architecture=config['target_model']['architecture'],
        pretrained=config['target_model']['pretrained'],
        weights=config['target_model']['weights'],
        pca_dim=config['target_model']['pca_dim'],
        device=config['experiment']['device'],
    )
    print(f"Target model: {target_model}")
    
    F_target = target_model.get_target_representations(
        torch.from_numpy(images).float(),
        fit_pca=True,
    )
    print(f"Target representations shape: {F_target.shape}")
    
    F_target = F_target.T
    print(f"Reformatted to: {F_target.shape}")
    
    cond_check = check_representation_conditions(F_target)
    print("\nCondition 3.2 Check (Full Rank Covariance):")
    for key, val in cond_check.items():
        print(f"  {key}: {val}")
    
    print("\n" + "="*80)
    print("STEP 3: GENERATING OPTIMAL AUGMENTATIONS")
    print("="*80)
    
    # Check if we should resume from checkpoint
    if resume and has_checkpoint:
        print(f"\nFound checkpoint at: {checkpoint_path}")
        print("Loading from checkpoint...")
        
        checkpoint = np.load(checkpoint_path)
        
        kernel_type = config['kernel']['type']
        kernel_params = config['kernel'].get(kernel_type, {})
        kernel = get_kernel(kernel_type, **kernel_params)
        
        generator = BarlowTwinsAugmentationGenerator(
            kernel=kernel,
            lambda_ridge=config['augmentation']['lambda_ridge'],
            mu_p=config['augmentation']['mu_p'],
            check_conditions=False,
        )
        
        # Restore generator state
        generator.K = checkpoint['K']
        generator.C = checkpoint['C']
        generator.B = checkpoint['B']
        generator.T_H_matrix = checkpoint['T_H_matrix']
        generator.X_train = checkpoint['X_train']
        
        F_target = checkpoint['F_target']
        
        print("Checkpoint loaded successfully!")
        print("Skipping Lyapunov equation solving...")
        
        aug_info = generator.get_augmentation_distribution()
        
    else:
        if resume and not has_checkpoint:
            print("\nNo checkpoint found. Starting from scratch...")
        
        kernel_type = config['kernel']['type']
        kernel_params = config['kernel'].get(kernel_type, {})
        kernel = get_kernel(kernel_type, **kernel_params)
        print(f"Kernel: {kernel}")
        
        generator = BarlowTwinsAugmentationGenerator(
            kernel=kernel,
            lambda_ridge=config['augmentation']['lambda_ridge'],
            mu_p=config['augmentation']['mu_p'],
            check_conditions=True,
        )
        
        generator.fit(X_train, F_target)
        
        aug_info = generator.get_augmentation_distribution()
        print("\nAugmentation Distribution Info:")
        print(f"  Min eigenvalue: {aug_info['min_eigenvalue']:.6e}")
        print(f"  Max eigenvalue: {aug_info['max_eigenvalue']:.6e}")
        print(f"  Condition number: {aug_info['condition_number']:.6e}")
        
        # Save checkpoint after Lyapunov equation is solved
        checkpoint_path = os.path.join(checkpoint_dir, 'generator_checkpoint.npz')
        print(f"\nSaving checkpoint to: {checkpoint_path}")
        np.savez(
            checkpoint_path,
            K=generator.K,
            C=generator.C,
            B=generator.B,
            T_H_matrix=generator.T_H_matrix,
            X_train=generator.X_train,
            F_target=F_target,
        )
        print("Checkpoint saved successfully!")
    
    print("\nAugmentation Distribution Info:")
    print(f"  Min eigenvalue: {aug_info['min_eigenvalue']:.6e}")
    print(f"  Max eigenvalue: {aug_info['max_eigenvalue']:.6e}")
    print(f"  Condition number: {aug_info['condition_number']:.6e}")
    
    print("\n" + "="*80)
    print("STEP 4: TRAINING WITH BARLOW TWINS LOSS")
    print("="*80)
    
    device = config['experiment']['device']
    d, n = F_target.shape
    
    C_learned = torch.randn(d, n, requires_grad=True, device=device)
    
    K_tensor = torch.from_numpy(generator.K).float().to(device)
    
    optimizer = optim.Adam([C_learned], lr=config['barlow_twins']['learning_rate'])
    
    bt_loss = BarlowTwinsLoss(
        lambda_param=config['barlow_twins']['lambda_param'],
        normalize=config['barlow_twins']['normalize_repr'],
    )
    
    num_epochs = config['barlow_twins']['num_epochs']
    log_interval = config['logging']['log_interval']
    
    procrustes_to_target = []
    procrustes_to_random = []
    
    Q_random = generate_random_orthogonal(d)
    F_random = Q_random @ F_target
    
    print(f"Training for {num_epochs} epochs...")
    print(f"Batch size: all {n} samples")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        if np.random.rand() < 0.5:
            Z1 = C_learned @ K_tensor
        else:
            T_matrix = torch.from_numpy(generator.T_H_matrix).float().to(device)
            Z1 = C_learned @ K_tensor @ T_matrix
        
        if np.random.rand() < 0.5:
            Z2 = C_learned @ K_tensor
        else:
            T_matrix = torch.from_numpy(generator.T_H_matrix).float().to(device)
            Z2 = C_learned @ K_tensor @ T_matrix
        
        loss, loss_info = bt_loss(Z1, Z2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        F_learned_np = (C_learned @ K_tensor).detach().cpu().numpy()
        dist_target = compute_procrustes_distance(F_learned_np, F_target)
        dist_random = compute_procrustes_distance(F_learned_np, F_random)
        
        procrustes_to_target.append(dist_target)
        procrustes_to_random.append(dist_random)
        
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {loss_info['loss']:.6f} | "
                f"On-diag: {loss_info['on_diag_loss']:.6f} | "
                f"Off-diag: {loss_info['off_diag_loss']:.6f} | "
                f"Procrustes: {dist_target:.6f}"
            )
    
    print("\n" + "="*80)
    print("STEP 5: FINAL EVALUATION")
    print("="*80)
    
    F_learned_final = (C_learned @ K_tensor).detach().cpu().numpy()
    
    optimality = generator.verify_optimality(
        F_learned_final,
        F_target,
        tolerance=1e-3,
    )
    print("\nOptimality Verification (Theorem 4.4):")
    for key, val in optimality.items():
        if key != 'optimal_rotation':
            print(f"  {key}: {val}")
    
    final_dist_target = procrustes_to_target[-1]
    final_dist_random = procrustes_to_random[-1]
    
    print(f"\nFinal Procrustes Distances:")
    print(f"  To target: {final_dist_target:.6f}")
    print(f"  To random: {final_dist_random:.6f}")
    print(f"  Improvement: {final_dist_random - final_dist_target:.6f}")
    
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)
    
    plot_procrustes_curve(
        procrustes_to_target,
        os.path.join(plots_dir, 'procrustes_distance.png'),
        target_distances=procrustes_to_random,
        labels=['To Target', 'To Random'],
        title='Procrustes Distance During Training (Barlow Twins)',
    )
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results = {
        'config': config,
        'final_procrustes_to_target': float(final_dist_target),
        'final_procrustes_to_random': float(final_dist_random),
        'optimality_check': convert_to_serializable({k: v for k, v in optimality.items() if k != 'optimal_rotation'}),
        'augmentation_info': {
            'min_eigenvalue': float(aug_info['min_eigenvalue']),
            'max_eigenvalue': float(aug_info['max_eigenvalue']),
            'condition_number': float(aug_info['condition_number']),
        },
        'condition_checks': convert_to_serializable(cond_check),
    }
    
    import json
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    np.save(
        os.path.join(output_dir, 'procrustes_to_target.npy'),
        np.array(procrustes_to_target)
    )
    np.save(
        os.path.join(output_dir, 'procrustes_to_random.npy'),
        np.array(procrustes_to_random)
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run optimal augmentation experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/barlow_twins_cifar100.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    
    args = parser.parse_args()
    
    run_experiment(args.config, resume=args.resume)
