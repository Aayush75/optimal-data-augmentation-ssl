"""
Test all major functions to catch any runtime errors before running full experiments.
"""

import sys
import os
sys.path.insert(0, '.')

print("="*60)
print("COMPREHENSIVE ERROR CHECK")
print("="*60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from src.kernels import LinearKernel, RBFKernel, PolynomialKernel, get_kernel
    from src.target_models import TargetModel
    from src.augmentation_generator import BarlowTwinsAugmentationGenerator
    from src.losses import BarlowTwinsLoss
    from src.preimage import PreImageSolver
    from src.utils import load_cifar100, images_to_matrix, matrix_to_images
    print("   OK - All imports successful")
except Exception as e:
    print("   FAIL - Import error: {}".format(e))
    sys.exit(1)

# Test 2: Test get_kernel function with different inputs
print("\n2. Testing get_kernel function...")
try:
    k1 = get_kernel('linear')
    k2 = get_kernel('rbf', sigma=1.0)
    k3 = get_kernel('polynomial', degree=3, coef0=1.0)
    print("   OK - get_kernel works with all kernel types")
except Exception as e:
    print("   FAIL - get_kernel error: {}".format(e))
    sys.exit(1)

# Test 3: Test kernel config parsing (like in scripts)
print("\n3. Testing kernel config parsing...")
try:
    import yaml
    with open('configs/kernels.yaml', 'r') as f:
        kernel_configs = yaml.safe_load(f)
    
    # Test parsing like in generate_images.py
    kernel_name = 'rbf_medium'
    kernel_config = kernel_configs['kernels'][kernel_name]
    kernel_type = kernel_config['type']
    kernel_params = {k: v for k, v in kernel_config.items() if k not in ['type', 'description']}
    kernel = get_kernel(kernel_type, **kernel_params)
    print("   OK - Kernel config parsing works")
except Exception as e:
    print("   FAIL - Config parsing error: {}".format(e))
    sys.exit(1)

# Test 4: Test main experiment config
print("\n4. Testing main experiment config...")
try:
    with open('configs/barlow_twins_cifar100.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test kernel parsing like in run_experiment.py
    kernel_type = config['kernel']['type']
    kernel_params = config['kernel'].get(kernel_type, {})
    kernel = get_kernel(kernel_type, **kernel_params)
    print("   OK - Main config parsing works")
except Exception as e:
    print("   FAIL - Main config error: {}".format(e))
    sys.exit(1)

# Test 5: Test small data pipeline
print("\n5. Testing data pipeline with small dataset...")
try:
    import numpy as np
    import torch
    
    # Create small dummy data
    n_samples = 10
    dummy_images = np.random.randn(n_samples, 3, 32, 32).astype(np.float32)
    X = images_to_matrix(dummy_images)
    
    # Create dummy target representations
    F_target = np.random.randn(32, n_samples).astype(np.float32)
    
    print("   OK - Data matrices created: X={}, F_target={}".format(X.shape, F_target.shape))
except Exception as e:
    print("   FAIL - Data pipeline error: {}".format(e))
    sys.exit(1)

# Test 6: Test augmentation generator
print("\n6. Testing augmentation generator...")
try:
    kernel = RBFKernel(sigma=1.0)
    generator = BarlowTwinsAugmentationGenerator(
        kernel=kernel,
        lambda_ridge=1.0,
        mu_p=1.0,
        check_conditions=False
    )
    
    indices = np.array([0, 1, 2])
    X_aug = generator.fit_transform(X, F_target, indices=indices)
    
    print("   OK - Augmentation generator works: X_aug={}".format(X_aug.shape))
except Exception as e:
    print("   FAIL - Generator error: {}".format(e))
    sys.exit(1)

# Test 7: Test Barlow Twins loss
print("\n7. Testing Barlow Twins loss...")
try:
    loss_fn = BarlowTwinsLoss(lambda_param=0.005, normalize=False)
    
    Z1 = torch.randn(10, 32)
    Z2 = torch.randn(10, 32)
    
    loss, info = loss_fn(Z1, Z2)
    
    print("   OK - Loss computed: {:.6f}".format(loss.item()))
except Exception as e:
    print("   FAIL - Loss error: {}".format(e))
    sys.exit(1)

# Test 8: Test pre-image solver
print("\n8. Testing pre-image solver...")
try:
    solver = PreImageSolver(mu_p=1.0)
    
    # Small test
    X_small = np.random.randn(100, 5)
    K_small = np.dot(X_small.T, X_small)
    theta = np.random.randn(100, 3)
    
    X_preimage = solver.solve_batch(X_small, K_small, theta.T)
    
    print("   OK - Pre-image solver works: {}".format(X_preimage.shape))
except Exception as e:
    print("   FAIL - Pre-image error: {}".format(e))
    sys.exit(1)

# Test 9: Check for common config issues
print("\n9. Checking config file completeness...")
try:
    required_keys = [
        'experiment', 'data', 'target_model', 'kernel', 
        'augmentation', 'barlow_twins', 'logging', 'visualization'
    ]
    
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError("Missing config keys: {}".format(missing))
    
    # Check nested keys
    if 'normalize_repr' not in config['barlow_twins']:
        raise ValueError("Missing 'normalize_repr' in barlow_twins config")
    
    if 'log_interval' not in config['logging']:
        raise ValueError("Missing 'log_interval' in logging config")
    
    print("   OK - All required config keys present")
except Exception as e:
    print("   FAIL - Config check error: {}".format(e))
    sys.exit(1)

# Test 10: Test TargetModel initialization
print("\n10. Testing TargetModel...")
try:
    # Test with pretrained=False to avoid downloading
    model = TargetModel(
        architecture='resnet18',
        pretrained=False,
        pca_dim=32,
        device='cpu'
    )
    
    dummy_input = torch.randn(5, 3, 32, 32)
    features = model.get_target_representations(dummy_input, fit_pca=True)
    
    print("   OK - TargetModel works: features={}".format(features.shape))
except Exception as e:
    print("   FAIL - TargetModel error: {}".format(e))
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nYou can now run:")
print("  python scripts/download_models.py")
print("  python scripts/run_experiment.py")
print("  python scripts/generate_images.py")
print("  python scripts/visualize_augmentations.py")
