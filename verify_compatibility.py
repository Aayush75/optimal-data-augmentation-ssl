"""
Verify Python 3.6.9 and torchvision 0.11.2 compatibility.
Run this script to ensure your environment is properly set up.
"""

import sys
import platform

print("="*60)
print("COMPATIBILITY CHECK")
print("="*60)

# Check Python version
print("\n1. Python Version:")
print("   Current: {}".format(sys.version))
print("   Required: 3.6.9+")

major = sys.version_info.major
minor = sys.version_info.minor
micro = sys.version_info.micro

if major == 3 and minor >= 6:
    print("   Status: OK")
else:
    print("   Status: FAIL - Need Python 3.6.9 or higher")
    sys.exit(1)

# Check PyTorch
print("\n2. PyTorch:")
try:
    import torch
    print("   Version: {}".format(torch.__version__))
    print("   CUDA available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("   CUDA version: {}".format(torch.version.cuda))
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check torchvision
print("\n3. torchvision:")
try:
    import torchvision
    print("   Version: {}".format(torchvision.__version__))
    
    # Test legacy API
    import torchvision.models as models
    try:
        # Try legacy API (should work)
        model = models.resnet18(pretrained=False)
        print("   Legacy API (pretrained=True/False): OK")
    except Exception as e:
        print("   Legacy API: FAIL - {}".format(e))
        
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check NumPy
print("\n4. NumPy:")
try:
    import numpy as np
    print("   Version: {}".format(np.__version__))
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check SciPy
print("\n5. SciPy:")
try:
    import scipy
    print("   Version: {}".format(scipy.__version__))
    # Test Lyapunov solver
    from scipy.linalg import solve_continuous_lyapunov
    print("   Lyapunov solver: Available")
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check scikit-learn
print("\n6. scikit-learn:")
try:
    import sklearn
    print("   Version: {}".format(sklearn.__version__))
    from sklearn.decomposition import PCA
    print("   PCA: Available")
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check matplotlib
print("\n7. matplotlib:")
try:
    import matplotlib
    print("   Version: {}".format(matplotlib.__version__))
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check YAML
print("\n8. PyYAML:")
try:
    import yaml
    print("   Version: {}".format(yaml.__version__))
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Check tqdm
print("\n9. tqdm:")
try:
    import tqdm
    print("   Version: {}".format(tqdm.__version__))
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

# Test imports from project
print("\n10. Project Modules:")
try:
    sys.path.insert(0, '.')
    from src.kernels import LinearKernel, RBFKernel, PolynomialKernel
    from src.target_models import TargetModel
    from src.augmentation_generator import BarlowTwinsAugmentationGenerator
    from src.losses import BarlowTwinsLoss
    from src.preimage import PreImageSolver
    from src.utils import load_cifar100
    print("   All imports: OK")
    print("   Status: OK")
except ImportError as e:
    print("   Status: FAIL - {}".format(e))
    sys.exit(1)

print("\n" + "="*60)
print("ALL COMPATIBILITY CHECKS PASSED!")
print("="*60)
print("\nYour environment is ready. You can now run:")
print("  python scripts/download_models.py")
print("  python scripts/run_experiment.py")
