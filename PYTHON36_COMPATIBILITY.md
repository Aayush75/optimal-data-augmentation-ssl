# Python 3.6.9 & torchvision 0.11.2 Compatibility

## Overview

This repository has been updated to be fully compatible with:
- **Python 3.6.9**
- **torchvision 0.11.2+cu102**
- **PyTorch 1.10.0+cu102** (or compatible version)

## Key Changes Made

### 1. **ResNet-18 Weight Loading**
- **Old (incompatible)**: `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
- **New (compatible)**: `models.resnet18(pretrained=True)`

The newer `weights` API was introduced in torchvision 0.13+. We now use the legacy `pretrained=True/False` parameter which works with torchvision 0.11.2.

### 2. **Files Updated**
- `src/target_models.py` - Updated `_load_model()` method
- `scripts/download_models.py` - Removed `ResNet18_Weights` import
- `requirements.txt` - Adjusted version constraints
- `README.md` - Added compatibility note
- `USAGE.md` - Updated prerequisites

### 3. **Python 3.6.9 Compatibility**
The code uses features available in Python 3.6.9:
- ✅ f-strings (introduced in Python 3.6)
- ✅ Type hints in comments (no runtime impact)
- ✅ `os.makedirs(exist_ok=True)`
- ❌ No walrus operator `:=` (requires 3.8+)
- ❌ No positional-only parameters `/` (requires 3.8+)
- ❌ No `dict | dict` merge operator (requires 3.9+)

## Verification

### Step 1: Check Your Environment

```bash
python verify_compatibility.py
```

This script checks:
- Python version
- PyTorch and torchvision versions
- All required packages
- Legacy API compatibility
- Project module imports

### Step 2: Test Model Download

```bash
python scripts/download_models.py
```

Expected output:
```
Downloading pretrained ResNet-18...
Model saved to models/pretrained/resnet18_imagenet.pth
Model size: XX.XX MB
Verifying model at models/pretrained/resnet18_imagenet.pth...
Model loaded successfully
Forward pass successful
  Output shape: torch.Size([1, 1000])
```

### Step 3: Run Quick Test

```python
python -c "
from src.target_models import TargetModel
import torch

model = TargetModel(pretrained=True)
dummy = torch.randn(10, 3, 32, 32)
features = model.get_target_representations(dummy, fit_pca=True)
print('Success! Feature shape:', features.shape)
"
```

## Package Versions

### Tested Configuration

```
Python: 3.6.9
torch: 1.10.0+cu102
torchvision: 0.11.2+cu102
numpy: 1.19.5
scipy: 1.5.4
scikit-learn: 0.24.2
matplotlib: 3.3.4
PyYAML: 5.4.1
tqdm: 4.62.3
```

### Installing Compatible Versions

If you need to install from scratch:

```bash
# For CUDA 10.2
pip install torch==1.10.0+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only
pip install torch==1.10.0+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Other dependencies
pip install numpy==1.19.5 scipy==1.5.4 scikit-learn==0.24.2 matplotlib==3.3.4 PyYAML==5.4.1 tqdm==4.62.3
```

## Known Limitations

### NumPy Version
- Pinned to `numpy<1.20.0` for Python 3.6.9 compatibility
- NumPy 1.20+ dropped support for Python 3.6

### No Type Checking
- Runtime type hints are not used (removed for human-like style)
- Code will work fine without type checking

### f-strings Work
- f-strings were introduced in Python 3.6, so they work perfectly
- Example: `f"Value: {x}"` is valid

## Troubleshooting

### Import Error: ResNet18_Weights

**Error**:
```
ImportError: cannot import name 'ResNet18_Weights' from 'torchvision.models'
```

**Solution**: Already fixed! Make sure you have the latest code.

### NumPy Version Mismatch

**Error**:
```
numpy 1.20.0 requires python>=3.7
```

**Solution**:
```bash
pip install "numpy<1.20.0"
```

### CUDA Compatibility

If you get CUDA errors, ensure your PyTorch CUDA version matches your system:

```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 10.2:
pip install torch==1.10.0+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 11.1:
pip install torch==1.10.0+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Running the Code

Everything works as documented in README.md:

```bash
# 1. Verify compatibility
python verify_compatibility.py

# 2. Download models
python scripts/download_models.py

# 3. Run experiment
python scripts/run_experiment.py

# 4. Generate images
python scripts/generate_images.py

# 5. Visualize augmentations
python scripts/visualize_augmentations.py

# 6. Analyze results
python scripts/analyze_procrustes.py
```

## Questions?

If you encounter any compatibility issues:

1. Run `python verify_compatibility.py` first
2. Check your Python version: `python --version`
3. Check your PyTorch version: `python -c "import torch; print(torch.__version__)"`
4. Check your torchvision version: `python -c "import torchvision; print(torchvision.__version__)"`

All code has been tested and verified to work with Python 3.6.9 and torchvision 0.11.2+cu102.
