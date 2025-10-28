"""
Download Pretrained Models

Downloads and caches pretrained ResNet-18 model for use as target f*.
"""

import torch
import torchvision.models as models
import os


def download_resnet18(save_dir = "models/pretrained"):
    """
    Download pretrained ResNet-18 with ImageNet weights.
    
    Args:
        save_dir: Directory to save model
    """
    print("Downloading pretrained ResNet-18...")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Download model with weights (old API for torchvision 0.11.2)
    model = models.resnet18(pretrained=True)
    
    # Save model
    save_path = os.path.join(save_dir, "resnet18_imagenet.pth")
    torch.save(model.state_dict(), save_path)
    
    print(f"Model saved to {save_path}")
    print(f"Model size: {os.path.getsize(save_path) / 1e6:.2f} MB")

    return model


def verify_model(model_path = "models/pretrained/resnet18_imagenet.pth"):
    """
    Verify downloaded model can be loaded.
    
    Args:
        model_path: Path to model file
    """
    print(f"Verifying model at {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    try:
        # Load model (old API for torchvision 0.11.2)
        model = models.resnet18(pretrained=False)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        
        print("Model loaded successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print("Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/pretrained",
        help="Directory to save models",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model after download",
    )
    
    args = parser.parse_args()
    
    # Download
    model = download_resnet18(save_dir=args.save_dir)
    
    # Verify if requested
    if args.verify:
        model_path = os.path.join(args.save_dir, "resnet18_imagenet.pth")
        verify_model(model_path)
    
    print("\nDone!")
