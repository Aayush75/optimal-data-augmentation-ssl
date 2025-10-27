"""
Target Model (f*) for Optimal Augmentation Generation

Loads pretrained ResNet-18 and extracts target representations
that we want to achieve through optimal augmentations.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA


class TargetModel:
    """
    Target model f* that provides desired representations.
    
    Based on Section 4 of the paper, we use a pretrained model
    (ResNet-18 on ImageNet) to obtain target representations.
    
    Args:
        architecture: Model architecture (default: "resnet18")
        pretrained: Whether to use pretrained weights (default: True)
        weights: Specific weights to load (default: "IMAGENET1K_V1")
        feature_layer: Layer to extract features from (default: "avgpool")
        pca_dim: Dimension to reduce features to via PCA (default: 64)
        device: Device to run model on
    """
    
    def __init__(
        self,
        architecture = "resnet18",
        pretrained = True,
        weights = "IMAGENET1K_V1",
        feature_layer = "avgpool",
        pca_dim = 64,
        device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.architecture = architecture
        self.pretrained = pretrained
        self.weights = weights
        self.feature_layer = feature_layer
        self.pca_dim = pca_dim
        self.device = device
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # For feature extraction
        self.features = None
        self._register_hook()
        
        # PCA for dimensionality reduction
        self.pca = None
        self.pca_fitted = False
        
        # CIFAR-100 preprocessing
        self.preprocess = self._get_preprocessing()
        
    def _load_model(self) -> nn.Module:
        """Load pretrained model."""
        if self.architecture.lower() == "resnet18":
            if self.pretrained:
                # Use new weights API
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1
                model = models.resnet18(weights=weights)
            else:
                model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def _register_hook(self):
        """Register forward hook to extract features."""
        def hook(module, input, output):
            # Flatten the output
            self.features = output.detach().cpu()
            if len(self.features.shape) > 2:
                # If output is not flattened (e.g., from avgpool)
                self.features = self.features.view(self.features.size(0), -1)
        
        # Register hook on the specified layer
        if self.feature_layer == "avgpool":
            self.model.avgpool.register_forward_hook(hook)
        else:
            raise ValueError(f"Unsupported feature layer: {self.feature_layer}")
    
    def _get_preprocessing(self) -> transforms.Compose:
        """
        Get preprocessing transforms for CIFAR-100.
        
        ResNet-18 expects 224x224 images, but CIFAR-100 is 32x32.
        We resize to 224x224 to match ImageNet pretraining.
        """
        # ImageNet normalization statistics
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        return transforms.Compose([
            transforms.Resize(224),  # Resize CIFAR-100 from 32x32 to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    
    @torch.no_grad()
    def extract_features(
        self, 
        images,
        batch_size = 64,
    ) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: (n, c, h, w) tensor or array of images
            batch_size: Batch size for processing
            
        Returns:
            features: (n, d) array of features
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        all_features = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)
            
            # Forward pass (hook will capture features)
            _ = self.model(batch)
            
            # Get captured features
            all_features.append(self.features.numpy())
        
        # Concatenate all batches
        features = np.concatenate(all_features, axis=0)
        return features
    
    def fit_pca(self, features):
        """
        Fit PCA on features for dimensionality reduction.
        
        As mentioned in Appendix C of the paper:
        "we reduced the dimension of the target representations to 64 using PCA"
        
        Args:
            features: (n, d) array of features
        """
        print(f"Fitting PCA to reduce dimension from {features.shape[1]} to {self.pca_dim}")
        
        self.pca = PCA(n_components=self.pca_dim, random_state=42)
        self.pca.fit(features)
        self.pca_fitted = True
        
        # Report explained variance
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA explained variance: {explained_var:.4f}")
    
    def transform_pca(self, features) -> np.ndarray:
        """
        Apply PCA transformation to features.
        
        Args:
            features: (n, d) array of features
            
        Returns:
            transformed: (n, pca_dim) array
        """
        if not self.pca_fitted:
            raise RuntimeError("PCA not fitted. Call fit_pca first.")
        
        return self.pca.transform(features)
    
    def get_target_representations(
        self,
        images,
        fit_pca = False,
        batch_size = 64,
    ) -> np.ndarray:
        """
        Get target representations f*(x) for images.
        
        This is the main method to obtain target representations
        that we want to achieve through optimal augmentations.
        
        Args:
            images: (n, c, h, w) images
            fit_pca: Whether to fit PCA on these features
            batch_size: Batch size for processing
            
        Returns:
            representations: (n, pca_dim) array of target representations
        """
        # Extract features
        features = self.extract_features(images, batch_size=batch_size)
        
        # Fit PCA if requested
        if fit_pca:
            self.fit_pca(features)
        
        # Apply PCA transformation
        if self.pca_fitted:
            representations = self.transform_pca(features)
        else:
            representations = features
        
        return representations
    
    def __repr__(self):
        return (f"TargetModel(architecture={self.architecture}, "
                f"pretrained={self.pretrained}, "
                f"pca_dim={self.pca_dim}, "
                f"device={self.device})")


def check_representation_conditions(F, tol = 1e-10) -> dict:
    """
    Check if target representations satisfy Condition 3.2 (full rank covariance).
    
    Args:
        F: (d, n) matrix where F[:, i] = f*(x_i)
        tol: Tolerance for rank determination
        
    Returns:
        Dictionary with diagnostic information
    """
    d, n = F.shape
    
    # Compute covariance: cov(F) = 1/n * F H H^T F^T
    # where H = I - 1/n * 1 1^T is the centering matrix
    F_centered = F - F.mean(axis=1, keepdims=True)
    cov_F = (F_centered @ F_centered.T) / n
    
    # Check rank
    rank = np.linalg.matrix_rank(cov_F, tol=tol)
    is_full_rank = (rank == d)
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(cov_F)
    min_eigenvalue = np.min(eigenvalues)
    
    return {
        "full_rank": is_full_rank,
        "rank": rank,
        "dimension": d,
        "num_samples": n,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": np.max(eigenvalues),
        "condition_number": np.max(eigenvalues) / max(min_eigenvalue, tol),
    }
