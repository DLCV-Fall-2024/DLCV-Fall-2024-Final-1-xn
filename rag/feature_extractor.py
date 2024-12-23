from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class FeatureExtractor:
    def __init__(self, depth_anything_model, device="cuda"):
        self.model = depth_anything_model
        self.device = device
        self.model.eval()

        # Get dimensions from the model
        self.hidden_dim = self.model.pretrained.embed_dim  # DINOv2 embedding dimension
        self.depth_feature_dim = 64  # From our 8x8 pooled depth features

        # Set up image preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(518),  # Default size used in Depth Anything
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_embedding_dim(self) -> int:
        """
        Returns the dimension of the final concatenated feature vector
        Global features + Spatial features + Depth features
        """
        return self.hidden_dim + self.hidden_dim + self.depth_feature_dim

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for the model"""
        if isinstance(image, Image.Image):
            return self.preprocess(image)
        return image

    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from an image using DINOv2 backbone and depth information

        Args:
            image: Preprocessed image tensor of shape [B, C, H, W]

        Returns:
            Dictionary containing:
            - global_features: Class token features
            - spatial_features: Averaged patch token features
            - depth_features: Processed depth map features
        """
        with torch.no_grad():
            # Get intermediate features from last 4 layers
            intermediate_features = self.model.pretrained.get_intermediate_layers(
                image, n=4, return_class_token=True  # Get last 4 layers like depth_head
            )

            # Get last layer features (index -1)
            last_layer_patches, last_layer_cls = intermediate_features[-1]

            # Get depth prediction
            depth_pred = self.model(image)  # [B, H, W]

            # Process spatial (patch) features
            # Average pooling over patches for a compact spatial representation
            spatial_features = last_layer_patches.mean(dim=1)  # [B, C]

            # Process depth features
            # Convert depth into a compact feature representation
            depth_features = self._process_depth_features(depth_pred)

            return {
                "global_features": last_layer_cls,  # [B, C]
                "spatial_features": spatial_features,  # [B, C]
                "depth_features": depth_features,  # [B, C_depth]
            }

    def _process_depth_features(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Convert depth map into a compact feature representation

        Args:
            depth_map: Depth prediction of shape [B, H, W]

        Returns:
            Processed depth features of shape [B, C_depth]
        """
        B = depth_map.shape[0]
        pooled_depth = F.adaptive_avg_pool2d(
            depth_map.unsqueeze(1), (8, 8)
        )  # [B, 1, 8, 8]
        depth_features = pooled_depth.reshape(B, -1)  # [B, 64]

        return depth_features

    def extract_and_concatenate(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract and concatenate all features into a single vector

        Args:
            image: Preprocessed image tensor

        Returns:
            Concatenated feature vector
        """
        features = self.extract_features(image)

        # Concatenate all features
        combined_features = torch.cat(
            [
                features["global_features"],
                features["spatial_features"],
                features["depth_features"],
            ],
            dim=1,
        )

        return combined_features
