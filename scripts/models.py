"""
models.py

This file contains the backend logic for model instantiation.

TensorFlow models are defaulted to using CPU-only

Device logic (CPU vs GPU) for TF is present in this file, as TF's device(s)
can only be allocated once
"""

import torch
import tensorflow as tf
from scripts.model_interface import *
from torchvision import models

def build_backend(model_id: str) -> EmbeddingBackend:
    """
    Builds the backend for a given model.
    
    Args:
        model_id : A string representing the model name, the link to the model, or a filepath to the model
        
        model_id could be:
        - "resnet50"
        - "resnet101"
        - "google/medsiglip-448"
        - etc.

    Returns:
        The appropriate EmbeddingBackend instance
    """
    # Sample models
    if model_id == "resnet50":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        return TorchvisionBackend(model_id, model, device)

    if model_id == "google/medsiglip-448":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceVisionBackend(model_id, device, output_key = "pooler_output")
    
    if model_id == "mobilenet_v2":
        device = "cpu"

        tf.config.set_visible_devices([], "GPU")

        model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3),
                                                  include_top = False, pooling = "avg")
        return TensorFlowBackend(model_id, model, device)
    
    if model_id == "microsoft/rad-dino":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceVisionBackend(model_id, device, output_key = "pooler_output")
    
    if model_id == "google/vit-base-patch16-224-in21k":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceVisionBackend(model_id, device, output_key = "pooler_output")

    # Add more models here
    raise ValueError(f"Unknown model_id: {model_id}")