"""
models.py

This file contains the backend logic for model instantiation.

TensorFlow models are defaulted to using CPU-only

Device logic (CPU vs GPU) for TF is present in this file, as TF's device(s)
can only be allocated once
"""

import torch
from scripts.model_interface import (
    EmbeddingBackend,
    HuggingFaceVisionBackend,
    TensorFlowBackend,
    TorchvisionBackend,
)
from torchvision import models


HUGGINGFACE_REVISIONS = {
    "google/medsiglip-448": "9cea28a1a1195f665105faa6e8544c112fd960a4",
    "microsoft/rad-dino": "110cbc18d5133582e320b43d53bf5c44e410c936",
    "google/vit-base-patch16-224-in21k": (
        "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
    ),
}


def _torchvision_preprocessing_spec(model_id, weights, transform):
    return {
        "schema_version": 1,
        "model_id": model_id,
        "framework": "torchvision",
        "weights": str(weights),
        "input": {
            "color_mode": "RGB",
            "resize_size": list(transform.resize_size),
            "crop_size": list(transform.crop_size),
            "interpolation": transform.interpolation.value,
            "antialias": bool(transform.antialias),
            "rescale": "uint8_to_float_[0,1]",
            "mean": list(transform.mean),
            "std": list(transform.std),
        },
        "embedding_output": {
            "method": "classifier_replaced_with_identity",
            "pooling": "architecture_global_average_pool",
        },
    }


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
        weights = models.ResNet50_Weights.DEFAULT
        transform = weights.transforms()
        model = models.resnet50(weights = weights)
        model.fc = torch.nn.Identity()
        return TorchvisionBackend(
            model_id,
            model,
            device,
            transform=transform,
            preprocessing_spec=_torchvision_preprocessing_spec(
                model_id, weights, transform
            ),
        )

    if model_id == "google/medsiglip-448":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceVisionBackend(
            model_id,
            device,
            revision=HUGGINGFACE_REVISIONS[model_id],
            output_key="pooler_output",
        )
    
    if model_id == "mobilenet_v2":
        import tensorflow as tf

        device = "cpu"

        tf.config.set_visible_devices([], "GPU")

        model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3),
                                                  include_top = False, pooling = "avg",
                                                  weights = "imagenet")
        return TensorFlowBackend(
            model_id,
            model,
            device,
            preprocess_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            preprocessing_spec={
                "schema_version": 1,
                "model_id": model_id,
                "framework": "tensorflow.keras",
                "weights": "imagenet",
                "input": {
                    "color_mode": "RGB",
                    "resize_size": [224, 224],
                    "interpolation": "bilinear",
                    "source_value_range": [0, 255],
                    "preprocess_function": (
                        "tf.keras.applications.mobilenet_v2.preprocess_input"
                    ),
                    "model_value_range": [-1, 1],
                },
                "embedding_output": {
                    "method": "model_forward",
                    "pooling": "global_average_pool",
                },
            },
        )
    
    if model_id == "microsoft/rad-dino":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceVisionBackend(
            model_id,
            device,
            revision=HUGGINGFACE_REVISIONS[model_id],
            output_key="pooler_output",
        )
    
    if model_id == "google/vit-base-patch16-224-in21k":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceVisionBackend(
            model_id,
            device,
            revision=HUGGINGFACE_REVISIONS[model_id],
            output_key="pooler_output",
        )

    # Add more models here
    raise ValueError(f"Unknown model_id: {model_id}")
