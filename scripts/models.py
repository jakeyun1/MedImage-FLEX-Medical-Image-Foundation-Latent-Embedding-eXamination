"""
models.py

This file contains the backend logic for model instantiation.

TensorFlow models are defaulted to using CPU-only

Device logic (CPU vs GPU) for TF is present in this file, as TF's device(s)
can only be allocated once
"""

import json
import os

import numpy as np
import torch
from scripts.model_interface import (
    CallableEmbeddingBackend,
    EmbeddingBackend,
    HuggingFaceVisionBackend,
    TensorFlowBackend,
    TorchvisionBackend,
)
from torchvision import models


BIOMEDCLIP_REPO_ID = (
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
BIOMEDCLIP_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
MED_FLAMINGO_REPO_ID = "med-flamingo/med-flamingo"
MED_FLAMINGO_REVISION = "7243cd83bd426ceade9c4de9844cc5e5f3ff75e0"
CXR_FOUNDATION_REPO_ID = "google/cxr-foundation"

HUGGINGFACE_REVISIONS = {
    "google/medsiglip-448": "9cea28a1a1195f665105faa6e8544c112fd960a4",
    "microsoft/rad-dino": "110cbc18d5133582e320b43d53bf5c44e410c936",
    "google/vit-base-patch16-224-in21k": (
        "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
    ),
}


def _optional_import_error(model_name, dependency, exc):
    raise ImportError(
        f"{model_name} requires the optional dependency {dependency}. "
        "Install it in the model-specific HPC environment."
    ) from exc


def _require_revision(value, context):
    if not isinstance(value, str) or len(value) != 40 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(
            f"{context} must use an immutable 40-character commit revision."
        )
    return value


def _build_biomedclip_backend(model_id, device):
    try:
        from huggingface_hub import hf_hub_download
        from open_clip import create_model_and_transforms
        from open_clip.factory import _MODEL_CONFIGS
    except ImportError as exc:
        _optional_import_error("BiomedCLIP", "open_clip_torch", exc)

    revision = _require_revision(BIOMEDCLIP_REVISION, "BiomedCLIP")
    config_path = hf_hub_download(
        BIOMEDCLIP_REPO_ID,
        "open_clip_config.json",
        revision=revision,
    )
    checkpoint_path = hf_hub_download(
        BIOMEDCLIP_REPO_ID,
        "open_clip_pytorch_model.bin",
        revision=revision,
    )
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_config = config.get("model_cfg")
    preprocess_config = config.get("preprocess_cfg")
    if not isinstance(model_config, dict) or not isinstance(
        preprocess_config, dict
    ):
        raise ValueError("BiomedCLIP is missing its OpenCLIP model configuration.")

    local_model_name = "medimage_flex_biomedclip"
    _MODEL_CONFIGS[local_model_name] = model_config
    model, _, validation_transform = create_model_and_transforms(
        model_name=local_model_name,
        pretrained=checkpoint_path,
        **{
            f"image_{key}": value
            for key, value in preprocess_config.items()
        },
    )
    model = model.to(device).eval()
    embedding_dim = int(model_config["embed_dim"])

    def encode(images):
        if isinstance(images, torch.Tensor):
            batch = images
        else:
            image_list = list(images)
            if not image_list:
                raise ValueError("BiomedCLIP received an empty image batch.")
            if isinstance(image_list[0], torch.Tensor):
                batch = torch.stack(image_list)
            else:
                batch = torch.stack([
                    validation_transform(image.convert("RGB"))
                    for image in image_list
                ])
        return model.encode_image(batch.to(device), normalize=False)

    return CallableEmbeddingBackend(
        model_id=model_id,
        device=device,
        transform=validation_transform,
        encode_fn=encode,
        embedding_dim=embedding_dim,
        preprocessing_spec={
            "schema_version": 1,
            "model_id": model_id,
            "framework": "open_clip",
            "weights": {
                "source": BIOMEDCLIP_REPO_ID,
                "revision": revision,
                "checkpoint": "open_clip_pytorch_model.bin",
            },
            "input": {
                "color_mode": "RGB",
                "processor": "native OpenCLIP validation transform",
                "processor_settings": preprocess_config,
            },
            "embedding_output": {
                "method": "encode_image",
                "pooling": "native projected image embedding",
            },
        },
    )


def _build_cxr_foundation_backend(model_id):
    revision = _require_revision(
        os.environ.get("MEDIMAGE_FLEX_CXR_FOUNDATION_REVISION"),
        "CXR Foundation",
    )
    try:
        import tensorflow as tf
        from clientside.clients import HuggingFaceCxrClient
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        _optional_import_error(
            "CXR Foundation",
            "the Google-Health/cxr-foundation Python package",
            exc,
        )

    snapshot_path = snapshot_download(
        repo_id=CXR_FOUNDATION_REPO_ID,
        revision=revision,
        cache_dir=os.environ.get("MEDIMAGE_FLEX_HF_CACHE_DIR"),
        allow_patterns=[
            "elixr-c-v2-pooled/*",
            "pax-elixr-b-text/*",
        ],
    )
    elixr_model = tf.saved_model.load(
        os.path.join(snapshot_path, "elixr-c-v2-pooled")
    )
    qformer_model = tf.saved_model.load(
        os.path.join(snapshot_path, "pax-elixr-b-text")
    )
    client = HuggingFaceCxrClient(
        bert_model=None,
        elixr_c_model=elixr_model,
        qformer_model=qformer_model,
    )

    def encode(images):
        if isinstance(images, torch.Tensor):
            raise TypeError(
                "CXR Foundation requires untransformed PIL images."
            )
        grayscale_images = [image.convert("L") for image in images]
        if not grayscale_images:
            raise ValueError("CXR Foundation received an empty image batch.")
        outputs = client.get_image_embeddings_from_images(grayscale_images)
        rows = []
        for output in outputs:
            if output.error:
                raise RuntimeError(f"CXR Foundation inference failed: {output.error}")
            tokens = np.asarray(output.general_img_emb, dtype=np.float32)
            if tokens.ndim != 2 or tokens.shape[-1] != 768:
                raise ValueError(
                    "CXR Foundation general_img_emb must have shape "
                    f"[tokens, 768]; found {tokens.shape}."
                )
            rows.append(tokens.mean(axis=0))
        return torch.from_numpy(np.stack(rows))

    return CallableEmbeddingBackend(
        model_id=model_id,
        device="tensorflow",
        transform=None,
        encode_fn=encode,
        embedding_dim=768,
        preprocessing_spec={
            "schema_version": 1,
            "model_id": model_id,
            "framework": "tensorflow_saved_model",
            "weights": {
                "source": CXR_FOUNDATION_REPO_ID,
                "revision": revision,
                "image_model": "elixr-c-v2-pooled",
                "projection_model": "pax-elixr-b-text",
            },
            "input": {
                "color_mode": "grayscale",
                "processor": "Google CXR Foundation client",
            },
            "embedding_output": {
                "field": "general_img_emb",
                "pooling": "mean over image tokens",
            },
        },
    )


def _build_med_flamingo_backend(model_id, device):
    llama_path = os.environ.get("MEDIMAGE_FLEX_MEDFLAMINGO_LLAMA_PATH")
    if not llama_path:
        raise ValueError(
            "Med-Flamingo requires MEDIMAGE_FLEX_MEDFLAMINGO_LLAMA_PATH "
            "to point to the local Llama-7B v1 model used by the official release."
        )
    if not os.path.isdir(llama_path):
        raise ValueError(f"Med-Flamingo Llama path does not exist: {llama_path}")

    try:
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms
    except ImportError as exc:
        _optional_import_error("Med-Flamingo", "open-flamingo==0.0.2", exc)

    revision = _require_revision(MED_FLAMINGO_REVISION, "Med-Flamingo")
    model, image_processor, _ = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4,
    )
    checkpoint_path = hf_hub_download(
        MED_FLAMINGO_REPO_ID,
        "model.pt",
        revision=revision,
        cache_dir=os.environ.get("MEDIMAGE_FLEX_HF_CACHE_DIR"),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_perceiver = [
        key for key in incompatible.missing_keys
        if key.startswith("perceiver.")
    ]
    if missing_perceiver:
        raise RuntimeError(
            "Med-Flamingo checkpoint did not populate the Perceiver weights."
        )

    vision_encoder = model.vision_encoder
    if hasattr(vision_encoder, "visual"):
        vision_encoder = vision_encoder.visual
    vision_encoder = vision_encoder.to(device).eval()
    perceiver = model.perceiver.to(device).eval()
    del model

    def encode(images):
        if isinstance(images, torch.Tensor):
            batch = images
        else:
            image_list = list(images)
            if not image_list:
                raise ValueError("Med-Flamingo received an empty image batch.")
            if isinstance(image_list[0], torch.Tensor):
                batch = torch.stack(image_list)
            else:
                batch = torch.stack([
                    image_processor(image.convert("RGB"))
                    for image in image_list
                ])

        vision_output = vision_encoder(batch.to(device))
        if not isinstance(vision_output, (tuple, list)) or len(vision_output) < 2:
            raise ValueError(
                "Med-Flamingo vision encoder did not return patch tokens."
            )
        vision_tokens = vision_output[1]
        if vision_tokens.ndim != 3:
            raise ValueError(
                "Med-Flamingo vision tokens must have shape [batch, tokens, "
                f"dimension]; found {tuple(vision_tokens.shape)}."
            )
        visual_latents = perceiver(vision_tokens[:, None, None, :, :])
        if visual_latents.ndim != 4 or visual_latents.shape[1] != 1:
            raise ValueError(
                "Med-Flamingo Perceiver output must have shape "
                f"[batch, 1, latents, dimension]; found {tuple(visual_latents.shape)}."
            )
        return visual_latents[:, 0].mean(dim=1)

    return CallableEmbeddingBackend(
        model_id=model_id,
        device=device,
        transform=image_processor,
        encode_fn=encode,
        embedding_dim=1024,
        preprocessing_spec={
            "schema_version": 1,
            "model_id": model_id,
            "framework": "open_flamingo",
            "weights": {
                "source": MED_FLAMINGO_REPO_ID,
                "revision": revision,
                "checkpoint": "model.pt",
                "vision_encoder": "OpenAI CLIP ViT-L/14",
            },
            "input": {
                "color_mode": "RGB",
                "processor": "native OpenFlamingo CLIP image processor",
            },
            "embedding_output": {
                "method": "vision encoder followed by trained Perceiver",
                "pooling": "mean over 64 Perceiver latent tokens",
            },
        },
    )


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
    if model_id == "biomedclip":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _build_biomedclip_backend(model_id, device)

    if model_id == "cxr-foundation":
        return _build_cxr_foundation_backend(model_id)

    if model_id in {"medflamingo", MED_FLAMINGO_REPO_ID}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _build_med_flamingo_backend(model_id, device)

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
