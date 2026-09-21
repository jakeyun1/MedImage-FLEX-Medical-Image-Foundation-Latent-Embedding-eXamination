"""
model_interface.py

This file contains the backend logic the for various model types.
"""

from abc import ABC, abstractmethod
from enum import Enum
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel, AutoProcessor

try:
    import tensorflow as tf
except ImportError:  # TensorFlow is only required for TensorFlow-backed models.
    tf = None


def _json_safe(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _require_matrix(output, context):
    if not hasattr(output, "shape") or len(output.shape) != 2:
        shape = getattr(output, "shape", None)
        raise ValueError(f"{context} must return a [batch, dimension] matrix; found {shape}.")
    return output

class EmbeddingBackend(ABC):
    def __init__(self, model_id, device, preprocessing_spec):
        self.model_id = model_id
        self.device = device
        if not isinstance(preprocessing_spec, dict) or not preprocessing_spec:
            raise ValueError("Every embedding backend must declare a preprocessing contract.")
        self.preprocessing_spec = _json_safe(preprocessing_spec)

    def transform_function(self, image):
        """
        NOTE: OPTIONAL, must be implemented by a user

        Used to implement custom image transformations.

        Args:
            image : A raw medical image (e.g. PIL, NumPy, Tensor, etc.)

        Returns:
            A transformed image
        """
        return None

    @abstractmethod
    def get_transform(self):
        """
        Return a torchvision transform to apply when loading datasets.
        """
        pass

    @abstractmethod
    def encode_batch(self, images):
        """
        Computes the embeddings for a batch of images.

        Args:
            images : tensor or list of PIL images from your dataloader
            
        Returns:
            torch.Tensor [B, D] of embeddings on CPU (or GPU, one's choice)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self):
        """
        Returns the length of the embedding vector.
        """
        pass

class TorchvisionBackend(EmbeddingBackend):
    def __init__(self, model_id, model, device, transform, preprocessing_spec,
                 target_size = [224, 224]):
        super().__init__(model_id, device, preprocessing_spec)
        self.target_size = tuple(target_size)
        self.model = model.to(device).eval()
        self.transform = transform

    def get_transform(self):
        """
        Returns the transform to be applied to each image. 
        
        Changed for each specific model.
        """
        return self.transform

    @torch.no_grad()
    def encode_batch(self, images):
        """
        Computes the embeddings for a batch of images.
        """
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        images = images.to(self.device)
        embs = self.model(images)
        return _require_matrix(embs, f"{self.model_id} embedding output")

    @property
    def embedding_dim(self):
        """
        Returns the length of the embedding vector.
        """
        # You can infer this once, cache it
        dummy = torch.zeros(1, 3, 224, 224).to(self.device) # Image vector depends on model
        with torch.no_grad():
            out = self.model(dummy) # [B, D]
        return out.shape[-1]

class HuggingFaceVisionBackend(EmbeddingBackend):
    def __init__(self, model_id, device, revision, target_size = [448, 448],
                 output_key = None):
        # --- General models ---
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                f"{model_id} must use an immutable 40-character Hugging Face "
                "commit revision."
            )
        self.target_size = tuple(target_size)
        self.revision = revision
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        ).to(device).eval()
        
        # Prefer the image-only processor so text/tokenizer state is not part of
        # an image-embedding pipeline. Retain broader loaders as compatibility fallbacks.
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    revision=revision,
                    trust_remote_code=True,
                    use_fast=True,
                )
            except Exception:
                self.processor = AutoFeatureExtractor.from_pretrained(
                    model_id,
                    revision=revision,
                    trust_remote_code=True,
                    use_fast=True,
                )

        self.output_key = output_key
        if hasattr(self.model, "get_image_features"):
            embedding_output = {
                "method": "get_image_features",
                "fallback_key": output_key,
            }
        elif output_key is not None:
            embedding_output = {
                "method": "model_forward",
                "key": output_key,
            }
        else:
            raise ValueError(
                f"{model_id} must declare an embedding output key."
            )

        processor_fields = (
            "do_convert_rgb", "do_resize", "size", "resample",
            "do_center_crop", "crop_size", "do_rescale", "rescale_factor",
            "do_normalize", "image_mean", "image_std",
        )
        processor_settings = {
            field: _json_safe(getattr(self.processor, field))
            for field in processor_fields
            if hasattr(self.processor, field)
        }
        super().__init__(model_id, device, {
            "schema_version": 1,
            "model_id": model_id,
            "framework": "transformers",
            "weights": {
                "source": model_id,
                "revision": revision,
            },
            "input": {
                "color_mode": "RGB",
                "processor_class": type(self.processor).__name__,
                "processor_settings": processor_settings,
            },
            "embedding_output": embedding_output,
        })

    def get_transform(self):
        # Usually we bypass torch transforms and let the processor handle everything
        return None
        
    @torch.no_grad()
    def encode_batch(self, images):
        """
        Computes the embeddings for a batch of images.
        """
        # --- General models ---
        if isinstance(images, tuple):
            images = list(images)

        # Case 1: batch tensor [B, C, H, W]
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)

        # Case 2: list of PIL / numpy images
        else:
            proc = self.processor(images = images, return_tensors = "pt")
            # Ignore all non-vision fields (input_ids, etc.)
            pixel_values = proc["pixel_values"].to(self.device)

        # If the model exposes image-only API: `get_image_features`
        if hasattr(self.model, "get_image_features"):
            embs = self.model.get_image_features(pixel_values = pixel_values)
            
            # Catch newer HF versions returning an Output object instead of a tensor
            if not isinstance(embs, torch.Tensor):
                # Use user-provided key if it exists
                if self.output_key is None:
                    raise ValueError(
                        f"{self.model_id} get_image_features returned a structured "
                        "output without a declared key."
                    )
                embs = embs[self.output_key]

            return _require_matrix(embs, f"{self.model_id} embedding output")

        # Otherwise, assume it's a vision-only model where forward(pixel_values = pass) works
        outputs = self.model(pixel_values = pixel_values)

        embs = outputs[self.output_key]
        return _require_matrix(embs, f"{self.model_id} embedding output")

    @property
    def embedding_dim(self):
        """
        Returns the length of the embedding vector.
        """
        dummy = torch.zeros(1, 3, *self.target_size).to(self.device) # Image vector depends on model
        return self.encode_batch(dummy).shape[-1]
    
class TensorFlowBackend(EmbeddingBackend):
    def __init__(self, model_id, model_path_or_obj, device, preprocessing_spec,
                 preprocess_function=None, target_size = [224, 224], output_key = None):
        if tf is None:
            raise ImportError("TensorFlow is required for TensorFlowBackend.")
        super().__init__(model_id, device, preprocessing_spec)
        self.target_size = tuple(target_size)
        self.preprocess_function = preprocess_function
        self.resize_resample = Image.Resampling.BILINEAR

        self.output_key = output_key
        
        # If a local file path, load the model
        if isinstance(model_path_or_obj, str):
            try:
                self.model = tf.keras.models.load_model(model_path_or_obj, compile = False)
            except Exception:
                # Try to use the SavedModel load function
                self.model = tf.saved_model.load(model_path_or_obj)
        # Else assume the model was preloaded
        else:
            self.model = model_path_or_obj

    def get_transform(self):
        """
        Returns the transform to be applied to each image.

        Resize a PIL image and apply the preprocessing function declared by the model.
        """
        def to_tf_tensor(img):
            img = img.resize(self.target_size, resample=self.resize_resample)
            # Use a writable array because Keras preprocessing functions may operate in place.
            img = np.array(img, dtype=np.float32, copy=True)
            if self.preprocess_function is not None:
                img = self.preprocess_function(img)
            img = tf.convert_to_tensor(img, dtype = tf.float32)

            return img

        return to_tf_tensor

    def encode_batch(self, images):
        """
        Computes the embeddings for a batch of images.
        """
        # tf.stack turns a list of (H, W, 3) tf.tensors into one (B, H, W, 3) tensor.
        if isinstance(images, (tuple, list)):
            img_batch = tf.stack(images)
        else:
            img_batch = images

        # Ensure we are passing floats, not ints (common PIL issue)
        if img_batch.dtype != tf.float32:
            img_batch = tf.cast(img_batch, tf.float32)

        # Pass images to model
        tf_out = self.model(img_batch, training = False)

        # Handle dictionary outputs safely
        if isinstance(tf_out, dict):
            if not self.output_key:
                raise ValueError(
                    f"{self.model_id} returned a dictionary without a declared output key."
                )
            tf_out = tf_out[self.output_key]
    
        tf_out = tf_out.numpy() if hasattr(tf_out, 'numpy') else np.array(tf_out)
            
        # Return as a PyTorch tensor on the correct device (GPU/CPU)
        output = torch.from_numpy(tf_out).float().to(self.device)
        return _require_matrix(output, f"{self.model_id} embedding output")

    @property
    def embedding_dim(self):
        """
        Returns the length of the embedding vector.
        """
        # Auto-detect dimension using a dummy pass
        dummy = tf.zeros((1, *self.target_size, 3)) # Image vector depends on model
        out = self.model(dummy, training = False)
        
        if isinstance(out, dict):
            if not self.output_key:
                raise ValueError(
                    f"{self.model_id} returned a dictionary without a declared output key."
                )
            out = out[self.output_key]
                
        return out.shape[-1]
