"""
model_interface.py

This file contains the backend logic the for various model types.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
import tensorflow as tf
from torchvision import transforms
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoFeatureExtractor

class EmbeddingBackend(ABC):
    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device

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
    def __init__(self, model_id, model, device, target_size = [224, 224]):
        super().__init__(model_id, device)
        self.target_size = tuple(target_size)
        self.model = model.to(device).eval()

    def get_transform(self):
        """
        Returns the transform to be applied to each image. 
        
        Changed for each specific model.
        """
        # ResNet50/DenseNet121
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225]),
        ])

        return transform

    @torch.no_grad()
    def encode_batch(self, images):
        """
        Computes the embeddings for a batch of images.
        """
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        images = images.to(self.device)
        embs = self.model(images)           # assumes `final layer` == torch.nn.Identity()
        return embs

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
    def __init__(self, model_id, device, target_size = [448, 448], output_key = None):
        super().__init__(model_id, device)
        self.target_size = tuple(target_size)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code = True).to(device).eval()
        
        # Try three different ways to load an image processor 
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True, use_fast = True)
        except Exception:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code = True, use_fast = True)
            except Exception:
                self.processor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code = True, use_fast = True)

        # If the user does NOT specify output_key, we auto-detect it later
        self.output_key = output_key

    def get_transform(self):
        # Usually we bypass torch transforms and let the processor handle everything
        return None

    @torch.no_grad()
    def encode_batch(self, images):
        """
        Computes the embeddings for a batch of images.
        """
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
                if self.output_key is not None:
                    return embs[self.output_key]
                
                # Auto-detect and save the key for future batches
                for key in ["image_embeds", "pooler_output"]:
                    if key in embs and embs[key] is not None:
                        self.output_key = key
                        return embs[self.output_key]
                
                # Fallback: grab the first tensor we find
                for k, v in embs.items():
                    if isinstance(v, torch.Tensor):
                        self.output_key = k
                        return embs[self.output_key]

            return embs

        # Otherwise, assume it's a vision-only model where forward(pixel_values = pass) works
        outputs = self.model(pixel_values = pixel_values)

        # Auto-select a tensor output field once
        if self.output_key is None:
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    self.output_key = k
                    break

        embs = outputs[self.output_key]
        return embs

    @property
    def embedding_dim(self):
        """
        Returns the length of the embedding vector.
        """
        dummy = torch.zeros(1, 3, *self.target_size).to(self.device) # Image vector depends on model
        with torch.no_grad():
            out = self.model(pixel_values = dummy)

        if self.output_key is None:
            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    self.output_key = key
                    break

        return out[self.output_key].shape[-1]
    
class TensorFlowBackend(EmbeddingBackend):
    def __init__(self, model_id, model_path_or_obj, device, target_size = [224, 224], output_key = None):
        super().__init__(model_id, device)
        self.target_size = tuple(target_size)

        # If the user does NOT specify output_key, we auto-detect it later
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

        Uses either tf.image transformations or lets the model itself handle preprocessing
        This default transformation takes PIL -> NumPy -> tf.Tensor for direct use in a model
        No other preprocessing is applied to the image.

        Changed for each specific model.
        """
        def to_tf_tensor(img):
            img = img.resize(self.target_size)
            img = np.array(img)
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
            if self.output_key:
                tf_out = tf_out[self.output_key]
            else:
                for (k, v) in tf_out.items():
                    if isinstance(v, tf.Tensor) and len(v.shape) == 2:
                        self.output_key = k
                        break
                
                tf_out = tf_out[self.output_key]
    
        tf_out = tf_out.numpy() if hasattr(tf_out, 'numpy') else np.array(tf_out)
            
        # Return as a PyTorch tensor on the correct device (GPU/CPU)
        return torch.from_numpy(tf_out).float().to(self.device)

    @property
    def embedding_dim(self):
        """
        Returns the length of the embedding vector.
        """
        # Auto-detect dimension using a dummy pass
        dummy = tf.zeros((1, *self.target_size, 3)) # Image vector depends on model
        out = self.model(dummy, training = False)
        
        if isinstance(out, dict):
            if self.output_key:
                out = out[self.output_key]
            else:
                for (k, v) in out.items():
                    if isinstance(v, tf.Tensor) and len(v.shape) == 2:
                        self.output_key = k
                        break
                
                out = out[self.output_key]
                
        return out.shape[-1]