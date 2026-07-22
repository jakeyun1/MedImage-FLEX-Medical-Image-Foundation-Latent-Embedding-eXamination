"""
custom_transforms.py

OPTIONAL

This file is used for writing custom transform functions.

Local and bound functions cannot be pickled. As a result, top-level functions are needed if 
using num_workers > 0 for the DataLoader. The functions in this file can be imported and returned 
by model_interface.EmbeddingBackend.get_transform().
"""

import io
from PIL import Image

# Example transform
def transform_function(img):
    img = img.resize((1024, 1024), resample = Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    img.save(buffer, format = "PNG", quality = 95)
    image_bytes = buffer.getvalue()

    return image_bytes