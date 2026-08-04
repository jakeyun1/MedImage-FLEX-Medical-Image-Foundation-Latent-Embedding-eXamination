"""
extraction.py

This file contains the logic for extracting embeddings from a dataset.
"""

import os
import numpy as np
from tqdm import tqdm

PROHIBITED_CHARS = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]

def extract_embeddings(dataloader, backend, normalize = True, cache = False):
    """
    Extracts the embeddings for a given model from a given dataset.

    Args:
        dataloader : DataLoader object used to load image batches
        backend : EmbeddingBackend object used to reference a model
        normalize : If True, embeddings are normalized
        cache : If True, embeddings are stored locally to prevent future recomputation
        return_metadata : If True, include embedding source metadata
    
    Returns:
        all_embs : A list of all embeddings - one per image
        all_paths : A list of all local, absolute image paths
        source : Whether embeddings were computed or loaded from cache

    """
    all_embs = []
    all_paths = []

    def clean_filename(filename, desired_char):
        """
        Helper function for standardizing embedding filenames.

        Args:
            filename : Initial file basename
            desired_char : Replacement for illegal characters

        Returns:
            filename : Cleaned, legal file basename
        """
        for char in PROHIBITED_CHARS:
            filename = filename.replace(char, desired_char)

        return filename

    # Prepare filename and clean it to prevent path issues
    normalization = "normalized" if normalize else "raw"
    protocol_name = getattr(dataloader, "protocol_name", dataloader.dataset_name)
    filename = backend.model_id + "+" + protocol_name + "+" + normalization
    filename = clean_filename(filename, "-").replace(".", "")
    filepath = f".{os.sep}embeddings{os.sep}{filename}.npz"

    if cache and os.path.exists(filepath):
        print(f"Embeddings file detected! Loading \'{os.path.abspath(filepath)}\'")
        with np.load(filepath) as cached:
            all_embs = cached["embeddings"]
            all_paths = cached["image_paths"].tolist()
        if len(all_embs) != len(all_paths):
            raise ValueError("Cached embedding and image-path counts do not match.")
        source = "cache"

        return all_embs, all_paths, source

    for batch in tqdm(dataloader, desc = "Extracting embeddings"):
        images, paths = batch
        embs = backend.encode_batch(images)

        if normalize:
            embs = embs / embs.norm(p = 2, dim = -1, keepdim = True)

        all_embs.append(embs.cpu().numpy())
        all_paths.extend(paths)

    all_embs = np.concatenate(all_embs, axis = 0)
    source = "computed"
    
    if cache:
        os.makedirs(f".{os.sep}embeddings", exist_ok = True)
        
        # Prevents overwriting embedding files
        if not os.path.exists(filepath):
            np.savez(filepath, embeddings = all_embs, image_paths = np.asarray(all_paths))
            print(f"Embeddings cached to: {os.path.abspath(filepath)}\n")

    return all_embs, all_paths, source
