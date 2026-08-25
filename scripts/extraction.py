"""
extraction.py

This file contains the logic for extracting embeddings from a dataset.
"""

import os

import numpy as np
from tqdm import tqdm

from scripts.data_audit import ordered_ids_sha256
from scripts.dataset_contracts import get_dataset_contract


CACHE_SCHEMA_VERSION = 1
PROHIBITED_CHARS = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]


def _clean_filename(filename, desired_char):
    for char in PROHIBITED_CHARS:
        filename = filename.replace(char, desired_char)
    return filename


def _cache_paths(dataloader, backend, normalize):
    filename = f"{backend.model_id}+{dataloader.dataset_name}"
    filename = _clean_filename(filename, "-").replace(".", "")
    normalization = "normalized" if normalize else "unnormalized"
    base_path = f".{os.sep}embeddings{os.sep}{filename}-{normalization}"
    return f"{base_path}.npz", f".{os.sep}embeddings{os.sep}{filename}.npy"


def _sample_ids(dataset_name, image_paths):
    contract = get_dataset_contract(dataset_name)
    return [contract.sample_id_from_path(path) for path in image_paths]


def _require_unique_sample_ids(sample_ids, context):
    if len(set(sample_ids)) != len(sample_ids):
        seen = set()
        duplicates = []
        for sample_id in sample_ids:
            if sample_id in seen and sample_id not in duplicates:
                duplicates.append(sample_id)
            seen.add(sample_id)
        raise ValueError(f"{context} contains duplicate sample IDs: {duplicates[:5]}")


def _validate_embedding_array(embeddings, sample_ids, context):
    if embeddings.ndim != 2:
        raise ValueError(f"{context} embeddings must be a two-dimensional array.")
    if embeddings.shape[0] != len(sample_ids):
        raise ValueError(
            f"{context} embedding rows ({embeddings.shape[0]}) do not match "
            f"sample IDs ({len(sample_ids)})."
        )
    _require_unique_sample_ids(sample_ids, context)


def _write_embedding_cache(
    filepath,
    embeddings,
    sample_ids,
    dataset_name,
    model_id,
    normalize,
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temporary_path = f"{filepath}.tmp-{os.getpid()}.npz"
    try:
        np.savez(
            temporary_path,
            schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
            dataset_name=np.asarray(str(dataset_name)),
            model_id=np.asarray(str(model_id)),
            normalized=np.asarray(bool(normalize)),
            sample_ids=np.asarray(sample_ids, dtype=str),
            ordered_ids_sha256=np.asarray(ordered_ids_sha256(sample_ids)),
            embeddings=embeddings,
        )
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_embedding_cache(filepath, dataloader, backend, normalize):
    required_fields = {
        "schema_version",
        "dataset_name",
        "model_id",
        "normalized",
        "sample_ids",
        "ordered_ids_sha256",
        "embeddings",
    }
    try:
        with np.load(filepath, allow_pickle=False) as cache_file:
            missing_fields = sorted(required_fields - set(cache_file.files))
            if missing_fields:
                raise ValueError(f"missing fields: {missing_fields}")

            schema_version = int(cache_file["schema_version"].item())
            dataset_name = str(cache_file["dataset_name"].item())
            model_id = str(cache_file["model_id"].item())
            cached_normalize = bool(cache_file["normalized"].item())
            cached_sample_ids = cache_file["sample_ids"].astype(str).tolist()
            cached_hash = str(cache_file["ordered_ids_sha256"].item())
            embeddings = np.asarray(cache_file["embeddings"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid embedding cache at {filepath}: {exc}") from exc

    if schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Embedding cache schema mismatch at {filepath}: "
            f"expected {CACHE_SCHEMA_VERSION}, found {schema_version}."
        )
    if dataset_name != dataloader.dataset_name:
        raise ValueError(
            f"Embedding cache dataset mismatch at {filepath}: "
            f"expected {dataloader.dataset_name}, found {dataset_name}."
        )
    if model_id != str(backend.model_id):
        raise ValueError(
            f"Embedding cache model mismatch at {filepath}: "
            f"expected {backend.model_id}, found {model_id}."
        )
    if cached_normalize != bool(normalize):
        raise ValueError(f"Embedding cache normalization mismatch at {filepath}.")

    _validate_embedding_array(embeddings, cached_sample_ids, "Embedding cache")
    if cached_hash != ordered_ids_sha256(cached_sample_ids):
        raise ValueError(f"Embedding cache sample-ID hash mismatch at {filepath}.")

    current_paths = list(dataloader.dataset.image_paths)
    current_sample_ids = _sample_ids(dataloader.dataset_name, current_paths)
    _require_unique_sample_ids(current_sample_ids, "Current dataset")
    cached_set = set(cached_sample_ids)
    current_set = set(current_sample_ids)
    if cached_set != current_set:
        missing = sorted(cached_set - current_set)
        added = sorted(current_set - cached_set)
        raise ValueError(
            f"Embedding cache sample IDs do not match the current dataset at {filepath}; "
            f"missing from current dataset: {missing[:5]}; "
            f"new in current dataset: {added[:5]}."
        )

    path_by_sample_id = dict(zip(current_sample_ids, current_paths))
    aligned_paths = [path_by_sample_id[sample_id] for sample_id in cached_sample_ids]
    return embeddings, aligned_paths


def extract_embeddings(dataloader, backend, normalize=True, cache=False):
    """Extract embeddings while preserving their exact image-row correspondence."""
    filepath, legacy_filepath = _cache_paths(dataloader, backend, normalize)

    if cache and os.path.exists(filepath):
        print(f"Embeddings cache detected! Loading '{os.path.abspath(filepath)}'")
        embeddings, image_paths = _load_embedding_cache(
            filepath, dataloader, backend, normalize
        )
        return embeddings, image_paths, "cache"

    if cache and os.path.exists(legacy_filepath):
        print(
            f"Ignoring legacy matrix-only cache '{os.path.abspath(legacy_filepath)}'; "
            "it cannot prove image-row alignment."
        )

    embedding_batches = []
    image_paths = []
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images, paths = batch
        embeddings = backend.encode_batch(images)

        if normalize:
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        batch_embeddings = embeddings.cpu().numpy()
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(paths):
            raise ValueError(
                "Embedding batch rows do not match the batch image paths."
            )
        embedding_batches.append(batch_embeddings)
        image_paths.extend(paths)

    if not embedding_batches:
        raise ValueError("Embedding extraction produced no batches.")
    embeddings = np.concatenate(embedding_batches, axis=0)
    sample_ids = _sample_ids(dataloader.dataset_name, image_paths)
    _validate_embedding_array(embeddings, sample_ids, "Extracted embeddings")

    current_sample_ids = _sample_ids(
        dataloader.dataset_name, dataloader.dataset.image_paths
    )
    _require_unique_sample_ids(current_sample_ids, "Current dataset")
    if set(sample_ids) != set(current_sample_ids):
        raise ValueError(
            "Extracted sample IDs do not match the dataset sample IDs."
        )

    if cache:
        _write_embedding_cache(
            filepath,
            embeddings,
            sample_ids,
            dataloader.dataset_name,
            backend.model_id,
            normalize,
        )
        print(f"Embeddings cached to: {os.path.abspath(filepath)}\n")

    return embeddings, image_paths, "computed"
