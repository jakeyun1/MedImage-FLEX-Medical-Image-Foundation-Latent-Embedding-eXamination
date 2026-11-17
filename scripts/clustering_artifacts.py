"""Validated sample-level KMeans assignments for reproducible analyses."""

import hashlib
import os
import tempfile

import numpy as np

from scripts.data_audit import ordered_ids_sha256


CLUSTERING_ARTIFACT_SCHEMA_VERSION = 1


def _scalar(artifact, key):
    if key not in artifact or np.asarray(artifact[key]).ndim != 0:
        raise ValueError(f"Clustering field {key!r} must be present and scalar.")
    return np.asarray(artifact[key]).item()


def validate_clustering_artifact(artifact):
    if int(_scalar(artifact, "schema_version")) != CLUSTERING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Clustering artifact schema version is incompatible.")
    if not str(_scalar(artifact, "dataset_name")).strip():
        raise ValueError("Clustering dataset name must be nonempty.")
    k = int(_scalar(artifact, "k"))
    sample_ids = np.asarray(artifact.get("sample_ids", [])).astype(str)
    classes = np.asarray(artifact.get("classes", [])).astype(str)
    y_true = np.asarray(artifact.get("y_true", []))
    seeds = np.asarray(artifact.get("seeds", []))
    assignments = np.asarray(artifact.get("assignments", []))
    if sample_ids.ndim != 1 or len(sample_ids) < 1:
        raise ValueError("Clustering sample IDs must be a nonempty vector.")
    if len(np.unique(sample_ids)) != len(sample_ids) or np.any(sample_ids == ""):
        raise ValueError("Clustering sample IDs must be unique and nonempty.")
    if classes.ndim != 1 or len(classes) != k or len(np.unique(classes)) != k:
        raise ValueError("Clustering classes must be unique and match k.")
    if y_true.shape != sample_ids.shape or not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("Clustering targets must be an aligned integer vector.")
    if np.any(y_true < 0) or np.any(y_true >= k):
        raise ValueError("Clustering targets are outside the class set.")
    if seeds.ndim != 1 or len(seeds) < 2 or not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("Clustering seeds must be an integer vector of length >= 2.")
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("Clustering seeds must be unique.")
    if assignments.shape != (len(seeds), len(sample_ids)):
        raise ValueError("Clustering assignments are not aligned to seeds and samples.")
    if not np.issubdtype(assignments.dtype, np.integer):
        raise ValueError("Clustering assignments must use an integer dtype.")
    if np.any(assignments < 0) or np.any(assignments >= k):
        raise ValueError("Clustering assignments contain an invalid cluster index.")
    if any(len(np.unique(row)) != k for row in assignments):
        raise ValueError("Every KMeans seed must populate exactly k clusters.")
    expected_hash = ordered_ids_sha256(sample_ids.tolist())
    if str(_scalar(artifact, "ordered_sample_ids_sha256")) != expected_hash:
        raise ValueError("Clustering sample-ID fingerprint is invalid.")
    return True


def write_clustering_artifact(artifact, output_dir, filename="assignments.npz",
                              relative_path=None):
    validate_clustering_artifact(artifact)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp.npz", dir=output_dir
    )
    os.close(fd)
    try:
        np.savez_compressed(temp_path, **artifact)
        with np.load(temp_path, allow_pickle=False) as loaded:
            validate_clustering_artifact({key: loaded[key] for key in loaded.files})
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    digest = hashlib.sha256()
    with open(filepath, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": CLUSTERING_ARTIFACT_SCHEMA_VERSION,
        "path": relative_path if relative_path is not None else filepath,
        "sha256": digest.hexdigest(),
        "size_bytes": int(os.path.getsize(filepath)),
        "n_samples": int(len(artifact["sample_ids"])),
        "n_seeds": int(len(artifact["seeds"])),
        "k": int(np.asarray(artifact["k"]).item()),
        "ordered_sample_ids_sha256": str(
            np.asarray(artifact["ordered_sample_ids_sha256"]).item()
        ),
    }
