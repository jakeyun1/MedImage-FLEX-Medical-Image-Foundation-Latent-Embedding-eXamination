"""Validated per-query evidence artifacts for retrieval benchmarks."""

import hashlib
import os
import tempfile

import numpy as np

from scripts.data_audit import ordered_ids_sha256


RETRIEVAL_ARTIFACT_SCHEMA_VERSION = 1


def _scalar(artifact, key):
    if key not in artifact:
        raise ValueError(f"Retrieval artifact is missing required field '{key}'.")
    value = np.asarray(artifact[key])
    if value.ndim != 0:
        raise ValueError(f"Retrieval field '{key}' must be scalar.")
    return value.item()


def validate_retrieval_artifact(artifact):
    """Reject incomplete, misaligned, or internally inconsistent artifacts."""
    if int(_scalar(artifact, "schema_version")) != RETRIEVAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Retrieval artifact schema version is incompatible.")
    if str(_scalar(artifact, "label_type")) not in {"multiclass", "multilabel"}:
        raise ValueError("Retrieval label type is invalid.")
    for key in ("dataset_name", "similarity", "candidate_exclusion"):
        if not str(_scalar(artifact, key)).strip():
            raise ValueError(f"Retrieval field '{key}' must be non-empty.")
    if not isinstance(_scalar(artifact, "normalized"), (bool, np.bool_)):
        raise ValueError("Retrieval normalized flag must be boolean.")
    similarity = str(_scalar(artifact, "similarity"))
    expected_similarity = (
        "cosine" if bool(_scalar(artifact, "normalized")) else "dot_product"
    )
    if similarity != expected_similarity:
        raise ValueError("Retrieval similarity is inconsistent with normalization.")

    required = (
        "classes", "ks", "sample_ids", "group_ids", "unit_query_indices",
        "unit_sample_ids", "unit_group_ids", "unit_label_indices", "eligible",
        "exclusion_reasons", "n_candidates", "n_relevant", "first_relevant_rank",
        "average_precision", "hit_at_k", "ordered_sample_ids_sha256",
    )
    for key in required:
        if key not in artifact:
            raise ValueError(f"Retrieval artifact is missing required field '{key}'.")

    classes = np.asarray(artifact["classes"]).astype(str)
    ks = np.asarray(artifact["ks"])
    sample_ids = np.asarray(artifact["sample_ids"]).astype(str)
    group_ids = np.asarray(artifact["group_ids"]).astype(str)
    query_indices = np.asarray(artifact["unit_query_indices"])
    unit_sample_ids = np.asarray(artifact["unit_sample_ids"]).astype(str)
    unit_group_ids = np.asarray(artifact["unit_group_ids"]).astype(str)
    label_indices = np.asarray(artifact["unit_label_indices"])
    eligible = np.asarray(artifact["eligible"])
    reasons = np.asarray(artifact["exclusion_reasons"]).astype(str)
    n_candidates = np.asarray(artifact["n_candidates"])
    n_relevant = np.asarray(artifact["n_relevant"])
    first_rank = np.asarray(artifact["first_relevant_rank"])
    average_precision = np.asarray(artifact["average_precision"], dtype=np.float64)
    hit_at_k = np.asarray(artifact["hit_at_k"])

    if classes.ndim != 1 or len(classes) < 1:
        raise ValueError("Retrieval classes must be a non-empty vector.")
    if len(np.unique(classes)) != len(classes) or np.any(classes == ""):
        raise ValueError("Retrieval classes must be unique and non-empty.")
    if ks.ndim != 1 or len(ks) < 1 or not np.issubdtype(ks.dtype, np.integer):
        raise ValueError("Retrieval K values must be an integer vector.")
    if np.any(ks < 1) or len(np.unique(ks)) != len(ks):
        raise ValueError("Retrieval K values must be positive and unique.")
    if sample_ids.ndim != 1 or len(sample_ids) < 1:
        raise ValueError("Retrieval sample IDs must be a non-empty vector.")
    if len(np.unique(sample_ids)) != len(sample_ids) or np.any(sample_ids == ""):
        raise ValueError("Retrieval sample IDs must be unique and non-empty.")
    if group_ids.shape != sample_ids.shape or np.any(group_ids == ""):
        raise ValueError("Retrieval group IDs must be non-empty and aligned.")

    n_units = len(query_indices)
    vector_fields = (
        unit_sample_ids, unit_group_ids, label_indices, eligible, reasons,
        n_candidates, n_relevant, first_rank, average_precision,
    )
    if query_indices.ndim != 1 or any(value.shape != (n_units,) for value in vector_fields):
        raise ValueError("Retrieval unit arrays must be aligned vectors.")
    integer_fields = (query_indices, label_indices, n_candidates, n_relevant, first_rank)
    if any(not np.issubdtype(value.dtype, np.integer) for value in integer_fields):
        raise ValueError("Retrieval indices and counts must use integer dtypes.")
    if eligible.dtype != np.bool_:
        raise ValueError("Retrieval eligibility must use a boolean dtype.")
    if hit_at_k.shape != (n_units, len(ks)) or hit_at_k.dtype != np.bool_:
        raise ValueError("Retrieval Hit@K values must be an aligned boolean matrix.")

    if n_units:
        if np.any(query_indices < 0) or np.any(query_indices >= len(sample_ids)):
            raise ValueError("Retrieval query indices are out of bounds.")
        if not np.array_equal(unit_sample_ids, sample_ids[query_indices]):
            raise ValueError("Retrieval unit sample IDs are misaligned.")
        if not np.array_equal(unit_group_ids, group_ids[query_indices]):
            raise ValueError("Retrieval unit group IDs are misaligned.")
        if np.any(label_indices < 0) or np.any(label_indices >= len(classes)):
            raise ValueError("Retrieval label indices are outside the class set.")
        pairs = list(zip(query_indices.tolist(), label_indices.tolist()))
        if len(set(pairs)) != n_units:
            raise ValueError("Retrieval query-label units must be unique.")
        if np.any(n_candidates < 0) or np.any(n_candidates >= len(sample_ids)):
            raise ValueError("Retrieval candidate counts are invalid.")
        if np.any(n_relevant < 0) or np.any(n_relevant > n_candidates):
            raise ValueError("Retrieval relevant-candidate counts are invalid.")

    expected_eligible = n_relevant > 0
    if not np.array_equal(eligible, expected_eligible):
        raise ValueError("Retrieval eligibility must equal n_relevant > 0.")
    valid_reasons = {"", "no_candidates", "no_relevant_candidate"}
    if not set(reasons.tolist()).issubset(valid_reasons):
        raise ValueError("Retrieval artifact contains an unknown exclusion reason.")
    expected_reasons = np.where(
        eligible, "", np.where(n_candidates == 0, "no_candidates", "no_relevant_candidate")
    )
    if not np.array_equal(reasons, expected_reasons):
        raise ValueError("Retrieval exclusion reasons are inconsistent with counts.")

    if np.any(~np.isfinite(average_precision[eligible])):
        raise ValueError("Eligible retrieval units must have finite average precision.")
    if np.any((average_precision[eligible] <= 0) | (average_precision[eligible] > 1)):
        raise ValueError("Eligible average precision must lie in (0, 1].")
    if np.any(~np.isnan(average_precision[~eligible])):
        raise ValueError("Excluded retrieval units must have NaN average precision.")
    if np.any(first_rank[eligible] < 1) or np.any(first_rank[eligible] > n_candidates[eligible]):
        raise ValueError("Eligible first-relevant ranks are invalid.")
    if np.any(first_rank[~eligible] != -1):
        raise ValueError("Excluded first-relevant ranks must be -1.")
    expected_hits = eligible[:, None] & (first_rank[:, None] <= ks[None, :])
    if not np.array_equal(hit_at_k, expected_hits):
        raise ValueError("Hit@K values disagree with first-relevant ranks.")

    expected_hash = ordered_ids_sha256(sample_ids.tolist())
    if str(_scalar(artifact, "ordered_sample_ids_sha256")) != expected_hash:
        raise ValueError("Retrieval ordered sample-ID fingerprint is invalid.")
    return True


def summary_from_retrieval_artifact(artifact):
    """Reconstruct headline metrics solely from eligible saved units."""
    validate_retrieval_artifact(artifact)
    eligible = np.asarray(artifact["eligible"], dtype=bool)
    ks = np.asarray(artifact["ks"], dtype=np.int64)
    hits = np.asarray(artifact["hit_at_k"], dtype=bool)
    average_precision = np.asarray(artifact["average_precision"], dtype=np.float64)
    query_indices = np.asarray(artifact["unit_query_indices"], dtype=np.int64)
    n_samples = len(np.asarray(artifact["sample_ids"]))
    if not np.any(eligible):
        return {
            "n_total": n_samples,
            "n_eval": 0,
            "n_excluded_queries": n_samples,
            "n_evaluation_units": 0,
            "hit_at_k": {int(k): np.nan for k in ks},
            "map": np.nan,
        }
    evaluated_queries = np.unique(query_indices[eligible])
    return {
        "n_total": n_samples,
        "n_eval": int(len(evaluated_queries)),
        "n_excluded_queries": int(n_samples - len(evaluated_queries)),
        "n_evaluation_units": int(np.sum(eligible)),
        "hit_at_k": {
            int(k): float(np.mean(hits[eligible, index]))
            for index, k in enumerate(ks)
        },
        "map": float(np.mean(average_precision[eligible])),
    }


def validate_retrieval_summary(artifact, summary):
    """Require the serialized evidence to reproduce the JSON headline summary."""
    reconstructed = summary_from_retrieval_artifact(artifact)
    for key in ("n_total", "n_eval", "n_excluded_queries", "n_evaluation_units"):
        if int(summary[key]) != reconstructed[key]:
            raise ValueError(f"Retrieval artifact does not reproduce summary field '{key}'.")
    for k, value in reconstructed["hit_at_k"].items():
        observed = summary["hit_at_k"][k]
        if not np.allclose(observed, value, equal_nan=True, rtol=0, atol=1e-12):
            raise ValueError(f"Retrieval artifact does not reproduce Hit@{k}.")
    if not np.allclose(summary["map"], reconstructed["map"], equal_nan=True, rtol=0, atol=1e-12):
        raise ValueError("Retrieval artifact does not reproduce mAP.")
    return True


def validate_comparable_retrieval_artifacts(artifacts):
    """Require artifacts to contain the same paired statistical units."""
    artifacts = list(artifacts)
    if len(artifacts) < 2:
        raise ValueError("At least two retrieval artifacts are required for comparison.")
    for artifact in artifacts:
        validate_retrieval_artifact(artifact)

    reference = artifacts[0]
    scalar_fields = (
        "dataset_name", "label_type", "similarity", "normalized",
        "candidate_exclusion",
    )
    paired_arrays = (
        "classes", "ks", "sample_ids", "group_ids", "unit_query_indices",
        "unit_sample_ids", "unit_group_ids", "unit_label_indices", "eligible",
        "exclusion_reasons", "n_candidates", "n_relevant",
    )
    for artifact in artifacts[1:]:
        for key in scalar_fields:
            if _scalar(artifact, key) != _scalar(reference, key):
                raise ValueError(
                    f"Retrieval artifacts differ in comparison field '{key}'."
                )
        for key in paired_arrays:
            if not np.array_equal(np.asarray(artifact[key]), np.asarray(reference[key])):
                raise ValueError(
                    f"Retrieval artifacts do not align on paired field '{key}'."
                )
    return True


def write_retrieval_artifact(artifact, summary, output_dir, filename="queries.npz",
                             relative_path=None):
    """Atomically write, reload, and revalidate a compressed retrieval artifact."""
    validate_retrieval_summary(artifact, summary)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp.npz", dir=output_dir
    )
    os.close(fd)
    try:
        np.savez_compressed(temp_path, **artifact)
        with np.load(temp_path, allow_pickle=False) as loaded:
            reloaded = {key: loaded[key] for key in loaded.files}
        validate_retrieval_summary(reloaded, summary)
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    digest = hashlib.sha256()
    with open(filepath, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    eligible = np.asarray(artifact["eligible"], dtype=bool)
    return {
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "path": relative_path if relative_path is not None else filepath,
        "sha256": digest.hexdigest(),
        "size_bytes": int(os.path.getsize(filepath)),
        "n_samples": int(len(artifact["sample_ids"])),
        "n_units": int(len(eligible)),
        "n_eligible_units": int(np.sum(eligible)),
        "n_excluded_units": int(np.sum(~eligible)),
        "ordered_sample_ids_sha256": str(
            np.asarray(artifact["ordered_sample_ids_sha256"]).item()
        ),
    }
