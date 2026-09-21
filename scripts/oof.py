"""Validated out-of-fold prediction artifacts for classification benchmarks."""

import hashlib
import os
import tempfile

import numpy as np

from scripts.data_audit import ordered_ids_sha256


OOF_SCHEMA_VERSION = 1


class OOFAccumulator:
    """Collect each sample's held-out prediction exactly once in source order."""

    def __init__(self, dataset_name, adapter, sample_ids, group_ids, y_true,
                 classes, is_multilabel):
        self.dataset_name = str(dataset_name)
        self.adapter = str(adapter)
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        self.group_ids = np.asarray(
            sample_ids if group_ids is None else group_ids, dtype=str
        )
        raw_y_true = np.asarray(y_true)
        if not (
            np.issubdtype(raw_y_true.dtype, np.integer)
            or np.issubdtype(raw_y_true.dtype, np.bool_)
        ):
            raise ValueError("OOF targets must use an integer or boolean dtype.")
        self.expected_y_true = raw_y_true.astype(np.int64, copy=False)
        self.classes = np.asarray([str(value) for value in classes], dtype=str)
        self.is_multilabel = bool(is_multilabel)

        n_samples = len(self.sample_ids)
        if n_samples == 0:
            raise ValueError("OOF collection requires at least one sample.")
        if len(np.unique(self.sample_ids)) != n_samples:
            raise ValueError("OOF sample IDs must be unique.")
        if len(self.group_ids) != n_samples:
            raise ValueError("OOF group IDs must match the sample count.")
        if np.any(self.group_ids == ""):
            raise ValueError("OOF group IDs must be non-empty.")
        if self.expected_y_true.shape[0] != n_samples:
            raise ValueError("OOF targets must match the sample count.")

        if self.is_multilabel:
            if self.expected_y_true.ndim != 2:
                raise ValueError("Multilabel OOF targets must be a matrix.")
            n_outputs = self.expected_y_true.shape[1]
        else:
            if self.expected_y_true.ndim != 1:
                raise ValueError("Multiclass OOF targets must be a vector.")
            n_outputs = len(self.classes)
        if n_outputs != len(self.classes) or n_outputs < 2:
            raise ValueError("OOF classes do not match the target width.")

        self.coverage = np.zeros(n_samples, dtype=np.int8)
        self.outer_folds = np.full(n_samples, -1, dtype=np.int64)
        self.y_pred = np.empty_like(self.expected_y_true)
        self.y_score = np.full((n_samples, n_outputs), np.nan, dtype=np.float64)

    def record(self, test_indices, fold_index, y_true, y_pred, y_score):
        indices = np.asarray(test_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("Each OOF fold must contain at least one index.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("An OOF fold contains duplicate sample indices.")
        if indices.min() < 0 or indices.max() >= len(self.sample_ids):
            raise ValueError("An OOF fold contains an out-of-range sample index.")
        if np.any(self.coverage[indices] != 0):
            raise ValueError("An OOF sample was predicted more than once.")
        if not isinstance(fold_index, (int, np.integer)) or fold_index < 0:
            raise ValueError("OOF fold indices must be non-negative integers.")

        observed_true_raw = np.asarray(y_true)
        observed_pred_raw = np.asarray(y_pred)
        for name, values in (
            ("targets", observed_true_raw), ("predictions", observed_pred_raw)
        ):
            if not (
                np.issubdtype(values.dtype, np.integer)
                or np.issubdtype(values.dtype, np.bool_)
            ):
                raise ValueError(f"OOF {name} must use an integer or boolean dtype.")
        observed_true = observed_true_raw.astype(np.int64, copy=False)
        observed_pred = observed_pred_raw.astype(np.int64, copy=False)
        observed_score = np.asarray(y_score, dtype=np.float64)
        expected_shape = self.expected_y_true[indices].shape
        if observed_true.shape != expected_shape:
            raise ValueError("OOF held-out targets have an unexpected shape.")
        if not np.array_equal(observed_true, self.expected_y_true[indices]):
            raise ValueError("OOF held-out targets are misaligned with sample IDs.")
        if observed_pred.shape != expected_shape:
            raise ValueError("OOF predictions have an unexpected shape.")
        expected_score_shape = (len(indices), len(self.classes))
        if observed_score.shape != expected_score_shape:
            raise ValueError(
                f"OOF scores must have shape {expected_score_shape}; "
                f"found {observed_score.shape}."
            )
        if not np.isfinite(observed_score).all():
            raise ValueError("OOF scores must all be finite.")
        tolerance = 1e-7
        if np.any(observed_score < -tolerance) or np.any(observed_score > 1 + tolerance):
            raise ValueError("OOF probabilities must lie within [0, 1].")
        if not self.is_multilabel and not np.allclose(
            observed_score.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Multiclass OOF probabilities must sum to one.")

        self.coverage[indices] = 1
        self.outer_folds[indices] = int(fold_index)
        self.y_pred[indices] = observed_pred
        self.y_score[indices] = observed_score

    def finalize(self):
        if not np.all(self.coverage == 1):
            missing = self.sample_ids[self.coverage == 0].tolist()
            raise ValueError(f"OOF predictions are missing samples: {missing[:5]}")

        artifact = {
            "schema_version": np.asarray(OOF_SCHEMA_VERSION, dtype=np.int64),
            "dataset_name": np.asarray(self.dataset_name),
            "adapter": np.asarray(self.adapter),
            "label_type": np.asarray(
                "multilabel" if self.is_multilabel else "multiclass"
            ),
            "score_type": np.asarray("probability"),
            "fold_index_base": np.asarray(0, dtype=np.int64),
            "classes": self.classes,
            "sample_ids": self.sample_ids,
            "group_ids": self.group_ids,
            "outer_folds": self.outer_folds,
            "y_true": self.expected_y_true,
            "y_pred": self.y_pred,
            "y_score": self.y_score,
            "ordered_sample_ids_sha256": np.asarray(
                ordered_ids_sha256(self.sample_ids.tolist())
            ),
        }
        validate_oof_artifact(artifact)
        return artifact


def _scalar(artifact, key):
    if key not in artifact:
        raise ValueError(f"OOF artifact is missing required field '{key}'.")
    value = np.asarray(artifact[key])
    if value.ndim != 0:
        raise ValueError(f"OOF field '{key}' must be scalar.")
    return value.item()


def validate_oof_artifact(artifact):
    """Reject incomplete, misaligned, non-finite, or group-leaking artifacts."""
    if int(_scalar(artifact, "schema_version")) != OOF_SCHEMA_VERSION:
        raise ValueError("OOF artifact schema version is incompatible.")
    if _scalar(artifact, "label_type") not in {"multiclass", "multilabel"}:
        raise ValueError("OOF label type is invalid.")
    if _scalar(artifact, "score_type") != "probability":
        raise ValueError("OOF score type must be probability.")
    if int(_scalar(artifact, "fold_index_base")) != 0:
        raise ValueError("OOF folds must use zero-based indices.")
    for key in ("dataset_name", "adapter"):
        if not str(_scalar(artifact, key)).strip():
            raise ValueError(f"OOF field '{key}' must be non-empty.")

    required_arrays = (
        "classes", "sample_ids", "group_ids", "outer_folds",
        "y_true", "y_pred", "y_score",
    )
    for key in required_arrays:
        if key not in artifact:
            raise ValueError(f"OOF artifact is missing required field '{key}'.")

    classes = np.asarray(artifact["classes"]).astype(str)
    sample_ids = np.asarray(artifact["sample_ids"]).astype(str)
    group_ids = np.asarray(artifact["group_ids"]).astype(str)
    folds = np.asarray(artifact["outer_folds"])
    y_true = np.asarray(artifact["y_true"])
    y_pred = np.asarray(artifact["y_pred"])
    y_score = np.asarray(artifact["y_score"], dtype=np.float64)
    n_samples = len(sample_ids)

    if classes.ndim != 1 or len(classes) < 2:
        raise ValueError("OOF classes must be a one-dimensional label set.")
    if len(np.unique(classes)) != len(classes) or np.any(classes == ""):
        raise ValueError("OOF classes must be unique and non-empty.")
    if sample_ids.ndim != 1 or n_samples == 0:
        raise ValueError("OOF sample IDs must be a non-empty vector.")
    if len(np.unique(sample_ids)) != n_samples:
        raise ValueError("OOF sample IDs must be unique.")
    if group_ids.shape != sample_ids.shape or np.any(group_ids == ""):
        raise ValueError("OOF group IDs must be non-empty and aligned.")
    if folds.shape != (n_samples,) or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("OOF fold assignments must be an integer vector.")
    if np.any(folds < 0):
        raise ValueError("OOF fold assignments must be non-negative.")
    if sorted(np.unique(folds).tolist()) != list(range(int(folds.max()) + 1)):
        raise ValueError("OOF fold assignments must be consecutive from zero.")
    if y_true.shape != y_pred.shape or y_true.shape[0] != n_samples:
        raise ValueError("OOF targets and predictions must be aligned.")
    for name, values in (("targets", y_true), ("predictions", y_pred)):
        if not (
            np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise ValueError(f"OOF {name} must use an integer or boolean dtype.")
    if y_score.shape != (n_samples, len(classes)):
        raise ValueError("OOF score columns must match the class set.")
    if not np.isfinite(y_score).all():
        raise ValueError("OOF scores must all be finite.")
    if np.any(y_score < -1e-7) or np.any(y_score > 1 + 1e-7):
        raise ValueError("OOF probabilities must lie within [0, 1].")

    label_type = str(_scalar(artifact, "label_type"))
    if label_type == "multiclass":
        if y_true.ndim != 1:
            raise ValueError("Multiclass OOF targets must be vectors.")
        if not np.allclose(y_score.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
            raise ValueError("Multiclass OOF probabilities must sum to one.")
        if np.any(y_true < 0) or np.any(y_true >= len(classes)):
            raise ValueError("Multiclass OOF targets are outside the class set.")
        if np.any(y_pred < 0) or np.any(y_pred >= len(classes)):
            raise ValueError("Multiclass OOF predictions are outside the class set.")
    else:
        if y_true.shape != (n_samples, len(classes)):
            raise ValueError("Multilabel OOF targets must match the label set.")
        if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred, (0, 1)).all():
            raise ValueError("Multilabel OOF targets and predictions must be binary.")

    group_fold_pairs = set()
    fold_by_group = {}
    for group_id, fold in zip(group_ids.tolist(), folds.tolist()):
        group_fold_pairs.add((group_id, int(fold)))
        fold_by_group.setdefault(group_id, set()).add(int(fold))
    leaking = [group for group, values in fold_by_group.items() if len(values) != 1]
    if leaking:
        raise ValueError(f"OOF groups span multiple outer folds: {leaking[:5]}")
    if not group_fold_pairs:
        raise ValueError("OOF artifact contains no evaluation groups.")

    expected_hash = ordered_ids_sha256(sample_ids.tolist())
    if str(_scalar(artifact, "ordered_sample_ids_sha256")) != expected_hash:
        raise ValueError("OOF ordered sample-ID fingerprint is invalid.")
    return True


def write_oof_artifact(artifact, output_dir, filename, relative_path=None):
    """Atomically write a validated compressed artifact and return JSON metadata."""
    validate_oof_artifact(artifact)
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
        validate_oof_artifact(reloaded)
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    digest = hashlib.sha256()
    with open(filepath, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "schema_version": OOF_SCHEMA_VERSION,
        "path": relative_path if relative_path is not None else filepath,
        "sha256": digest.hexdigest(),
        "size_bytes": int(os.path.getsize(filepath)),
        "n_samples": int(len(artifact["sample_ids"])),
        "n_groups": int(len(np.unique(artifact["group_ids"]))),
        "n_outputs": int(len(artifact["classes"])),
        "ordered_sample_ids_sha256": str(
            np.asarray(artifact["ordered_sample_ids_sha256"]).item()
        ),
    }
