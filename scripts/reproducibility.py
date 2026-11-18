"""
reproducibility.py

Deterministic cohort selection and reusable outer cross-validation folds.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from scripts.dataset_contracts import get_dataset_contract

PROHIBITED_CHARS = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]
MANIFEST_SCHEMA_VERSION = 4
COHORT_SENSITIVITY_SCHEMA_VERSION = 1
DEFAULT_COHORT_SENSITIVITY_SEEDS = (43, 44, 45)
COHORT_SENSITIVITY_DATASETS = ("ham10000", "chexpert")


def validate_cohort_sensitivity_seeds(primary_seed, sensitivity_seeds):
    """Validate deterministic, non-overlapping cohort sensitivity seeds."""
    if not isinstance(primary_seed, int) or isinstance(primary_seed, bool) or primary_seed < 0:
        raise ValueError("reproducibility.sample_seed must be a nonnegative integer.")
    if not isinstance(sensitivity_seeds, list):
        raise ValueError("reproducibility.cohort_sensitivity_seeds must be a JSON list.")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in sensitivity_seeds
    ):
        raise ValueError("Cohort sensitivity seeds must be nonnegative integers.")
    if len(sensitivity_seeds) != len(set(sensitivity_seeds)):
        raise ValueError("Cohort sensitivity seeds must be unique.")
    if primary_seed in sensitivity_seeds:
        raise ValueError("The primary sample seed cannot also be a sensitivity seed.")
    return tuple(sensitivity_seeds)


def cohort_overlap_summary(primary_sample_ids, sensitivity_sample_ids):
    """Return exact overlap evidence for two unique manifest cohorts."""
    primary = [str(sample_id) for sample_id in primary_sample_ids]
    sensitivity = [str(sample_id) for sample_id in sensitivity_sample_ids]
    if not primary or not sensitivity:
        raise ValueError("Cohort overlap requires two nonempty sample-ID lists.")
    if len(primary) != len(set(primary)) or len(sensitivity) != len(set(sensitivity)):
        raise ValueError("Cohort overlap sample IDs must be unique.")
    intersection = len(set(primary) & set(sensitivity))
    union = len(set(primary) | set(sensitivity))
    return {
        "primary_n_samples": int(len(primary)),
        "sensitivity_n_samples": int(len(sensitivity)),
        "intersection_n_samples": int(intersection),
        "union_n_samples": int(union),
        "primary_retained_fraction": float(intersection / len(primary)),
        "jaccard_index": float(intersection / union),
        "identical_sample_set": set(primary) == set(sensitivity),
    }


def cohort_seed_tag(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Cohort seed must be a nonnegative integer.")
    return f"seed_{seed:010d}"


def validate_cohort_selection_record(record):
    """Fail closed on malformed primary or sensitivity cohort metadata."""
    if not isinstance(record, dict):
        raise ValueError("Cohort selection metadata must be an object.")
    common = {
        "schema_version", "analysis_type", "role", "primary_sample_seed",
        "sample_seed", "sensitivity_sample_seeds", "max_samples",
        "n_eligible_samples", "n_selected_samples", "manifest_sha256",
        "manifest_applied_after_full_embedding_extraction", "fold_seed",
        "evaluation_seed", "outer_folds",
    }
    sensitivity_only = {
        "primary_manifest_sha256", "full_embedding_bank_reused",
        "additional_embedding_extraction", "overlap_with_primary",
        "primary_n_selected_samples",
    }
    role = record.get("role")
    expected = common if role == "primary" else common | sensitivity_only
    if role not in {"primary", "sensitivity"} or set(record) != expected:
        raise ValueError("Cohort selection metadata fields do not match its role.")
    if record["schema_version"] != COHORT_SENSITIVITY_SCHEMA_VERSION:
        raise ValueError("Cohort selection schema version is incompatible.")
    if record["analysis_type"] != "cohort_sampling":
        raise ValueError("Cohort selection analysis type is invalid.")
    primary_seed = record["primary_sample_seed"]
    sensitivity_seeds = validate_cohort_sensitivity_seeds(
        primary_seed, record["sensitivity_sample_seeds"]
    )
    sample_seed = record["sample_seed"]
    if role == "primary" and sample_seed != primary_seed:
        raise ValueError("Primary cohort selection must use the primary sample seed.")
    if role == "sensitivity" and sample_seed not in sensitivity_seeds:
        raise ValueError("Sensitivity cohort seed is not declared in its seed family.")
    for key in ("fold_seed", "evaluation_seed"):
        if not isinstance(record[key], int) or isinstance(record[key], bool) or record[key] < 0:
            raise ValueError(f"Cohort selection {key} must be a nonnegative integer.")
    if (
        not isinstance(record["outer_folds"], int)
        or isinstance(record["outer_folds"], bool)
        or record["outer_folds"] < 2
    ):
        raise ValueError("Cohort selection outer_folds must be an integer of at least 2.")
    for key in ("n_eligible_samples", "n_selected_samples"):
        if not isinstance(record[key], int) or isinstance(record[key], bool) or record[key] < 1:
            raise ValueError(f"Cohort selection {key} must be a positive integer.")
    if record["n_selected_samples"] > record["n_eligible_samples"]:
        raise ValueError("Selected cohort size cannot exceed eligible sample count.")
    if record["max_samples"] is not None and (
        not isinstance(record["max_samples"], int)
        or isinstance(record["max_samples"], bool)
        or record["max_samples"] < 1
    ):
        raise ValueError("Cohort max_samples must be null or a positive integer.")
    for key in ("manifest_sha256",):
        value = record[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Cohort selection {key} must be a SHA-256 string.")
    if record["manifest_applied_after_full_embedding_extraction"] is not True:
        raise ValueError("Cohort selection must follow full embedding extraction.")
    if role == "sensitivity":
        if not isinstance(record["primary_manifest_sha256"], str) or len(
            record["primary_manifest_sha256"]
        ) != 64:
            raise ValueError("Primary manifest SHA-256 is invalid.")
        if record["full_embedding_bank_reused"] is not True:
            raise ValueError("Sensitivity cohorts must reuse the full embedding bank.")
        if record["additional_embedding_extraction"] is not False:
            raise ValueError("Sensitivity cohorts cannot report additional extraction.")
        overlap = record["overlap_with_primary"]
        if not isinstance(overlap, dict):
            raise ValueError("Sensitivity cohort overlap metadata is missing.")
        if overlap.get("primary_n_samples") != record["primary_n_selected_samples"]:
            raise ValueError("Sensitivity overlap primary count is inconsistent.")
        if overlap.get("sensitivity_n_samples") != record["n_selected_samples"]:
            raise ValueError("Sensitivity overlap sample count is inconsistent.")
    return True


def validate_cohort_sensitivity_index(index):
    """Validate the per-model, per-dataset cohort sensitivity index."""
    required = {
        "schema_version", "analysis_type", "dataset", "model_id",
        "primary_sample_seed", "sensitivity_sample_seeds", "max_samples",
        "embedding_bank", "primary", "sensitivities", "fold_seed",
        "evaluation_seed", "outer_folds",
    }
    if not isinstance(index, dict) or set(index) != required:
        raise ValueError("Cohort sensitivity index fields are invalid.")
    if index["schema_version"] != COHORT_SENSITIVITY_SCHEMA_VERSION:
        raise ValueError("Cohort sensitivity index schema is incompatible.")
    if index["analysis_type"] != "cohort_sampling_sensitivity":
        raise ValueError("Cohort sensitivity index analysis type is invalid.")
    if not str(index["dataset"]).strip() or not str(index["model_id"]).strip():
        raise ValueError("Cohort sensitivity dataset and model must be nonempty.")
    seeds = validate_cohort_sensitivity_seeds(
        index["primary_sample_seed"], index["sensitivity_sample_seeds"]
    )
    for key in ("fold_seed", "evaluation_seed"):
        if not isinstance(index[key], int) or isinstance(index[key], bool) or index[key] < 0:
            raise ValueError(f"Cohort sensitivity index {key} is invalid.")
    if (
        not isinstance(index["outer_folds"], int)
        or isinstance(index["outer_folds"], bool)
        or index["outer_folds"] < 2
    ):
        raise ValueError("Cohort sensitivity index outer_folds is invalid.")
    entries = index["sensitivities"]
    if not isinstance(entries, list) or [entry.get("sample_seed") for entry in entries] != list(seeds):
        raise ValueError("Cohort sensitivity index entries do not match configured seeds.")
    if index["embedding_bank"].get("reused_for_every_cohort") is not True:
        raise ValueError("Cohort sensitivity index must document embedding reuse.")
    if index["embedding_bank"].get("additional_embedding_extraction_per_seed") is not False:
        raise ValueError("Cohort sensitivity index cannot report per-seed extraction.")
    for entry in entries:
        expected_tag = cohort_seed_tag(entry["sample_seed"])
        result_path = str(entry.get("result_path", "")).replace("\\", "/")
        if not result_path.endswith(f"/{expected_tag}.json"):
            raise ValueError("Cohort sensitivity result path does not match its seed.")
        if not isinstance(entry.get("manifest_sha256"), str) or len(
            entry["manifest_sha256"]
        ) != 64:
            raise ValueError("Cohort sensitivity manifest SHA-256 is invalid.")
    return True

def _clean_filename(filename, desired_char):
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


def sample_ids_from_paths(dataset_name, image_paths):
    contract = get_dataset_contract(dataset_name)
    return [contract.sample_id_from_path(path) for path in image_paths]


def _label_signature(label):
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if isinstance(label, tuple):
        label = list(label)
    return json.dumps(label, sort_keys=True, separators=(",", ":"), default=str)


def _selection_strata(label_signatures, min_count=2):
    counts = label_signatures.value_counts()
    strata = label_signatures.where(
        label_signatures.map(counts) >= min_count, "__rare_label_combination__"
    )
    if strata.value_counts().min() < min_count:
        return None
    return strata


def _select_complete_groups(eligible, max_samples, sample_seed, n_splits):
    """Select whole groups while staying as close as possible to a sample target."""
    group_records = []
    for group_id, rows in eligible.groupby("_group_id", sort=True):
        group_records.append({
            "group_id": group_id,
            "sample_count": int(len(rows)),
            "label_signature": json.dumps(
                sorted(set(rows["_label_signature"])),
                separators=(",", ":"),
            ),
        })
    groups = pd.DataFrame.from_records(group_records)

    strata = _selection_strata(groups["label_signature"])
    rng = np.random.default_rng(sample_seed)
    if strata is None:
        groups["priority"] = rng.random(len(groups))
        selection_method = "deterministic_group_shuffle_nearest_sample_target"
    else:
        priorities = np.empty(len(groups), dtype=float)
        stratum_values = strata.to_numpy()
        for stratum in sorted(set(stratum_values)):
            indices = np.flatnonzero(stratum_values == stratum)
            permuted_indices = rng.permutation(indices)
            priorities[permuted_indices] = (
                np.arange(len(indices), dtype=float) + rng.random(len(indices))
            ) / len(indices)
        groups["priority"] = priorities
        selection_method = "group_label_stratified_nearest_sample_target"

    groups = groups.sort_values(
        ["priority", "group_id"], kind="stable"
    ).reset_index(drop=True)
    cumulative_samples = groups["sample_count"].cumsum().to_numpy()
    candidate_indices = np.arange(n_splits - 1, len(groups))
    closest_index = candidate_indices[
        np.argmin(np.abs(cumulative_samples[candidate_indices] - max_samples))
    ]
    selected_group_ids = set(groups.loc[:closest_index, "group_id"])
    selected = eligible[eligible["_group_id"].isin(selected_group_ids)].copy()
    return selected, selection_method


def _manifest_path(
    manifest_dir, model_id, dataset_name, group_col, max_samples, sample_seed, n_splits, fold_seed
):
    cohort = "all" if max_samples is None else str(max_samples)
    group_key = group_col.lower().replace(" ", "-")
    filename = (
        f"{_clean_filename(model_id, '-')}_{dataset_name}_group-{group_key}_v{MANIFEST_SCHEMA_VERSION}_"
        f"n{cohort}_sample{sample_seed}_"
        f"folds{n_splits}_fold{fold_seed}.csv"
    )
    return os.path.join(manifest_dir, filename)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_experiment_manifest(
    metadata_df,
    model_id,
    dataset_name,
    id_col,
    label_col,
    manifest_dir,
    group_col=None,
    max_samples=5000,
    sample_seed=42,
    n_splits=5,
    fold_seed=42,
    available_sample_ids=None,
):
    """
    Create or validate a stable cohort manifest and outer-fold assignment.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError("dataset.max_samples must be null or a positive integer.")
    if n_splits < 2:
        raise ValueError("evaluation.outer_folds must be at least 2.")

    group_col = id_col if group_col is None else group_col
    if group_col not in metadata_df.columns:
        raise ValueError(f"Grouping column is missing from metadata: {group_col}")

    eligible = metadata_df.dropna(subset=[id_col, label_col]).copy()
    if eligible[group_col].isna().any():
        raise ValueError(f"Grouping column contains missing values: {group_col}")
    eligible[id_col] = eligible[id_col].astype(str)
    eligible["_group_id"] = eligible[group_col].astype(str)
    if eligible["_group_id"].str.strip().eq("").any():
        raise ValueError(f"Grouping column contains blank values: {group_col}")
    if available_sample_ids is not None:
        available_sample_ids = {str(sample_id) for sample_id in available_sample_ids}
        eligible = eligible[eligible[id_col].isin(available_sample_ids)].copy()
    if eligible.empty:
        raise ValueError("No eligible metadata rows have corresponding image files.")
    if eligible[id_col].duplicated().any():
        duplicates = eligible.loc[eligible[id_col].duplicated(), id_col].head(5).tolist()
        raise ValueError(f"Dataset sample IDs must be unique; duplicates include {duplicates}.")

    eligible["_label_signature"] = eligible[label_col].map(_label_signature)
    eligible_sample_count = int(len(eligible))
    eligible_group_count = int(eligible["_group_id"].nunique())
    if eligible_group_count < n_splits:
        raise ValueError(f"At least {n_splits} distinct groups are required.")

    expected_selected = eligible
    selection_method = "all_eligible_samples"
    if max_samples is not None and len(eligible) > max_samples:
        expected_selected, selection_method = _select_complete_groups(
            eligible,
            max_samples=max_samples,
            sample_seed=sample_seed,
            n_splits=n_splits,
        )

    path = _manifest_path(
        manifest_dir, model_id, dataset_name, group_col, max_samples, sample_seed,
        n_splits, fold_seed
    )
    os.makedirs(manifest_dir, exist_ok=True)

    if os.path.exists(path):
        manifest = pd.read_csv(
            path,
            dtype={"sample_id": str, "group_id": str, "label_signature": str},
        )
        expected_columns = {"sample_id", "group_id", "label_signature", "outer_fold"}
        if set(manifest.columns) != expected_columns:
            raise ValueError(f"Manifest has unexpected columns: {path}")
        if manifest["sample_id"].duplicated().any():
            raise ValueError(f"Manifest contains duplicate sample IDs: {path}")

        expected_sample_ids = set(expected_selected[id_col].astype(str))
        if set(manifest["sample_id"]) != expected_sample_ids:
            raise ValueError(
                f"Manifest cohort no longer matches deterministic group selection: {path}"
            )

        indexed = eligible.set_index(id_col, drop=False)
        missing = sorted(set(manifest["sample_id"]) - set(indexed.index))
        if missing:
            raise ValueError(f"Manifest samples are missing from current metadata: {missing[:5]}")

        selected = indexed.loc[manifest["sample_id"]].copy()
        current_signatures = selected["_label_signature"].tolist()
        if current_signatures != manifest["label_signature"].tolist():
            raise ValueError(f"Labels no longer match the stored manifest: {path}")
        if selected["_group_id"].tolist() != manifest["group_id"].tolist():
            raise ValueError(f"Group IDs no longer match the stored manifest: {path}")
    else:
        selected = expected_selected
        selected = selected.sort_values(id_col, kind="stable").reset_index(drop=True)
        fold_strata = _selection_strata(
            selected["_label_signature"], min_count=n_splits
        )
        if fold_strata is None:
            fold_strata = pd.Series("__all_samples__", index=selected.index)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=fold_seed
        )
        split_iterator = splitter.split(
            selected, fold_strata, groups=selected["_group_id"]
        )

        outer_folds = np.full(len(selected), -1, dtype=int)
        for fold, (_, test_indices) in enumerate(split_iterator):
            outer_folds[test_indices] = fold
        if np.any(outer_folds < 0):
            raise RuntimeError("Failed to assign every selected sample to an outer fold.")

        manifest = pd.DataFrame({
            "sample_id": selected[id_col].astype(str),
            "group_id": selected["_group_id"],
            "label_signature": selected["_label_signature"],
            "outer_fold": outer_folds,
        })
        manifest.to_csv(path, index=False)

    manifest["outer_fold"] = manifest["outer_fold"].astype(int)
    observed_folds = sorted(manifest["outer_fold"].unique().tolist())
    if observed_folds != list(range(n_splits)):
        raise ValueError(
            f"Manifest folds must be numbered 0 through {n_splits - 1}: {path}"
        )
    if manifest.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise ValueError(f"A group appears in more than one outer fold: {path}")

    eligible_group_sizes = eligible.groupby("_group_id").size()
    manifest_group_sizes = manifest.groupby("group_id").size()
    incomplete_groups = sorted(
        group_id
        for group_id, sample_count in manifest_group_sizes.items()
        if sample_count != eligible_group_sizes.get(group_id, -1)
    )
    if incomplete_groups:
        raise ValueError(
            f"Manifest contains incomplete groups: {incomplete_groups[:5]} in {path}"
        )

    fold_strata = _selection_strata(selected["_label_signature"], min_count=n_splits)
    if fold_strata is None:
        fold_stratification = "group_only"
    elif (selected["_label_signature"].value_counts() < n_splits).any():
        fold_stratification = "label_signature_with_rare_pool"
    else:
        fold_stratification = "label_signature"

    selected = selected.drop(columns=["_label_signature", "_group_id"])
    selected[id_col] = selected[id_col].astype(str)
    fold_assignments = dict(zip(manifest["sample_id"], manifest["outer_fold"]))
    manifest_info = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "path": os.path.abspath(path),
        "sha256": _file_sha256(path),
        "n_samples": int(len(manifest)),
        "n_eligible_samples": eligible_sample_count,
        "max_samples": max_samples,
        "sample_target_deviation": (
            None if max_samples is None else int(len(manifest) - max_samples)
        ),
        "sample_seed": int(sample_seed),
        "selection_method": selection_method,
        "selection_unit": group_col,
        "complete_groups": True,
        "outer_folds": int(n_splits),
        "fold_seed": int(fold_seed),
        "fold_strategy": "stratified_group_kfold",
        "fold_stratification": fold_stratification,
        "fold_unit": group_col,
        "n_groups": int(manifest["group_id"].nunique()),
        "n_eligible_groups": eligible_group_count,
    }
    return selected.reset_index(drop=True), fold_assignments, manifest_info
