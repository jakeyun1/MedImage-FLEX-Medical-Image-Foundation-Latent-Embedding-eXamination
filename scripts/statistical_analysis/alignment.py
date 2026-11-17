"""Exact paired-unit alignment checks across model runs."""

from typing import Sequence

import numpy as np

from scripts.statistical_analysis.loaders import LoadedDataset, StatisticalInputError


OOF_SCALARS = ("dataset_name", "adapter", "label_type", "score_type", "fold_index_base")
OOF_PAIRED_ARRAYS = (
    "classes", "sample_ids", "group_ids", "outer_folds", "y_true",
)
RETRIEVAL_SCALARS = (
    "dataset_name", "label_type", "similarity", "normalized", "candidate_exclusion",
)
RETRIEVAL_PAIRED_ARRAYS = (
    "classes", "ks", "sample_ids", "group_ids", "unit_query_indices",
    "unit_sample_ids", "unit_group_ids", "unit_label_indices", "eligible",
    "exclusion_reasons", "n_candidates", "n_relevant",
)


def _scalar(artifact, key):
    return np.asarray(artifact[key]).item()


def _compare_artifacts(reference, candidate, *, scalar_fields, array_fields, context):
    for field in scalar_fields:
        if _scalar(reference, field) != _scalar(candidate, field):
            raise StatisticalInputError(f"{context} differs on scalar field {field!r}.")
    for field in array_fields:
        if not np.array_equal(np.asarray(reference[field]), np.asarray(candidate[field])):
            raise StatisticalInputError(f"{context} differs on paired field {field!r}.")


def _optional_metadata(result):
    info = result.get("dataset_info", {})
    reproducibility = result.get("reproducibility", {})
    fingerprints = result.get("data_audit", {}).get("fingerprints", {})
    return {
        "dataset_info.label_type": info.get("label_type"),
        "dataset_info.evaluation_unit": info.get("evaluation_unit"),
        "dataset_info.classes": info.get("classes"),
        "reproducibility.sha256": reproducibility.get("sha256"),
        "data_audit.retained_sample_ids_sha256": fingerprints.get(
            "retained_sample_ids_sha256"
        ),
        "data_audit.retained_sample_group_labels_sha256": fingerprints.get(
            "retained_sample_group_labels_sha256"
        ),
    }


def validate_dataset_alignment(records: Sequence[LoadedDataset], reference_model_id):
    """Fail unless all model outcomes are defined on identical paired inputs."""
    records = list(records)
    if not records:
        raise StatisticalInputError("At least one loaded dataset is required for alignment.")
    datasets = {record.dataset for record in records}
    if len(datasets) != 1:
        raise StatisticalInputError(
            f"Alignment received multiple datasets: {sorted(datasets)}"
        )

    reference = next(
        (record for record in records if record.run.model.model_id == reference_model_id),
        records[0],
    )
    reference_metadata = _optional_metadata(reference.result)
    for candidate in records:
        if candidate is reference:
            continue
        pair = (
            f"{reference.run.model.model_id} vs "
            f"{candidate.run.model.model_id}/{candidate.dataset}"
        )
        _compare_artifacts(
            reference.oof, candidate.oof,
            scalar_fields=OOF_SCALARS,
            array_fields=OOF_PAIRED_ARRAYS,
            context=f"OOF alignment ({pair})",
        )
        _compare_artifacts(
            reference.retrieval, candidate.retrieval,
            scalar_fields=RETRIEVAL_SCALARS,
            array_fields=RETRIEVAL_PAIRED_ARRAYS,
            context=f"Retrieval alignment ({pair})",
        )
        candidate_metadata = _optional_metadata(candidate.result)
        for field, expected in reference_metadata.items():
            observed = candidate_metadata[field]
            if expected is not None and observed is not None and observed != expected:
                raise StatisticalInputError(
                    f"Result metadata alignment ({pair}) differs on {field!r}."
                )

    return {
        "status": "passed",
        "reference_model_id": reference.run.model.model_id,
        "n_models": len(records),
        "n_samples": int(len(np.asarray(reference.oof["sample_ids"]))),
        "n_groups": int(len(np.unique(np.asarray(reference.oof["group_ids"]).astype(str)))),
        "n_retrieval_units": int(len(np.asarray(reference.retrieval["eligible"]))),
        "n_eligible_retrieval_units": int(np.sum(reference.retrieval["eligible"])),
        "checks": [
            "classification_and_retrieval_within_model",
            "oof_paired_inputs_across_models",
            "retrieval_paired_inputs_across_models",
            "cohort_metadata_across_models_when_present",
        ],
    }
