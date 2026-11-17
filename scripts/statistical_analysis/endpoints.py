"""Reconstruct prespecified endpoints from validated evidence artifacts."""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from scripts.oof import pooled_macro_f1_from_oof
from scripts.retrieval_artifacts import summary_from_retrieval_artifact
from scripts.statistical_analysis.loaders import LoadedDataset, StatisticalInputError


@dataclass(frozen=True)
class ReconstructedEndpoints:
    model_id: str
    display_name: str
    dataset: str
    classification_adapter: str
    pooled_oof_macro_f1: float
    label_macro_map: float
    query_micro_map: float
    n_samples: int
    n_groups: int
    n_retrieval_units: int
    n_eligible_retrieval_units: int
    per_label_map: Mapping[str, Optional[float]]
    per_label_eligible_units: Mapping[str, int]


def _check_stored_value(stored, reconstructed, context):
    if stored is None:
        return
    if not np.isclose(float(stored), float(reconstructed), rtol=0, atol=1e-12):
        raise StatisticalInputError(
            f"Stored {context} ({stored}) does not match reconstructed value "
            f"({reconstructed})."
        )


def reconstruct_endpoints(loaded: LoadedDataset, *, adapter, require_all_labels_eligible):
    """Reconstruct both primary endpoints and verify stored values when available."""
    classification = float(pooled_macro_f1_from_oof(loaded.oof))
    retrieval_summary = summary_from_retrieval_artifact(loaded.retrieval)
    retrieval = float(retrieval_summary["label_macro_map"])
    if not np.isfinite(classification) or not np.isfinite(retrieval):
        raise StatisticalInputError(
            f"Reconstructed endpoint is non-finite for "
            f"{loaded.run.model.model_id}/{loaded.dataset}."
        )

    per_label = retrieval_summary["per_label"]
    empty_labels = [
        label for label, row in per_label.items() if int(row["n_queries"]) == 0
    ]
    if empty_labels and require_all_labels_eligible:
        raise StatisticalInputError(
            f"Retrieval labels have no eligible units for "
            f"{loaded.run.model.model_id}/{loaded.dataset}: {empty_labels}"
        )

    classification_json_key = {
        "logistic_regression": "logreg_cv",
        "knn": "knn_cv",
        "mlp": "mlp_cv",
    }.get(adapter)
    if classification_json_key:
        stored = loaded.result.get(classification_json_key, {}).get(
            "pooled_oof_f1_macro"
        )
        _check_stored_value(
            stored, classification,
            f"pooled OOF macro-F1 for {loaded.run.model.model_id}/{loaded.dataset}",
        )
    _check_stored_value(
        loaded.result.get("retrieval", {}).get("label_macro_map"),
        retrieval,
        f"label-macro mAP for {loaded.run.model.model_id}/{loaded.dataset}",
    )

    sample_ids = np.asarray(loaded.oof["sample_ids"])
    group_ids = np.asarray(loaded.oof["group_ids"]).astype(str)
    eligible = np.asarray(loaded.retrieval["eligible"], dtype=bool)
    return ReconstructedEndpoints(
        model_id=loaded.run.model.model_id,
        display_name=loaded.run.model.display_name,
        dataset=loaded.dataset,
        classification_adapter=adapter,
        pooled_oof_macro_f1=classification,
        label_macro_map=retrieval,
        query_micro_map=float(retrieval_summary["query_micro_map"]),
        n_samples=int(len(sample_ids)),
        n_groups=int(len(np.unique(group_ids))),
        n_retrieval_units=int(len(eligible)),
        n_eligible_retrieval_units=int(np.sum(eligible)),
        per_label_map={
            label: float(row["map"]) if np.isfinite(row["map"]) else None
            for label, row in per_label.items()
        },
        per_label_eligible_units={
            label: int(row["n_queries"]) for label, row in per_label.items()
        },
    )
