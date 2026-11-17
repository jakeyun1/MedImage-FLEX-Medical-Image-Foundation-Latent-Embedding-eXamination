"""Orchestration for statistical artifact validation and endpoint reconstruction."""

from dataclasses import asdict

from scripts.data_audit import file_sha256
from scripts.statistical_analysis.alignment import validate_dataset_alignment
from scripts.statistical_analysis.config import AnalysisConfig
from scripts.statistical_analysis.endpoints import reconstruct_endpoints
from scripts.statistical_analysis.inference import run_dataset_inference
from scripts.statistical_analysis.loaders import (
    discover_model_runs,
    load_chexpert_sensitivity_artifacts,
    load_dataset_artifacts,
)


REPORT_SCHEMA_VERSION = 3


def build_statistical_input_report(config: AnalysisConfig):
    """Validate all configured inputs and return a JSON-serializable report."""
    discovery = discover_model_runs(config)
    loaded_by_dataset = {dataset: [] for dataset in config.datasets}
    loaded_by_key = {}
    for model in config.models:
        run = discovery.runs.get(model.model_id)
        if run is None:
            continue
        for dataset in config.datasets:
            loaded = load_dataset_artifacts(config, run, dataset)
            loaded_by_dataset[dataset].append(loaded)
            loaded_by_key[(model.model_id, dataset)] = loaded

    alignments = {
        dataset: validate_dataset_alignment(records, config.reference_model_id)
        for dataset, records in loaded_by_dataset.items()
    }
    inference = {
        dataset: run_dataset_inference(records, config)
        for dataset, records in loaded_by_dataset.items()
    }

    sensitivity_by_policy = load_chexpert_sensitivity_artifacts(
        config,
        [discovery.runs[model.model_id] for model in config.models
         if model.model_id in discovery.runs],
        {
            model.model_id: loaded_by_key[(model.model_id, "chexpert")]
            for model in config.models
            if model.model_id in discovery.runs and "chexpert" in config.datasets
        },
    )
    sensitivity_alignment = {
        policy: validate_dataset_alignment(records, config.reference_model_id)
        for policy, records in sensitivity_by_policy.items()
    }
    sensitivity_endpoints = {
        policy: [
            asdict(reconstruct_endpoints(
                loaded,
                adapter=config.classification_adapter,
                require_all_labels_eligible=config.require_all_labels_eligible,
            ))
            for loaded in records
        ]
        for policy, records in sensitivity_by_policy.items()
    }
    sensitivity_artifacts = {
        policy: [
            {
                "model_id": loaded.run.model.model_id,
                "result_json": {
                    "path": str(loaded.result_path),
                    "sha256": loaded.result_sha256,
                },
                "oof": {
                    "path": str(loaded.oof_path),
                    "sha256": loaded.oof_sha256,
                },
                "retrieval": {
                    "path": str(loaded.retrieval_path),
                    "sha256": loaded.retrieval_sha256,
                },
            }
            for loaded in records
        ]
        for policy, records in sensitivity_by_policy.items()
    }

    endpoints = []
    artifacts = []
    for model in config.models:
        if model.model_id not in discovery.runs:
            continue
        for dataset in config.datasets:
            loaded = loaded_by_key[(model.model_id, dataset)]
            row = reconstruct_endpoints(
                loaded,
                adapter=config.classification_adapter,
                require_all_labels_eligible=config.require_all_labels_eligible,
            )
            endpoints.append(asdict(row))
            artifacts.append({
                "model_id": model.model_id,
                "dataset": dataset,
                "result_json": {
                    "path": str(loaded.result_path),
                    "sha256": loaded.result_sha256,
                    "schema_version": loaded.result["result_schema_version"],
                },
                "oof": {
                    "path": str(loaded.oof_path),
                    "sha256": loaded.oof_sha256,
                    "schema_version": int(loaded.oof["schema_version"]),
                },
                "retrieval": {
                    "path": str(loaded.retrieval_path),
                    "sha256": loaded.retrieval_sha256,
                    "schema_version": int(loaded.retrieval["schema_version"]),
                },
            })

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "config": {
            "path": str(config.source_path),
            "sha256": file_sha256(str(config.source_path)),
            "results_dir": str(config.results_dir),
            "classification_endpoint": config.classification_endpoint,
            "classification_adapter": config.classification_adapter,
            "retrieval_endpoint": config.retrieval_endpoint,
            "bootstrap_repetitions": config.bootstrap_repetitions,
            "permutation_repetitions": config.permutation_repetitions,
            "confidence_level": config.confidence_level,
            "inference_seed": config.inference_seed,
            "holm_family": config.holm_family,
        },
        "discovery": {
            "expected_model_ids": list(config.model_ids),
            "loaded_model_ids": [
                model.model_id for model in config.models
                if model.model_id in discovery.runs
            ],
            "missing_model_ids": list(discovery.missing_model_ids),
            "unexpected_model_ids": list(discovery.unexpected_model_ids),
            "runs": {
                model_id: {
                    "run_dir": str(run.run_dir),
                    "config_used_sha256": run.config_used_sha256,
                    "config_used_canonical_sha256": (
                        run.config_used_canonical_sha256
                    ),
                }
                for model_id, run in discovery.runs.items()
            },
        },
        "alignment": alignments,
        "inference": inference,
        "chexpert_sensitivity": {
            "status": "validated" if sensitivity_by_policy else "not_available",
            "policies": list(sensitivity_by_policy),
            "alignment": sensitivity_alignment,
            "endpoints": sensitivity_endpoints,
            "artifacts": sensitivity_artifacts,
            "inference_scope": "descriptive_only",
        },
        "endpoints": endpoints,
        "artifacts": artifacts,
    }
