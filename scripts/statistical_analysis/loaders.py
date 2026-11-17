"""Read-only discovery and loading of benchmark evidence artifacts."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from scripts.data_audit import file_sha256
from scripts.oof import validate_oof_artifact
from scripts.provenance import canonical_json_sha256
from scripts.retrieval_artifacts import validate_retrieval_artifact
from scripts.statistical_analysis.config import AnalysisConfig, ModelSpec


class StatisticalInputError(ValueError):
    """Raised when statistical inputs are missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class ModelRun:
    model: ModelSpec
    run_dir: Path
    config_used: Mapping
    config_used_sha256: str
    config_used_canonical_sha256: str


@dataclass(frozen=True)
class RunDiscovery:
    runs: Mapping[str, ModelRun]
    missing_model_ids: Tuple[str, ...]
    unexpected_model_ids: Tuple[str, ...]


@dataclass(frozen=True)
class LoadedDataset:
    run: ModelRun
    dataset: str
    result_path: Path
    result: Mapping
    result_sha256: str
    oof_path: Path
    oof: Mapping[str, np.ndarray]
    oof_sha256: str
    retrieval_path: Path
    retrieval: Mapping[str, np.ndarray]
    retrieval_sha256: str


def _read_json(path, context):
    try:
        with Path(path).open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise StatisticalInputError(f"Could not read {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StatisticalInputError(f"{context} must contain a JSON object: {path}")
    return value


def _model_id_from_config(config_used, path):
    model_id = config_used.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise StatisticalInputError(f"config_used.json has no valid model_id: {path}")
    return model_id.strip()


def discover_model_runs(config: AnalysisConfig):
    """Resolve exactly one run per expected model, using config content as identity."""
    if not config.results_dir.is_dir():
        raise StatisticalInputError(
            f"Results directory does not exist: {config.results_dir}"
        )

    candidates: Dict[str, list] = {}
    for child in sorted(config.results_dir.iterdir()):
        config_path = child / "config_used.json"
        if not child.is_dir() or not config_path.is_file():
            continue
        config_used = _read_json(config_path, "run configuration")
        model_id = _model_id_from_config(config_used, config_path)
        candidates.setdefault(model_id, []).append((child.resolve(), config_used))

    expected_ids = set(config.model_ids)
    unexpected = tuple(sorted(set(candidates) - expected_ids))
    if unexpected and not config.allow_unexpected_models:
        raise StatisticalInputError(
            f"Unexpected model runs found under results_dir: {list(unexpected)}"
        )

    runs = {}
    missing = []
    for model in config.models:
        if model.run_dir is not None:
            run_dir = model.run_dir
            config_path = run_dir / "config_used.json"
            if not config_path.is_file():
                missing.append(model.model_id)
                continue
            config_used = _read_json(config_path, "run configuration")
            observed_id = _model_id_from_config(config_used, config_path)
            if observed_id != model.model_id:
                raise StatisticalInputError(
                    f"Configured run_dir for {model.model_id!r} identifies model "
                    f"{observed_id!r}: {run_dir}"
                )
        else:
            matches = candidates.get(model.model_id, [])
            if not matches:
                missing.append(model.model_id)
                continue
            if len(matches) != 1:
                paths = [str(match[0]) for match in matches]
                raise StatisticalInputError(
                    f"Multiple runs found for {model.model_id!r}; set models[].run_dir "
                    f"to select one: {paths}"
                )
            run_dir, config_used = matches[0]
            config_path = run_dir / "config_used.json"

        runs[model.model_id] = ModelRun(
            model=model,
            run_dir=run_dir.resolve(),
            config_used=config_used,
            config_used_sha256=file_sha256(str(config_path)),
            config_used_canonical_sha256=canonical_json_sha256(config_used),
        )

    missing = tuple(missing)
    if missing and not config.allow_missing_models:
        raise StatisticalInputError(f"Required model runs are missing: {list(missing)}")
    if not runs:
        raise StatisticalInputError("No configured model runs were found.")
    return RunDiscovery(
        runs=runs,
        missing_model_ids=missing,
        unexpected_model_ids=unexpected,
    )


def _load_npz(path, context):
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError) as exc:
        raise StatisticalInputError(f"Could not load {context} {path}: {exc}") from exc


def _scalar(artifact, key, context):
    if key not in artifact:
        raise StatisticalInputError(f"{context} is missing scalar field {key!r}.")
    value = np.asarray(artifact[key])
    if value.ndim != 0:
        raise StatisticalInputError(f"{context} field {key!r} must be scalar.")
    return value.item()


def _artifact_path(run_dir, metadata, context):
    if not isinstance(metadata, dict):
        raise StatisticalInputError(f"{context} metadata must be an object.")
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise StatisticalInputError(f"{context} metadata has no valid path.")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise StatisticalInputError(f"{context} path must be relative to its run directory.")
    run_root = run_dir.resolve()
    resolved = (run_root / relative).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise StatisticalInputError(f"{context} path escapes its run directory: {raw_path}") from exc
    if not resolved.is_file():
        raise StatisticalInputError(f"{context} file does not exist: {resolved}")
    return resolved


def _verify_artifact_metadata(path, artifact, metadata, allowed_schemas, *, verify_sha, context):
    artifact_schema = int(_scalar(artifact, "schema_version", context))
    if artifact_schema not in allowed_schemas:
        raise StatisticalInputError(
            f"{context} schema {artifact_schema} is not accepted; "
            f"allowed: {list(allowed_schemas)}"
        )
    declared_schema = metadata.get("schema_version")
    if declared_schema != artifact_schema:
        raise StatisticalInputError(
            f"{context} metadata schema {declared_schema!r} disagrees with "
            f"artifact schema {artifact_schema}."
        )
    observed_sha = file_sha256(str(path))
    declared_sha = metadata.get("sha256")
    if verify_sha and declared_sha != observed_sha:
        raise StatisticalInputError(
            f"{context} SHA-256 mismatch: declared {declared_sha!r}, observed {observed_sha}."
        )
    if "size_bytes" in metadata and int(metadata["size_bytes"]) != path.stat().st_size:
        raise StatisticalInputError(f"{context} byte size disagrees with its metadata.")
    return observed_sha


def _require_equal(observed, expected, message):
    if observed != expected:
        raise StatisticalInputError(f"{message}: observed {observed!r}, expected {expected!r}.")


def _validate_dataset_contract(loaded: LoadedDataset):
    result = loaded.result
    info = result.get("dataset_info")
    if not isinstance(info, dict):
        raise StatisticalInputError(f"dataset_info is missing for {loaded.dataset}.")

    oof = loaded.oof
    retrieval = loaded.retrieval
    oof_context = f"{loaded.run.model.model_id}/{loaded.dataset} OOF"
    retrieval_context = f"{loaded.run.model.model_id}/{loaded.dataset} retrieval"
    for artifact, context in ((oof, oof_context), (retrieval, retrieval_context)):
        _require_equal(
            str(_scalar(artifact, "dataset_name", context)),
            loaded.dataset,
            f"{context} dataset name mismatch",
        )

    label_type = str(_scalar(oof, "label_type", oof_context))
    _require_equal(
        str(_scalar(retrieval, "label_type", retrieval_context)),
        label_type,
        f"{retrieval_context} label type mismatch",
    )
    _require_equal(info.get("label_type"), label_type, "dataset_info label type mismatch")

    oof_classes = np.asarray(oof["classes"]).astype(str)
    retrieval_classes = np.asarray(retrieval["classes"]).astype(str)
    if not np.array_equal(oof_classes, retrieval_classes):
        raise StatisticalInputError(
            f"Classification and retrieval classes differ for "
            f"{loaded.run.model.model_id}/{loaded.dataset}."
        )
    if list(info.get("classes", [])) != oof_classes.tolist():
        raise StatisticalInputError(
            f"dataset_info classes disagree with artifacts for "
            f"{loaded.run.model.model_id}/{loaded.dataset}."
        )

    for field in ("sample_ids", "group_ids"):
        if not np.array_equal(np.asarray(oof[field]), np.asarray(retrieval[field])):
            raise StatisticalInputError(
                f"OOF and retrieval artifacts differ on {field!r} for "
                f"{loaded.run.model.model_id}/{loaded.dataset}."
            )
    query_indices = np.asarray(retrieval["unit_query_indices"], dtype=np.int64)
    label_indices = np.asarray(retrieval["unit_label_indices"], dtype=np.int64)
    y_true = np.asarray(oof["y_true"], dtype=np.int64)
    if label_type == "multiclass":
        expected_pairs = np.column_stack((np.arange(len(y_true)), y_true))
    else:
        expected_pairs = np.argwhere(y_true == 1)
    observed_pairs = np.column_stack((query_indices, label_indices))
    if not np.array_equal(observed_pairs, expected_pairs):
        raise StatisticalInputError(
            f"Retrieval query-label units disagree with OOF targets for "
            f"{loaded.run.model.model_id}/{loaded.dataset}."
        )
    n_samples = len(np.asarray(oof["sample_ids"]))
    _require_equal(info.get("n_samples"), n_samples, "dataset_info sample count mismatch")

    reproducibility = result.get("reproducibility", {})
    if "n_samples" in reproducibility:
        _require_equal(
            reproducibility["n_samples"], n_samples,
            "reproducibility sample count mismatch",
        )
    if "n_groups" in reproducibility:
        _require_equal(
            reproducibility["n_groups"],
            len(np.unique(np.asarray(oof["group_ids"]).astype(str))),
            "reproducibility group count mismatch",
        )

    ordered_hash = str(_scalar(oof, "ordered_sample_ids_sha256", oof_context))
    _require_equal(
        str(_scalar(retrieval, "ordered_sample_ids_sha256", retrieval_context)),
        ordered_hash,
        "OOF/retrieval ordered sample-ID hash mismatch",
    )


def load_dataset_artifacts(config: AnalysisConfig, run: ModelRun, dataset: str,
                           result_path=None):
    """Load, authenticate, validate, and cross-check one model-dataset pair."""
    result_path = (
        run.run_dir / f"{dataset}.json"
        if result_path is None else Path(result_path)
    )
    if not result_path.is_file():
        raise StatisticalInputError(
            f"Required dataset result is missing for {run.model.model_id}: {result_path}"
        )
    result = _read_json(result_path, "dataset result")
    result_schema = result.get("result_schema_version")
    if result_schema not in config.result_schema_versions:
        raise StatisticalInputError(
            f"Unsupported result schema {result_schema!r} for "
            f"{run.model.model_id}/{dataset}; allowed: {list(config.result_schema_versions)}"
        )
    info = result.get("dataset_info", {})
    _require_equal(info.get("dataset_name"), dataset, "dataset result name mismatch")
    embedded_model_id = result.get("embedding_info", {}).get("model_id")
    if embedded_model_id is not None:
        _require_equal(
            embedded_model_id, run.model.model_id,
            "dataset result model identity mismatch",
        )
    recorded_config_sha = result.get("run_provenance", {}).get(
        "config_canonical_sha256"
    )
    if recorded_config_sha is not None:
        _require_equal(
            recorded_config_sha,
            run.config_used_canonical_sha256,
            "dataset result/config_used.json provenance mismatch",
        )

    oof_block = result.get("oof_predictions")
    if not isinstance(oof_block, dict) or oof_block.get("enabled") is not True:
        raise StatisticalInputError(
            f"OOF evidence is not enabled for {run.model.model_id}/{dataset}."
        )
    artifacts = oof_block.get("artifacts")
    if not isinstance(artifacts, dict) or config.classification_adapter not in artifacts:
        raise StatisticalInputError(
            f"OOF adapter {config.classification_adapter!r} is missing for "
            f"{run.model.model_id}/{dataset}."
        )
    oof_metadata = artifacts[config.classification_adapter]
    oof_path = _artifact_path(
        run.run_dir, oof_metadata,
        f"{run.model.model_id}/{dataset} OOF artifact",
    )
    oof = _load_npz(oof_path, "OOF artifact")
    if "schema_version" in oof_block:
        _require_equal(
            oof_block["schema_version"], oof_metadata.get("schema_version"),
            "OOF block/artifact metadata schema mismatch",
        )
    oof_sha = _verify_artifact_metadata(
        oof_path, oof, oof_metadata, config.oof_artifact_schema_versions,
        verify_sha=config.verify_artifact_sha256,
        context=f"{run.model.model_id}/{dataset} OOF artifact",
    )
    try:
        validate_oof_artifact(oof)
    except ValueError as exc:
        raise StatisticalInputError(
            f"Invalid OOF artifact for {run.model.model_id}/{dataset}: {exc}"
        ) from exc
    _require_equal(
        str(_scalar(oof, "adapter", f"{run.model.model_id}/{dataset} OOF")),
        config.classification_adapter,
        "OOF artifact adapter mismatch",
    )
    oof_sample_ids = np.asarray(oof["sample_ids"])
    oof_group_ids = np.asarray(oof["group_ids"])
    for key, expected in (
        ("n_samples", len(oof_sample_ids)),
        ("n_groups", len(np.unique(oof_group_ids.astype(str)))),
        ("n_outputs", len(np.asarray(oof["classes"]))),
        ("ordered_sample_ids_sha256", str(_scalar(
            oof, "ordered_sample_ids_sha256", "OOF artifact"
        ))),
    ):
        if key in oof_metadata:
            _require_equal(oof_metadata[key], expected, f"OOF metadata {key} mismatch")

    retrieval_block = result.get("retrieval_queries")
    if not isinstance(retrieval_block, dict) or retrieval_block.get("enabled") is not True:
        raise StatisticalInputError(
            f"Retrieval evidence is not enabled for {run.model.model_id}/{dataset}."
        )
    retrieval_metadata = retrieval_block.get("artifact")
    retrieval_path = _artifact_path(
        run.run_dir, retrieval_metadata,
        f"{run.model.model_id}/{dataset} retrieval artifact",
    )
    retrieval = _load_npz(retrieval_path, "retrieval artifact")
    if "schema_version" in retrieval_block:
        _require_equal(
            retrieval_block["schema_version"],
            retrieval_metadata.get("schema_version"),
            "Retrieval block/artifact metadata schema mismatch",
        )
    retrieval_sha = _verify_artifact_metadata(
        retrieval_path, retrieval, retrieval_metadata,
        config.retrieval_artifact_schema_versions,
        verify_sha=config.verify_artifact_sha256,
        context=f"{run.model.model_id}/{dataset} retrieval artifact",
    )
    try:
        validate_retrieval_artifact(retrieval)
    except ValueError as exc:
        raise StatisticalInputError(
            f"Invalid retrieval artifact for {run.model.model_id}/{dataset}: {exc}"
        ) from exc
    retrieval_eligible = np.asarray(retrieval["eligible"], dtype=bool)
    retrieval_sample_ids = np.asarray(retrieval["sample_ids"])
    for key, expected in (
        ("n_samples", len(retrieval_sample_ids)),
        ("n_units", len(retrieval_eligible)),
        ("n_eligible_units", int(np.sum(retrieval_eligible))),
        ("n_excluded_units", int(np.sum(~retrieval_eligible))),
        ("ordered_sample_ids_sha256", str(_scalar(
            retrieval, "ordered_sample_ids_sha256", "retrieval artifact"
        ))),
    ):
        if key in retrieval_metadata:
            _require_equal(
                retrieval_metadata[key], expected,
                f"Retrieval metadata {key} mismatch",
            )

    loaded = LoadedDataset(
        run=run,
        dataset=dataset,
        result_path=result_path.resolve(),
        result=result,
        result_sha256=file_sha256(str(result_path)),
        oof_path=oof_path,
        oof=oof,
        oof_sha256=oof_sha,
        retrieval_path=retrieval_path,
        retrieval=retrieval,
        retrieval_sha256=retrieval_sha,
    )
    _validate_dataset_contract(loaded)
    return loaded


CHEXPERT_SENSITIVITY_POLICIES = ("u_zeros", "u_ones")


def load_chexpert_sensitivity_artifacts(config, runs, primary_by_model):
    """Auto-discover and strictly validate matched CheXpert sensitivities."""
    if "chexpert" not in config.datasets:
        return {}
    runs = list(runs)
    paths = {
        (run.model.model_id, policy):
            run.run_dir / "chexpert_sensitivity" / f"{policy}.json"
        for run in runs
        for policy in CHEXPERT_SENSITIVITY_POLICIES
    }
    existing = {key for key, path in paths.items() if path.is_file()}
    if not existing:
        return {}
    missing = sorted(set(paths) - existing)
    if missing:
        raise StatisticalInputError(
            "CheXpert sensitivity artifacts are incomplete; missing "
            f"{missing[:10]}."
        )

    loaded_by_policy = {policy: [] for policy in CHEXPERT_SENSITIVITY_POLICIES}
    embedding_fields = (
        "model_id", "embedding_dim", "normalized",
        "n_source_image_embeddings", "n_evaluation_embeddings",
        "n_evaluated_embeddings", "ordered_source_image_ids_sha256",
        "ordered_evaluation_ids_sha256", "evaluation_unit",
        "preprocessing_sha256",
    )
    for run in runs:
        model_id = run.model.model_id
        primary = primary_by_model[model_id]
        primary_policy = primary.result.get("data_audit", {}).get(
            "dataset_specific", {}
        ).get("uncertainty_policy")
        for policy in CHEXPERT_SENSITIVITY_POLICIES:
            loaded = load_dataset_artifacts(
                config, run, "chexpert", result_path=paths[(model_id, policy)]
            )
            block = loaded.result.get("sensitivity_analysis")
            if not isinstance(block, dict):
                raise StatisticalInputError(
                    f"Sensitivity metadata is missing for {model_id}/{policy}."
                )
            _require_equal(
                block.get("uncertainty_policy"), policy,
                f"Sensitivity policy mismatch for {model_id}",
            )
            if primary_policy is not None:
                _require_equal(
                    block.get("primary_uncertainty_policy"), primary_policy,
                    f"Sensitivity primary policy mismatch for {model_id}/{policy}",
                )
            for flag in (
                "matched_cohort", "matched_patient_groups",
                "matched_outer_folds", "reused_embeddings",
            ):
                _require_equal(
                    block.get(flag), True,
                    f"Sensitivity flag {flag} mismatch for {model_id}/{policy}",
                )
            fingerprint = loaded.result.get("data_audit", {}).get(
                "sensitivity_evaluation", {}
            ).get("fingerprint")
            _require_equal(
                block.get("label_fingerprint"), fingerprint,
                f"Sensitivity label fingerprint mismatch for {model_id}/{policy}",
            )
            for field in embedding_fields:
                _require_equal(
                    loaded.result.get("embedding_info", {}).get(field),
                    primary.result.get("embedding_info", {}).get(field),
                    f"Sensitivity embedding field {field} mismatch for {model_id}/{policy}",
                )
            _require_equal(
                loaded.result.get("reproducibility"),
                primary.result.get("reproducibility"),
                f"Sensitivity reproducibility mismatch for {model_id}/{policy}",
            )
            for field in ("sample_ids", "group_ids", "outer_folds"):
                if not np.array_equal(
                    np.asarray(loaded.oof[field]), np.asarray(primary.oof[field])
                ):
                    raise StatisticalInputError(
                        f"Sensitivity {policy} changed {field} for {model_id}."
                    )
            loaded_by_policy[policy].append(loaded)
    return loaded_by_policy
