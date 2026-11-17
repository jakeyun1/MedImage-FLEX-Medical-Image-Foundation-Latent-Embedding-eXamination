"""Strict, dependency-free configuration parsing for statistical inputs."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple


CONFIG_SCHEMA_VERSION = 2


class ConfigurationError(ValueError):
    """Raised when a statistical analysis configuration is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    run_dir: Optional[Path]


@dataclass(frozen=True)
class AnalysisConfig:
    source_path: Path
    results_dir: Path
    models: Tuple[ModelSpec, ...]
    datasets: Tuple[str, ...]
    reference_model_id: str
    classification_adapter: str
    classification_endpoint: str
    retrieval_endpoint: str
    require_all_labels_eligible: bool
    bootstrap_repetitions: int
    permutation_repetitions: int
    confidence_level: float
    inference_seed: int
    pairwise_alternative: str
    holm_family: str
    result_schema_versions: Tuple[int, ...]
    oof_artifact_schema_versions: Tuple[int, ...]
    retrieval_artifact_schema_versions: Tuple[int, ...]
    allow_missing_models: bool
    allow_unexpected_models: bool
    verify_artifact_sha256: bool

    @property
    def model_ids(self):
        return tuple(model.model_id for model in self.models)


def _object(value, context):
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be an object.")
    return value


def _exact_keys(value, *, required, optional=(), context):
    value = _object(value, context)
    observed = set(value)
    required = set(required)
    allowed = required | set(optional)
    missing = sorted(required - observed)
    unknown = sorted(observed - allowed)
    if missing:
        raise ConfigurationError(f"{context} is missing required keys: {missing}")
    if unknown:
        raise ConfigurationError(f"{context} has unknown keys: {unknown}")
    return value


def _nonempty_string(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty string.")
    return value.strip()


def _boolean(value, context):
    if not isinstance(value, bool):
        raise ConfigurationError(f"{context} must be boolean.")
    return value


def _integer(value, context, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{context} must be an integer greater than or equal to {minimum}."
        )
    return value


def _probability(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context} must be numeric and lie in (0, 1).")
    value = float(value)
    if not 0 < value < 1:
        raise ConfigurationError(f"{context} must lie in (0, 1).")
    return value


def _unique_strings(value, context):
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{context} must be a non-empty array.")
    strings = tuple(
        _nonempty_string(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(strings)) != len(strings):
        raise ConfigurationError(f"{context} must not contain duplicates.")
    return strings


def _schema_versions(value, context):
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{context} must be a non-empty array.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
        raise ConfigurationError(f"{context} must contain positive integers.")
    if len(set(value)) != len(value):
        raise ConfigurationError(f"{context} must not contain duplicates.")
    return tuple(value)


def load_analysis_config(path):
    """Load and strictly validate one versioned analysis configuration."""
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open(encoding="utf-8") as config_file:
            raw = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read analysis config {source_path}: {exc}") from exc

    required = (
        "schema_version", "results_dir", "models", "datasets",
        "reference_model_id", "classification", "retrieval", "inference",
        "compatibility", "completeness", "integrity",
    )
    raw = _exact_keys(raw, required=required, context="config")
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported config schema_version {raw['schema_version']!r}; "
            f"expected {CONFIG_SCHEMA_VERSION}."
        )

    results_value = Path(_nonempty_string(raw["results_dir"], "results_dir")).expanduser()
    results_dir = (
        results_value.resolve() if results_value.is_absolute()
        else (source_path.parent / results_value).resolve()
    )

    if not isinstance(raw["models"], list) or not raw["models"]:
        raise ConfigurationError("models must be a non-empty array.")
    models = []
    for index, model_raw in enumerate(raw["models"]):
        context = f"models[{index}]"
        model_raw = _exact_keys(
            model_raw,
            required=("model_id", "display_name"),
            optional=("run_dir",),
            context=context,
        )
        run_dir = None
        if "run_dir" in model_raw:
            run_value = Path(_nonempty_string(model_raw["run_dir"], f"{context}.run_dir")).expanduser()
            run_dir = (
                run_value.resolve() if run_value.is_absolute()
                else (results_dir / run_value).resolve()
            )
        models.append(ModelSpec(
            model_id=_nonempty_string(model_raw["model_id"], f"{context}.model_id"),
            display_name=_nonempty_string(
                model_raw["display_name"], f"{context}.display_name"
            ),
            run_dir=run_dir,
        ))
    model_ids = [model.model_id for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ConfigurationError("models contains duplicate model_id values.")

    datasets = _unique_strings(raw["datasets"], "datasets")
    reference_model_id = _nonempty_string(
        raw["reference_model_id"], "reference_model_id"
    )
    if reference_model_id not in model_ids:
        raise ConfigurationError("reference_model_id must appear in models.")

    classification = _exact_keys(
        raw["classification"],
        required=("adapter", "endpoint"),
        context="classification",
    )
    classification_endpoint = _nonempty_string(
        classification["endpoint"], "classification.endpoint"
    )
    if classification_endpoint != "pooled_oof_macro_f1":
        raise ConfigurationError(
            "classification.endpoint must be 'pooled_oof_macro_f1'."
        )

    retrieval = _exact_keys(
        raw["retrieval"],
        required=("endpoint", "require_all_labels_eligible"),
        context="retrieval",
    )
    retrieval_endpoint = _nonempty_string(retrieval["endpoint"], "retrieval.endpoint")
    if retrieval_endpoint != "label_macro_map":
        raise ConfigurationError("retrieval.endpoint must be 'label_macro_map'.")

    inference = _exact_keys(
        raw["inference"],
        required=(
            "bootstrap_repetitions", "permutation_repetitions",
            "confidence_level", "seed", "pairwise_alternative", "holm_family",
        ),
        context="inference",
    )
    pairwise_alternative = _nonempty_string(
        inference["pairwise_alternative"], "inference.pairwise_alternative"
    )
    if pairwise_alternative != "two_sided":
        raise ConfigurationError(
            "inference.pairwise_alternative must be 'two_sided'."
        )
    holm_family = _nonempty_string(
        inference["holm_family"], "inference.holm_family"
    )
    if holm_family != "within_dataset_endpoint":
        raise ConfigurationError(
            "inference.holm_family must be 'within_dataset_endpoint'."
        )

    compatibility = _exact_keys(
        raw["compatibility"],
        required=(
            "result_schema_versions", "oof_artifact_schema_versions",
            "retrieval_artifact_schema_versions",
        ),
        context="compatibility",
    )
    completeness = _exact_keys(
        raw["completeness"],
        required=("allow_missing_models", "allow_unexpected_models"),
        context="completeness",
    )
    integrity = _exact_keys(
        raw["integrity"],
        required=("verify_artifact_sha256",),
        context="integrity",
    )

    return AnalysisConfig(
        source_path=source_path,
        results_dir=results_dir,
        models=tuple(models),
        datasets=datasets,
        reference_model_id=reference_model_id,
        classification_adapter=_nonempty_string(
            classification["adapter"], "classification.adapter"
        ),
        classification_endpoint=classification_endpoint,
        retrieval_endpoint=retrieval_endpoint,
        require_all_labels_eligible=_boolean(
            retrieval["require_all_labels_eligible"],
            "retrieval.require_all_labels_eligible",
        ),
        bootstrap_repetitions=_integer(
            inference["bootstrap_repetitions"],
            "inference.bootstrap_repetitions",
            minimum=1,
        ),
        permutation_repetitions=_integer(
            inference["permutation_repetitions"],
            "inference.permutation_repetitions",
            minimum=1,
        ),
        confidence_level=_probability(
            inference["confidence_level"], "inference.confidence_level"
        ),
        inference_seed=_integer(inference["seed"], "inference.seed", minimum=0),
        pairwise_alternative=pairwise_alternative,
        holm_family=holm_family,
        result_schema_versions=_schema_versions(
            compatibility["result_schema_versions"],
            "compatibility.result_schema_versions",
        ),
        oof_artifact_schema_versions=_schema_versions(
            compatibility["oof_artifact_schema_versions"],
            "compatibility.oof_artifact_schema_versions",
        ),
        retrieval_artifact_schema_versions=_schema_versions(
            compatibility["retrieval_artifact_schema_versions"],
            "compatibility.retrieval_artifact_schema_versions",
        ),
        allow_missing_models=_boolean(
            completeness["allow_missing_models"],
            "completeness.allow_missing_models",
        ),
        allow_unexpected_models=_boolean(
            completeness["allow_unexpected_models"],
            "completeness.allow_unexpected_models",
        ),
        verify_artifact_sha256=_boolean(
            integrity["verify_artifact_sha256"],
            "integrity.verify_artifact_sha256",
        ),
    )
