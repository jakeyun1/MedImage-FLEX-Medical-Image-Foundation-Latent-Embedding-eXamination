"""Stable, lightweight provenance metadata for benchmark result files."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import platform
import subprocess
import sys


PACKAGE_DISTRIBUTIONS = {
    "kagglehub": ("kagglehub",),
    "numpy": ("numpy",),
    "optuna": ("optuna",),
    "pandas": ("pandas",),
    "pillow": ("pillow",),
    "scikit_learn": ("scikit-learn",),
    "tensorflow": ("tensorflow", "tensorflow-cpu"),
    "torch": ("torch",),
    "torchvision": ("torchvision",),
    "transformers": ("transformers",),
}


def _json_value(value):
    """Convert common scalar and container values to stable JSON primitives."""
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fingerprint values must not contain NaN or infinity.")
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def canonical_json_sha256(value):
    """Hash one value after deterministic JSON canonicalization."""
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_dataset_fingerprint(sample_ids, group_ids, labels):
    """Describe retained IDs and processed semantics without reading image bytes."""
    if not (len(sample_ids) == len(group_ids) == len(labels)):
        raise ValueError("Dataset fingerprint inputs must have equal lengths.")

    records = [
        {
            "sample_id": str(sample_id),
            "group_id": str(group_id),
            "label": _json_value(label),
        }
        for sample_id, group_id, label in zip(sample_ids, group_ids, labels)
    ]
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Dataset fingerprint sample IDs must be unique.")
    records.sort(key=lambda record: record["sample_id"])

    return {
        "schema_version": 1,
        "canonicalization": "sample_id_sorted_json_v1",
        "retained_sample_ids_sha256": canonical_json_sha256(
            [record["sample_id"] for record in records]
        ),
        "retained_sample_group_labels_sha256": canonical_json_sha256(records),
        "image_bytes_hashed": False,
    }


def _distribution_version(candidates):
    for distribution in candidates:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def _git_provenance(repository_path):
    result = {
        "commit": None,
        "branch": None,
        "tracked_changes": None,
        "untracked_files": None,
    }
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
    }
    outputs = {}
    try:
        for key, command in commands.items():
            completed = subprocess.run(
                command,
                cwd=repository_path,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs[key] = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return result

    result["commit"] = outputs["commit"] or None
    result["branch"] = outputs["branch"] or None
    status_lines = outputs["status"].splitlines()
    result["tracked_changes"] = any(
        not line.startswith("??") for line in status_lines
    )
    result["untracked_files"] = any(
        line.startswith("??") for line in status_lines
    )
    return result


def build_run_provenance(config, repository_path):
    """Return compact code, configuration, and runtime metadata for one run."""
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_canonical_sha256": canonical_json_sha256(config),
        "git": _git_provenance(repository_path),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "package_versions": {
                name: _distribution_version(distributions)
                for name, distributions in PACKAGE_DISTRIBUTIONS.items()
            },
        },
    }
