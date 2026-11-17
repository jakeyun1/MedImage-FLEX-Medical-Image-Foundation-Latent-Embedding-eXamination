#!/usr/bin/env python3
"""Validate benchmark artifacts and reconstruct prespecified paper endpoints."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

# Permit direct execution as documented while keeping package imports absolute.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.statistical_analysis import (
    build_statistical_input_report,
    load_analysis_config,
)
from scripts.statistical_analysis.config import ConfigurationError
from scripts.statistical_analysis.loaders import StatisticalInputError


def _write_json_atomic(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Analysis configuration JSON")
    parser.add_argument("--output", help="Optional path for the validation report JSON")
    arguments = parser.parse_args(argv)
    try:
        config = load_analysis_config(arguments.config)
        report = build_statistical_input_report(config)
    except (ConfigurationError, StatisticalInputError, KeyError, TypeError) as exc:
        print(f"Statistical input validation failed: {exc}", file=sys.stderr)
        return 2

    if arguments.output:
        _write_json_atomic(arguments.output, report)
        print(f"Validated statistical inputs: {Path(arguments.output).resolve()}")
    else:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
