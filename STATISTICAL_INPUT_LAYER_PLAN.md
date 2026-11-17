# Statistical input layer: implementation plan

## Purpose

This layer turns completed benchmark runs into validated, paired statistical
inputs. It is deliberately additive and read-only: it does not change model
execution, rewrite result folders, or implement bootstrap/permutation inference.

The two primary endpoints are reconstructed from the evidence artifacts:

- classification: pooled out-of-fold macro-F1 for one prespecified adapter;
- retrieval: label-macro mean average precision (mAP) over eligible query-label
  units.

This prevents the statistical analysis from silently using the older fold-mean
classification summary or query-micro retrieval mAP.

## Components

### 1. Configuration contract

Files:

- `analysis_config.schema.json`: declarative JSON Schema for editors and CI;
- `analysis_config.example.json`: final eight-model benchmark example;
- `scripts/statistical_analysis/config.py`: dependency-free strict parser.

The configuration fixes:

- expected model IDs and display names;
- the optional exact run directory for each model;
- required datasets and reference model;
- the classification adapter and primary endpoint names;
- accepted result and artifact schema versions;
- preliminary-run completeness policy;
- artifact SHA-256 verification and retrieval label-coverage policy.

Unknown keys, duplicate models/datasets, unsupported endpoint names, invalid
schema lists, and a reference model outside the model set are rejected. Relative
paths are resolved against the configuration file, except model `run_dir` values,
which are resolved under `results_dir`.

### 2. Run and artifact loaders

File: `scripts/statistical_analysis/loaders.py`

Run discovery reads each immediate child directory's `config_used.json`; folder
names are never treated as model identity. An exact `run_dir` can be supplied to
remove ambiguity. Duplicate runs for an expected model fail closed.

For every model-dataset pair, the loader:

1. reads `<run>/<dataset>.json`;
2. checks the result schema, dataset name, and model identity;
3. resolves artifact paths from JSON metadata and prevents path traversal;
4. verifies the file SHA-256 and declared artifact schema;
5. loads NPZ files with `allow_pickle=False`;
6. invokes the benchmark's existing OOF and retrieval artifact validators;
7. cross-checks artifact metadata, dataset metadata, and the OOF/retrieval sample
   populations.

No result or artifact is modified.

### 3. Endpoint reconstruction

File: `scripts/statistical_analysis/endpoints.py`

Classification is reconstructed from the complete saved OOF `y_true` and
`y_pred` arrays. Retrieval is reconstructed from eligible per-query evidence,
first within each declared label and then equally across labels. The output also
retains per-label mAP and eligible-unit counts for later resampling diagnostics.

When a newer result JSON already stores the same endpoint, it is checked against
the reconstructed value at a strict numerical tolerance. Older schema-5 summaries
are not incorrectly interpreted as these endpoints.

### 4. Alignment validation

File: `scripts/statistical_analysis/alignment.py`

Within each model-dataset pair, classification and retrieval must share the same
dataset, label type, declared classes, ordered sample IDs, and group IDs.

Across models, the following paired inputs must match exactly:

- OOF: dataset, adapter, label type, classes, ordered sample IDs, group IDs,
  outer-fold assignment, and targets;
- retrieval: dataset, label type, similarity protocol, normalization,
  candidate-exclusion rule, classes, K values, ordered sample/group IDs, query
  units, label indices, eligibility, exclusion reasons, and candidate/relevant
  counts;
- result metadata when present: dataset semantics, cohort manifest SHA-256, and
  semantic data-audit fingerprints.

Predictions, probabilities, ranks, and AP values are intentionally not alignment
fields because those are model outcomes.

### 5. Orchestration and report

Files:

- `scripts/statistical_analysis/pipeline.py`;
- `scripts/validate_statistical_inputs.py`.

The CLI produces a deterministic JSON validation report containing discovery
status, missing/unexpected models, alignment references, artifact provenance,
and reconstructed endpoints. It prints to stdout unless `--output` is supplied;
output files are written atomically.

Example:

```bash
python3 scripts/validate_statistical_inputs.py \
  --config analysis_config.example.json \
  --output output/statistical_inputs.json
```

## Failure policy

The layer fails before statistical testing for a missing required dataset,
ambiguous run, unsupported schema, missing/corrupt artifact, hash mismatch,
metadata disagreement, incomplete required label coverage, or any paired-unit
misalignment. Missing models are only reportable when
`completeness.allow_missing_models` is explicitly true.

## Acceptance criteria

- Existing benchmark code and saved runs remain untouched.
- Schema-5 and schema-6 result JSONs can be consumed when their evidence
  artifacts use the accepted schemas.
- The preliminary seven-model run passes all artifact and alignment checks while
  reporting CXR Foundation as missing.
- A changed target, sample order, group, fold, retrieval unit, or artifact hash
  causes validation to fail.
- Reconstructed endpoints agree with the benchmark's producer-side functions.

## Statistical inference layer

Group-aware paired bootstrap confidence intervals, omnibus and pairwise
permutation tests, and Holm multiplicity adjustment are implemented separately in
`scripts/statistical_analysis/inference.py`. Their design and validation contract
are documented in `STATISTICAL_INFERENCE_PLAN.md`. Rank stability and efficiency
summaries remain separate future steps.
