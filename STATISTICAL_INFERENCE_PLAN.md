# Statistical inference layer

## Scope

This layer consumes only artifacts that have passed schema, integrity, endpoint,
and cross-model alignment validation. It adds:

- paired group-cluster percentile bootstrap confidence intervals;
- one omnibus randomization test per dataset and endpoint;
- two-sided pairwise randomization tests;
- Holm-adjusted pairwise p-values.

The benchmark execution code and result folders remain unchanged.

## Paired group bootstrap

For each dataset, groups—not individual images or query-label rows—are sampled
with replacement. The same group multiplicities are applied to every model in a
replicate, preserving pairing and within-patient/lesion dependence.

Classification uses per-group, per-model TP/FP/FN sufficient statistics and
recomputes pooled macro-F1. Retrieval uses per-group, per-model, per-label AP sums
and eligible-unit counts and recomputes label-macro mAP. Model intervals and
model-to-model difference intervals therefore use the same paired replicates.

Classification resamples missing positive support for any declared class/finding,
and retrieval resamples missing any declared eligible label, are rejected and
replaced. The report records the number rejected. This avoids silently changing a
macro endpoint's label set or converting an undefined label score to zero.

## Permutation tests

The omnibus null is that model labels are exchangeable within each group. Each
randomization independently permutes all model assignments inside every group,
then reconstructs the endpoint. The statistic is the sum of squared deviations of
model scores from their mean.

For each pair, the two models' outcomes are independently swapped within groups.
The two-sided p-value compares the absolute randomized difference with the
absolute observed difference. Pairwise tests share a family-level randomization
stream for efficiency; every marginal pairwise randomization remains valid.

Monte Carlo p-values use `(extreme + 1) / (repetitions + 1)`, so zero p-values are
impossible and the attainable resolution is reported.

## Holm adjustment

Holm correction is applied to every pairwise raw p-value within one prespecified
family: one dataset × one endpoint. With seven models the family contains 21
tests; with all eight it contains 28. The omnibus test is reported separately and
is not inserted into the pairwise family.

Sorted raw p-values are multiplied by their remaining family size, followed by a
cumulative maximum and truncation at one. Both raw and adjusted p-values are
retained. Rejection uses `alpha = 1 - confidence_level`.

## Efficiency and reproducibility

Inference operates on compressed per-group sufficient statistics rather than
re-reading sample rows for every replicate. Work is processed in memory-bounded
chunks. Random streams are derived from the configured seed plus stable dataset,
endpoint, and analysis labels, so results are deterministic and unaffected by
Python hash randomization.

The example configuration uses 2,000 bootstrap and 10,000 permutation replicates.

## Fail-closed validation

Inference stops if:

- fewer than two aligned models are available;
- model IDs are duplicated or datasets are mixed;
- sufficient statistics do not reproduce the saved-artifact endpoints to `1e-12`;
- a requested bootstrap interval is non-finite;
- retrieval cannot produce the requested number of defined resamples;
- p-values are empty, non-finite, or outside `[0, 1]`;
- a permutation difference disagrees with the reconstructed point estimates.
