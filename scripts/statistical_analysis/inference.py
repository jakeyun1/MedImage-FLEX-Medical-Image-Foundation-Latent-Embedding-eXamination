"""Efficient group-paired bootstrap and randomization inference."""

from itertools import combinations
import hashlib
import json

import numpy as np

from scripts.oof import pooled_macro_f1_from_oof
from scripts.retrieval_artifacts import summary_from_retrieval_artifact
from scripts.statistical_analysis.loaders import LoadedDataset, StatisticalInputError


NUMERICAL_TOLERANCE = 1e-12
TARGET_WORKING_BYTES = 64 * 1024 * 1024


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise StatisticalInputError("Holm adjustment requires a non-empty p-value vector.")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise StatisticalInputError("Holm p-values must be finite and lie in [0, 1].")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    multipliers = len(values) - np.arange(len(values))
    adjusted_ordered = np.minimum(
        1.0, np.maximum.accumulate(ordered * multipliers)
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def _rng(seed, *tokens):
    payload = json.dumps(tokens, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256(payload).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


def _working_chunk_size(n_groups, n_models=1, tail_size=1, itemsize=8, maximum=256):
    bytes_per_row = max(1, n_groups * n_models * tail_size * itemsize)
    return max(1, min(maximum, TARGET_WORKING_BYTES // bytes_per_row))


def _f1_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    true_positive = counts[..., 0]
    denominator = 2 * true_positive + counts[..., 1] + counts[..., 2]
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return np.mean(f1, axis=-1)


def _map_from_sums(sums, counts):
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise StatisticalInputError(
            "Label-macro mAP is undefined because a resample contains no eligible "
            "unit for at least one declared label."
        )
    denominator = counts if sums.ndim == counts.ndim else np.expand_dims(counts, axis=-2)
    label_means = sums / denominator
    return np.mean(label_means, axis=-1)


def _classification_contributions(records):
    reference = records[0]
    group_ids = np.asarray(reference.oof["group_ids"]).astype(str)
    groups, group_index = np.unique(group_ids, return_inverse=True)
    y_true = np.asarray(reference.oof["y_true"], dtype=np.int64)
    classes = np.asarray(reference.oof["classes"])
    label_type = str(np.asarray(reference.oof["label_type"]).item())
    values = np.zeros(
        (len(groups), len(records), len(classes), 3), dtype=np.int64
    )

    for model_index, record in enumerate(records):
        y_pred = np.asarray(record.oof["y_pred"], dtype=np.int64)
        for label_index in range(len(classes)):
            if label_type == "multiclass":
                truth = y_true == label_index
                prediction = y_pred == label_index
            else:
                truth = y_true[:, label_index] == 1
                prediction = y_pred[:, label_index] == 1
            np.add.at(
                values[:, model_index, label_index, 0],
                group_index,
                truth & prediction,
            )
            np.add.at(
                values[:, model_index, label_index, 1],
                group_index,
                ~truth & prediction,
            )
            np.add.at(
                values[:, model_index, label_index, 2],
                group_index,
                truth & ~prediction,
            )

    point = _f1_from_counts(np.sum(values, axis=0))
    class_support = np.sum(values[:, 0, :, 0] + values[:, 0, :, 2], axis=0)
    if np.any(class_support == 0):
        missing = np.asarray(classes).astype(str)[class_support == 0].tolist()
        raise StatisticalInputError(
            f"Classification labels have no positive support: {missing}"
        )
    expected = np.asarray([
        pooled_macro_f1_from_oof(record.oof) for record in records
    ])
    if not np.allclose(point, expected, rtol=0, atol=NUMERICAL_TOLERANCE):
        raise StatisticalInputError(
            "Classification sufficient statistics do not reconstruct pooled OOF macro-F1."
        )
    return groups, values, point


def _retrieval_contributions(records):
    reference = records[0]
    sample_group_ids = np.asarray(reference.retrieval["group_ids"]).astype(str)
    groups = np.unique(sample_group_ids)
    group_lookup = {group: index for index, group in enumerate(groups.tolist())}
    unit_groups = np.asarray(reference.retrieval["unit_group_ids"]).astype(str)
    unit_group_index = np.asarray([group_lookup[group] for group in unit_groups])
    label_indices = np.asarray(
        reference.retrieval["unit_label_indices"], dtype=np.int64
    )
    eligible = np.asarray(reference.retrieval["eligible"], dtype=bool)
    n_labels = len(np.asarray(reference.retrieval["classes"]))
    sums = np.zeros((len(groups), len(records), n_labels), dtype=np.float64)
    counts = np.zeros((len(groups), n_labels), dtype=np.int64)
    np.add.at(
        counts,
        (unit_group_index[eligible], label_indices[eligible]),
        1,
    )
    for model_index, record in enumerate(records):
        average_precision = np.asarray(
            record.retrieval["average_precision"], dtype=np.float64
        )
        np.add.at(
            sums[:, model_index, :],
            (unit_group_index[eligible], label_indices[eligible]),
            average_precision[eligible],
        )

    total_counts = np.sum(counts, axis=0)
    point = _map_from_sums(np.sum(sums, axis=0), total_counts)
    expected = np.asarray([
        summary_from_retrieval_artifact(record.retrieval)["label_macro_map"]
        for record in records
    ])
    if not np.allclose(point, expected, rtol=0, atol=NUMERICAL_TOLERANCE):
        raise StatisticalInputError(
            "Retrieval sufficient statistics do not reconstruct label-macro mAP."
        )
    return groups, sums, counts, point


def _bootstrap_classification(values, repetitions, rng):
    n_groups = values.shape[0]
    chunk_size = _working_chunk_size(n_groups, maximum=256)
    result = np.empty((repetitions, values.shape[1]), dtype=np.float64)
    probabilities = np.full(n_groups, 1 / n_groups)
    completed = 0
    attempted = 0
    maximum_attempts = max(1000, repetitions * 100)
    while completed < repetitions and attempted < maximum_attempts:
        requested = min(chunk_size, repetitions - completed)
        multiplicities = rng.multinomial(n_groups, probabilities, size=requested)
        attempted += requested
        aggregate = np.einsum("bg,gmlc->bmlc", multiplicities, values, optimize=True)
        truth_support = aggregate[:, 0, :, 0] + aggregate[:, 0, :, 2]
        valid = np.all(truth_support > 0, axis=1)
        if not np.any(valid):
            continue
        scores = _f1_from_counts(aggregate[valid])
        accepted = min(len(scores), repetitions - completed)
        result[completed:completed + accepted] = scores[:accepted]
        completed += accepted
    if completed != repetitions:
        raise StatisticalInputError(
            "Could not obtain the requested number of classification bootstrap "
            "replicates containing every declared label."
        )
    return result, attempted - repetitions


def _bootstrap_retrieval(sums, counts, repetitions, rng):
    n_groups = sums.shape[0]
    chunk_size = _working_chunk_size(n_groups, maximum=256)
    probabilities = np.full(n_groups, 1 / n_groups)
    result = np.empty((repetitions, sums.shape[1]), dtype=np.float64)
    completed = 0
    attempted = 0
    maximum_attempts = max(1000, repetitions * 100)
    while completed < repetitions and attempted < maximum_attempts:
        requested = min(chunk_size, repetitions - completed)
        multiplicities = rng.multinomial(n_groups, probabilities, size=requested)
        attempted += requested
        label_counts = multiplicities @ counts
        valid = np.all(label_counts > 0, axis=1)
        if not np.any(valid):
            continue
        multiplicities = multiplicities[valid]
        label_counts = label_counts[valid]
        aggregate = np.einsum(
            "bg,gml->bml", multiplicities, sums, optimize=True
        )
        scores = _map_from_sums(aggregate, label_counts)
        accepted = min(len(scores), repetitions - completed)
        result[completed:completed + accepted] = scores[:accepted]
        completed += accepted
    if completed != repetitions:
        raise StatisticalInputError(
            "Could not obtain the requested number of defined retrieval bootstrap "
            "replicates. Check rare-label group support."
        )
    return result, attempted - repetitions


def _pairwise_family_permutation(values, score_function, repetitions, rng):
    """Test every model pair using one shared, valid family randomization stream."""
    n_groups = values.shape[0]
    pairs = list(combinations(range(values.shape[1]), 2))
    first = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    n_pairs = len(pairs)
    tail_size = int(np.prod(values.shape[2:]))
    chunk_size = _working_chunk_size(
        n_groups,
        n_models=n_pairs,
        tail_size=tail_size,
        itemsize=values.dtype.itemsize,
        maximum=256,
    )
    aggregate_a = np.sum(values[:, first], axis=0)
    aggregate_b = np.sum(values[:, second], axis=0)
    observed = np.asarray(
        score_function(aggregate_a) - score_function(aggregate_b),
        dtype=np.float64,
    )
    difference = values[:, second] - values[:, first]
    extreme = np.zeros(n_pairs, dtype=np.int64)
    completed = 0
    while completed < repetitions:
        size = min(chunk_size, repetitions - completed)
        swaps = rng.integers(0, 2, size=(size, n_groups), dtype=np.int8)
        delta = np.tensordot(swaps, difference, axes=(1, 0))
        permuted = score_function(aggregate_a + delta) - score_function(
            aggregate_b - delta
        )
        extreme += np.sum(
            np.abs(permuted) >= np.abs(observed)[None, :] - 1e-15,
            axis=0,
        )
        completed += size
    return pairs, observed, (extreme + 1) / (repetitions + 1)


def _omnibus_permutation(values, score_function, repetitions, rng):
    n_groups, n_models = values.shape[:2]
    tail_size = int(np.prod(values.shape[2:]))
    observed_scores = np.asarray(score_function(np.sum(values, axis=0)))
    observed = float(np.sum((observed_scores - np.mean(observed_scores)) ** 2))
    chunk_size = _working_chunk_size(
        n_groups,
        n_models=n_models,
        tail_size=tail_size,
        itemsize=values.dtype.itemsize,
        maximum=64,
    )
    extreme = 0
    completed = 0
    trailing = (None,) * (values.ndim - 2)
    while completed < repetitions:
        size = min(chunk_size, repetitions - completed)
        permutation = np.argsort(
            rng.random((size, n_groups, n_models)), axis=2
        )
        indices = permutation[(...,) + trailing]
        expanded = np.broadcast_to(values, (size,) + values.shape)
        randomized = np.take_along_axis(expanded, indices, axis=2)
        randomized_scores = score_function(np.sum(randomized, axis=1))
        statistic = np.sum(
            (randomized_scores - np.mean(randomized_scores, axis=1, keepdims=True)) ** 2,
            axis=1,
        )
        extreme += int(np.sum(statistic >= observed - 1e-15))
        completed += size
    return observed, (extreme + 1) / (repetitions + 1)


def _confidence_interval(values, confidence_level):
    alpha = (1 - confidence_level) / 2
    bounds = np.quantile(values, [alpha, 1 - alpha])
    if not np.all(np.isfinite(bounds)):
        raise StatisticalInputError("Bootstrap confidence interval is non-finite.")
    return [float(bounds[0]), float(bounds[1])]


def _endpoint_inference(
    *, dataset, endpoint, model_ids, groups, values, point, bootstrap_scores,
    bootstrap_rejected, score_function, config,
):
    model_rows = [
        {
            "model_id": model_id,
            "estimate": float(point[index]),
            "confidence_interval": _confidence_interval(
                bootstrap_scores[:, index], config.confidence_level
            ),
        }
        for index, model_id in enumerate(model_ids)
    ]

    pairs, observed_values, raw_p_values = _pairwise_family_permutation(
        values,
        score_function,
        config.permutation_repetitions,
        _rng(config.inference_seed, dataset, endpoint, "pairwise_family"),
    )
    pairwise = []
    for pair_index, (first, second) in enumerate(pairs):
        observed = float(observed_values[pair_index])
        p_value = float(raw_p_values[pair_index])
        bootstrap_difference = (
            bootstrap_scores[:, first] - bootstrap_scores[:, second]
        )
        expected_difference = float(point[first] - point[second])
        if not np.isclose(
            observed, expected_difference, rtol=0, atol=NUMERICAL_TOLERANCE
        ):
            raise StatisticalInputError(
                "Pairwise permutation statistic disagrees with endpoint estimates."
            )
        pairwise.append({
            "model_a": model_ids[first],
            "model_b": model_ids[second],
            "estimate_a_minus_b": expected_difference,
            "confidence_interval": _confidence_interval(
                bootstrap_difference, config.confidence_level
            ),
            "p_value_raw": float(p_value),
        })

    adjusted = holm_adjust(raw_p_values)
    alpha = round(1 - config.confidence_level, 15)
    for row, adjusted_p in zip(pairwise, adjusted):
        row["p_value_holm"] = float(adjusted_p)
        row["reject_holm"] = bool(adjusted_p <= alpha)

    omnibus_statistic, omnibus_p = _omnibus_permutation(
        values,
        score_function,
        config.permutation_repetitions,
        _rng(config.inference_seed, dataset, endpoint, "omnibus"),
    )
    return {
        "endpoint": endpoint,
        "n_groups": int(len(groups)),
        "models": model_rows,
        "bootstrap": {
            "method": "paired_group_cluster_percentile",
            "repetitions": config.bootstrap_repetitions,
            "confidence_level": config.confidence_level,
            "rejected_undefined_replicates": int(bootstrap_rejected),
        },
        "omnibus_permutation": {
            "method": "within_group_model_label_randomization",
            "statistic": "sum_squared_model_deviation",
            "observed": omnibus_statistic,
            "p_value": float(omnibus_p),
            "repetitions": config.permutation_repetitions,
        },
        "pairwise": pairwise,
        "pairwise_permutation": {
            "method": "within_group_pair_swap",
            "alternative": config.pairwise_alternative,
            "repetitions": config.permutation_repetitions,
            "shared_randomizations_within_family": True,
        },
        "multiplicity": {
            "method": "holm",
            "family": config.holm_family,
            "family_size": len(pairwise),
            "alpha": alpha,
        },
        "monte_carlo_p_value_resolution": 1 / (config.permutation_repetitions + 1),
    }


def run_dataset_inference(records, config):
    """Run validated inference for both endpoints on one aligned dataset."""
    records = list(records)
    if len(records) < 2:
        raise StatisticalInputError(
            "Paired inference requires at least two aligned model runs per dataset."
        )
    dataset = records[0].dataset
    if any(record.dataset != dataset for record in records):
        raise StatisticalInputError("Inference records must all belong to one dataset.")
    model_ids = [record.run.model.model_id for record in records]
    if len(set(model_ids)) != len(model_ids):
        raise StatisticalInputError("Inference model IDs must be unique.")

    class_groups, class_values, class_point = _classification_contributions(records)
    class_bootstrap, class_rejected = _bootstrap_classification(
        class_values,
        config.bootstrap_repetitions,
        _rng(config.inference_seed, dataset, "pooled_oof_macro_f1", "bootstrap"),
    )
    classification = _endpoint_inference(
        dataset=dataset,
        endpoint="pooled_oof_macro_f1",
        model_ids=model_ids,
        groups=class_groups,
        values=class_values,
        point=class_point,
        bootstrap_scores=class_bootstrap,
        bootstrap_rejected=class_rejected,
        score_function=_f1_from_counts,
        config=config,
    )

    retrieval_groups, retrieval_sums, retrieval_counts, retrieval_point = (
        _retrieval_contributions(records)
    )
    retrieval_bootstrap, retrieval_rejected = _bootstrap_retrieval(
        retrieval_sums,
        retrieval_counts,
        config.bootstrap_repetitions,
        _rng(config.inference_seed, dataset, "label_macro_map", "bootstrap"),
    )
    total_retrieval_counts = np.sum(retrieval_counts, axis=0)

    def retrieval_score(aggregate):
        return _map_from_sums(aggregate, total_retrieval_counts)

    retrieval = _endpoint_inference(
        dataset=dataset,
        endpoint="label_macro_map",
        model_ids=model_ids,
        groups=retrieval_groups,
        values=retrieval_sums,
        point=retrieval_point,
        bootstrap_scores=retrieval_bootstrap,
        bootstrap_rejected=retrieval_rejected,
        score_function=retrieval_score,
        config=config,
    )
    return {
        "dataset": dataset,
        "model_ids": model_ids,
        "endpoints": {
            "pooled_oof_macro_f1": classification,
            "label_macro_map": retrieval,
        },
    }
