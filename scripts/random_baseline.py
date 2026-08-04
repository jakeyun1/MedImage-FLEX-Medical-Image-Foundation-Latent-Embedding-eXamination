"""
Generate and evaluate dimension-matched random embedding baselines.
"""

import json
import os
from time import perf_counter

import numpy as np

from scripts.reproducibility import sample_ids_from_paths


HEADLINE_METRICS = (
    ("mlp_cv", "accuracy"),
    ("mlp_cv", "f1_weighted"),
    ("mlp_cv", "precision_weighted"),
    ("mlp_cv", "roc_auc"),
    ("knn_cv", "best_scores", "accuracy"),
    ("knn_cv", "best_scores", "f1_weighted"),
    ("knn_cv", "best_scores", "precision_weighted"),
    ("knn_cv", "best_scores", "roc_auc"),
    ("logreg_cv", "accuracy"),
    ("logreg_cv", "f1_weighted"),
    ("logreg_cv", "precision_weighted"),
    ("logreg_cv", "roc_auc"),
    ("retrieval", "recall_at_k", "1"),
    ("retrieval", "recall_at_k", "5"),
    ("retrieval", "recall_at_k", "10"),
    ("retrieval", "map"),
    ("clustering", "class_count_k", "ARI"),
    ("clustering", "class_count_k", "NMI"),
    ("clustering", "class_count_k", "Silhouette"),
)


def generate_random_embeddings(shape, seed, normalize=True):
    """
    Generate independent standard-normal vectors with a reproducible seed.
    """
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("Random embedding shape must be (n_samples, embedding_dim).")

    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal(shape, dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, np.finfo(np.float32).tiny)

    return embeddings


def _write_json(content, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def _read_metric(result, path):
    value = result
    for key in path:
        if isinstance(value, dict) and key not in value and key.isdigit():
            key = int(key)
        value = value[key]

    # Classification summaries store [outer-fold mean, outer-fold std].
    if isinstance(value, list):
        value = value[0]

    return float(value)


def aggregate_benchmark_results(results, seeds):
    metrics = {}

    for path in HEADLINE_METRICS:
        values = np.asarray([_read_metric(result, path) for result in results], dtype=float)
        finite_values = values[np.isfinite(values)]
        metric_name = ".".join(path)

        if len(finite_values) == 0:
            metrics[metric_name] = {
                "mean": None,
                "std": None,
                "ci_95": [None, None],
                "values": [None for _ in values],
            }
            continue

        metrics[metric_name] = {
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0,
            "ci_95": [
                float(np.percentile(finite_values, 2.5)),
                float(np.percentile(finite_values, 97.5)),
            ],
            "values": [float(value) if np.isfinite(value) else None for value in values],
        }

    return {
        "baseline_type": "random_gaussian",
        "n_repeats": len(results),
        "seeds": [int(seed) for seed in seeds],
        "interval_method": "empirical_percentile_across_random_embedding_seeds",
        "metrics": metrics,
    }


def _validate_cached_result(result, expected, manifest_info):
    info = result.get("baseline_info", {})
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"Cached random baseline has incompatible {key}: "
                f"expected {value!r}, found {info.get(key)!r}."
            )

    if manifest_info is not None:
        cached_hash = result.get("reproducibility", {}).get("sha256")
        if cached_hash != manifest_info.get("sha256"):
            raise ValueError("Cached random baseline uses a different manifest.")


def run_random_baseline(
    dataset_name,
    embedding_shape,
    metadata_df,
    image_paths,
    id_col,
    label_col,
    output_dir,
    matched_model_id,
    normalize=True,
    repeats=20,
    seed_start=42,
    overwrite=False,
    outer_folds=None,
    n_splits=5,
    evaluation_seed=42,
    manifest_info=None,
    group_col=None,
    benchmark_fn=None,
):
    """
    Run the complete benchmark on repeated dimension-matched random embeddings.

    Existing per-seed results are reused by default so an interrupted baseline
    run can resume without repeating completed benchmark evaluations.
    """
    if repeats < 1:
        raise ValueError("random_baseline.repeats must be at least 1.")
    if outer_folds is None:
        raise ValueError("A stored outer-fold manifest is required.")

    image_sample_ids = sample_ids_from_paths(dataset_name, image_paths)
    if len(image_sample_ids) != embedding_shape[0]:
        raise ValueError("Embedding and image-path counts do not match.")
    if len(set(image_sample_ids)) != len(image_sample_ids):
        raise ValueError("Image paths contain duplicate sample IDs.")

    path_by_id = dict(zip(image_sample_ids, image_paths))
    missing = sorted(set(outer_folds) - set(path_by_id))
    if missing:
        raise ValueError(f"Manifest samples are missing image paths: {missing[:5]}")
    cohort_paths = [path_by_id[sample_id] for sample_id in outer_folds]
    cohort_shape = (len(cohort_paths), int(embedding_shape[1]))

    if benchmark_fn is None:
        from scripts.run_benchmark import run_benchmark
        benchmark_fn = run_benchmark

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    seeds = list(range(seed_start, seed_start + repeats))
    seed_results = []

    print(
        f"\n=== Random Gaussian Baseline: {dataset_name} "
        f"({repeats} seeds, dim={embedding_shape[1]}) ==="
    )

    for run_index, seed in enumerate(seeds, start=1):
        result_path = os.path.join(dataset_output_dir, f"seed_{seed:010d}.json")
        expected = {
            "baseline_type": "random_gaussian",
            "matched_model_id": matched_model_id,
            "embedding_dim": int(cohort_shape[1]),
            "n_embeddings": int(cohort_shape[0]),
            "normalized": bool(normalize),
            "representation_seed": int(seed),
            "evaluation_seed": int(evaluation_seed),
        }

        if os.path.exists(result_path) and not overwrite:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            _validate_cached_result(result, expected, manifest_info)
            print(f"Random baseline {run_index}/{repeats}: reusing seed {seed}.")
            seed_results.append(result)
            continue

        print(f"Random baseline {run_index}/{repeats}: evaluating seed {seed}.")
        embeddings = generate_random_embeddings(
            cohort_shape,
            seed=seed,
            normalize=normalize,
        )

        start = perf_counter()
        benchmark_kwargs = {
            "id_col": id_col,
            "label_col": label_col,
            "outer_folds": outer_folds,
            "n_splits": n_splits,
            "random_state": evaluation_seed,
        }
        if group_col is not None:
            benchmark_kwargs["group_col"] = group_col
        result = benchmark_fn(
            dataset_name, embeddings, metadata_df, cohort_paths, **benchmark_kwargs
        )
        result["baseline_info"] = {
            **expected,
            "distribution": "independent_standard_normal",
            "generator": "numpy.random.default_rng",
            "numpy_version": np.__version__,
            "source_n_embeddings": int(embedding_shape[0]),
        }
        if manifest_info is not None:
            result["reproducibility"] = manifest_info
        result["runtime"]["stages"]["total_baseline_run"] = perf_counter() - start

        _write_json(result, result_path)
        seed_results.append(result)

    summary = aggregate_benchmark_results(seed_results, seeds)
    summary["matched_model_id"] = matched_model_id
    summary["embedding_dim"] = int(cohort_shape[1])
    summary["n_embeddings"] = int(cohort_shape[0])
    summary["source_n_embeddings"] = int(embedding_shape[0])
    summary["normalized"] = bool(normalize)
    if manifest_info is not None:
        summary["reproducibility"] = manifest_info
    _write_json(summary, os.path.join(dataset_output_dir, "summary.json"))

    print(f"Random baseline results saved to: {os.path.abspath(dataset_output_dir)}\n")
    return summary
