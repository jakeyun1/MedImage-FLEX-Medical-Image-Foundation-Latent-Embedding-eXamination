"""
permuted_baseline.py

Evaluate genuine embeddings after breaking their image correspondence.
"""

import json
import os
from time import perf_counter

import numpy as np

from scripts.random_baseline import aggregate_benchmark_results
from scripts.reproducibility import sample_ids_from_paths


def generate_permuted_embeddings(embeddings, seed, row_indices=None):
    array = np.asarray(embeddings)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("Embeddings must have shape (n_samples, embedding_dim).")

    if row_indices is None:
        row_indices = np.arange(array.shape[0])
    row_indices = np.asarray(row_indices, dtype=int)
    if len(row_indices) < 2 or len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("At least two unique embedding row indices are required.")
    if np.any(row_indices < 0) or np.any(row_indices >= array.shape[0]):
        raise ValueError("Permutation row indices are out of bounds.")

    rng = np.random.default_rng(seed)
    identity = np.arange(len(row_indices))
    for _ in range(100):
        permutation = rng.permutation(len(row_indices))
        if not np.any(permutation == identity):
            break
    else:
        raise RuntimeError("Could not generate a fixed-point-free permutation.")

    permuted = array.copy()
    permuted[row_indices] = array[row_indices[permutation]]
    return permuted, permutation


def _validate_cached_result(result, expected, manifest_info):
    info = result.get("baseline_info", {})
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"Cached permuted baseline has incompatible {key}: "
                f"expected {value!r}, found {info.get(key)!r}."
            )

    if manifest_info is not None:
        cached_hash = result.get("reproducibility", {}).get("sha256")
        if cached_hash != manifest_info.get("sha256"):
            raise ValueError("Cached permuted baseline uses a different manifest.")


def run_permuted_baseline(
    dataset_name,
    embeddings,
    metadata_df,
    image_paths,
    id_col,
    label_col,
    output_dir,
    matched_model_id,
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
    Repeatedly permute embedding rows while leaving image paths unchanged.
    """
    if repeats < 1:
        raise ValueError("permuted_baseline.repeats must be at least 1.")

    embedding_array = np.asarray(embeddings)
    if embedding_array.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")
    if outer_folds is None:
        raise ValueError("A stored outer-fold manifest is required.")

    image_sample_ids = sample_ids_from_paths(dataset_name, image_paths)
    if len(image_sample_ids) != len(embedding_array):
        raise ValueError("Embedding and image-path counts do not match.")
    if len(set(image_sample_ids)) != len(image_sample_ids):
        raise ValueError("Image paths contain duplicate sample IDs.")

    index_by_id = {sample_id: index for index, sample_id in enumerate(image_sample_ids)}
    missing = sorted(set(outer_folds) - set(index_by_id))
    if missing:
        raise ValueError(f"Manifest samples are missing embeddings: {missing[:5]}")
    cohort_indices = np.asarray(
        [index_by_id[sample_id] for sample_id in outer_folds], dtype=int
    )

    if benchmark_fn is None:
        from scripts.run_benchmark import run_benchmark
        benchmark_fn = run_benchmark

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    seeds = list(range(seed_start, seed_start + repeats))
    seed_results = []

    print(
        f"\n=== Permuted Embedding Baseline: {dataset_name} "
        f"({repeats} seeds, dim={embedding_array.shape[1]}) ==="
    )

    for run_index, seed in enumerate(seeds, start=1):
        result_path = os.path.join(dataset_output_dir, f"seed_{seed:010d}.json")
        expected = {
            "baseline_type": "permuted_real_embeddings",
            "matched_model_id": matched_model_id,
            "embedding_dim": int(embedding_array.shape[1]),
            "n_embeddings": int(len(cohort_indices)),
            "permutation_seed": int(seed),
            "evaluation_seed": int(evaluation_seed),
            "permutation_method": "random_derangement",
        }

        if os.path.exists(result_path) and not overwrite:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            _validate_cached_result(result, expected, manifest_info)
            print(f"Permuted baseline {run_index}/{repeats}: reusing seed {seed}.")
            seed_results.append(result)
            continue

        print(f"Permuted baseline {run_index}/{repeats}: evaluating seed {seed}.")
        permuted_embeddings, permutation = generate_permuted_embeddings(
            embedding_array, seed, row_indices=cohort_indices
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
            dataset_name, permuted_embeddings, metadata_df, image_paths,
            **benchmark_kwargs
        )
        result["baseline_info"] = {
            **expected,
            "permutation": "selected_embedding_rows_only",
            "image_paths_permuted": False,
            "generator": "numpy.random.default_rng",
            "numpy_version": np.__version__,
            "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
            "source_n_embeddings": int(embedding_array.shape[0]),
        }
        if manifest_info is not None:
            result["reproducibility"] = manifest_info
        result["runtime"]["stages"]["total_baseline_run"] = perf_counter() - start

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        seed_results.append(result)

    summary = aggregate_benchmark_results(seed_results, seeds)
    summary["baseline_type"] = "permuted_real_embeddings"
    summary["interval_method"] = "empirical_percentile_across_permutation_seeds"
    summary["matched_model_id"] = matched_model_id
    summary["embedding_dim"] = int(embedding_array.shape[1])
    summary["n_embeddings"] = int(len(cohort_indices))
    summary["source_n_embeddings"] = int(embedding_array.shape[0])
    if manifest_info is not None:
        summary["reproducibility"] = manifest_info

    with open(os.path.join(dataset_output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Permuted baseline results saved to: {os.path.abspath(dataset_output_dir)}\n")
    return summary
