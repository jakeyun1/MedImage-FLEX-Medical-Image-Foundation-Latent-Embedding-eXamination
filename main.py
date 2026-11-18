"""
main.py

Executes the testbench.
"""

import os
import json
import argparse
from copy import deepcopy
import numpy as np
from datetime import datetime
from time import perf_counter

from scripts.dataset_contracts import get_dataset_contract
from scripts.provenance import build_dataset_fingerprint, build_run_provenance

PROHIBITED_CHARS = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", "_"]

# Make the working directory relative to the testbench
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def clean_name(name, desired_char):
    """
    Helper function for standardizing model names.

    Args:
        name : Initial model ID
        desired_char : Replacement for illegal characters

    Returns:
        filename : Cleaned, legal model name
    """
    for char in PROHIBITED_CHARS:
        name = name.replace(char, desired_char)

    return name

def load_config(config_path: str):
    """
    Helper function for loading the configuration JSON file.

    Args:
        config_path : The path to the config JSON file

    Returns:
        The dict representation of the JSON file
    """
    with open(config_path, "r", encoding = "utf-8") as f:
        return json.load(f)

def write_json(content, output_path: str):
    """
    Writes the JSON content to a JSON file.

    Args:
        content : Dict to store
        output_path : File path to write to
    """
    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(content, f, indent = 2)

def main():
    """
    Runs MedImage-FLEX
    """
    parser = argparse.ArgumentParser(description = "Run MedImage-FLEX — a medical FM embedding testbench")
    parser.add_argument("--config", required = True, help = "Path to config JSON file")
    parser.add_argument("--num-workers", default = 0, help = "Number of DataLoader workers, default is 0")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print("Please use absolute paths for config files.")
        raise

    num_workers = int(args.num_workers)

    model_id = cfg["model_id"]
    model_name = clean_name(model_id, "-")
    output_dir = cfg.get("output_dir", f".{os.sep}results")

    datasets = cfg["dataset"]["datasets"]
    batch_size = int(cfg["dataset"].get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("dataset.batch_size must be a positive integer.")
    shuffle = cfg["dataset"].get("shuffle", False)
    chexpert_uncertainty_policy = cfg["dataset"].get(
        "chexpert_uncertainty_policy", "finding_specific"
    )
    chexpert_sensitivity_policies = cfg["dataset"].get(
        "chexpert_sensitivity_policies",
        [
            policy for policy in ("finding_specific", "u_zeros", "u_ones")
            if policy != chexpert_uncertainty_policy
        ],
    )

    normalize_embeddings = cfg.get("normalize_embeddings", True)
    cache_embeddings = cfg.get("cache_embeddings", False)

    random_baseline_cfg = cfg.get("random_baseline", {})
    run_random_embeddings = random_baseline_cfg.get("enabled", False)
    random_baseline_repeats = int(random_baseline_cfg.get("repeats", 50))
    random_baseline_seed = int(random_baseline_cfg.get("seed", 42))
    overwrite_random_baseline = random_baseline_cfg.get("overwrite", False)

    label_prior_cfg = cfg.get("label_prior_baseline", {})
    run_label_prior = label_prior_cfg.get("enabled", False)

    permuted_baseline_cfg = cfg.get("permuted_baseline", {})
    run_permuted_embeddings = permuted_baseline_cfg.get("enabled", False)
    permuted_baseline_repeats = int(permuted_baseline_cfg.get("repeats", 50))
    permuted_baseline_seed = int(permuted_baseline_cfg.get("seed", 42))
    overwrite_permuted_baseline = permuted_baseline_cfg.get("overwrite", False)

    reproducibility_cfg = cfg.get("reproducibility", {})
    max_samples = cfg["dataset"].get("max_samples", 5000)
    if max_samples is not None:
        max_samples = int(max_samples)
    sample_seed = int(reproducibility_cfg.get("sample_seed", 42))
    cohort_sensitivity_seeds_raw = reproducibility_cfg.get(
        "cohort_sensitivity_seeds", [43, 44, 45]
    )
    fold_seed = int(reproducibility_cfg.get("fold_seed", 42))
    evaluation_seed = int(reproducibility_cfg.get("evaluation_seed", 42))
    outer_folds = int(reproducibility_cfg.get("outer_folds", 5))
    manifest_dir = reproducibility_cfg.get("manifest_dir", f".{os.sep}manifests")

    run_provenance = build_run_provenance(cfg, script_dir)

    os.makedirs(output_dir, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # This dictates the path for the results directory for a model
    run_folder = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_folder)

    write_json(cfg, os.path.join(run_folder, "config_used.json"))

    # Prepare for multiprocessing if requested
    if num_workers >= 1:
        import torch.multiprocessing as mp
        mp.set_start_method(method = "spawn", force = True)
    else:
        num_workers = max(0, num_workers)

    # Load dependecies once JSON file is parsed
    from scripts.dataloading import (
        _apply_chexpert_uncertainty_policy,
        load_dataset,
        validate_chexpert_uncertainty_policies,
    )
    from scripts.data_audit import ordered_ids_sha256
    from scripts.models import build_backend
    from scripts.extraction import extract_embeddings, preprocessing_fingerprint
    from scripts.label_prior_baseline import run_label_prior_baselines
    from scripts.odir import (
        pool_odir_patient_embeddings,
    )
    from scripts.permuted_baseline import run_permuted_baseline
    from scripts.random_baseline import run_random_baseline
    from scripts.reproducibility import (
        COHORT_SENSITIVITY_DATASETS,
        COHORT_SENSITIVITY_SCHEMA_VERSION,
        cohort_overlap_summary,
        cohort_seed_tag,
        prepare_experiment_manifest,
        sample_ids_from_paths,
        validate_cohort_selection_record,
        validate_cohort_sensitivity_index,
        validate_cohort_sensitivity_seeds,
    )
    from scripts.run_benchmark import run_benchmark
    from scripts.runtime_metadata import (
        build_embedding_runtime,
        collect_backend_execution_metadata,
        collect_peak_memory,
        reset_peak_memory,
        synchronize_backend,
    )

    chexpert_uncertainty_policy, chexpert_sensitivity_policies = (
        validate_chexpert_uncertainty_policies(
            chexpert_uncertainty_policy, chexpert_sensitivity_policies
        )
    )
    cohort_sensitivity_seeds = validate_cohort_sensitivity_seeds(
        sample_seed, cohort_sensitivity_seeds_raw
    )

    print(f"\n=== Running Benchmark: {model_name} ===")
    print(f"Model: {model_id}")
    print(f"Output folder: {os.path.abspath(run_folder)}\n")

    backend_loading_start = perf_counter()
    backend = build_backend(model_id)
    synchronize_backend(backend)
    backend_loading_seconds = perf_counter() - backend_loading_start

    for dataset_name in datasets:
        dataset_start = perf_counter()
        print(f"Dataset: {dataset_name}")

        contract = get_dataset_contract(dataset_name)
        id_col = contract.id_column
        label_col = contract.label_column
        group_col = contract.group_column

        # Load dataset and corresponding backend transform
        transform = backend.get_transform()

        load_start = perf_counter()
        dataloader, metadata_df = load_dataset(
            dataset_name,
            transform = transform,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            chexpert_uncertainty_policy = chexpert_uncertainty_policy
        )
        dataset_loading_seconds = perf_counter() - load_start

        # Extract and optionally cache every eligible source image. The frozen
        # manifest is applied afterward so alternative evaluation seeds can reuse
        # the same full embedding bank without rerunning the foundation model.
        reset_peak_memory(backend)
        embeddings, image_paths, source, embedding_timing = extract_embeddings(
            dataloader,
            backend,
            normalize = normalize_embeddings,
            cache = cache_embeddings,
            return_runtime = True,
        )
        embedding_seconds = embedding_timing["total_seconds"]
        if len(embeddings) != len(image_paths):
            raise ValueError("Embedding and image-path counts do not match.")

        source_embedding_array = np.asarray(embeddings)
        source_sample_ids = sample_ids_from_paths(dataset_name, image_paths)
        evaluation_audit = {
            "evaluation_unit": contract.evaluation_unit,
            "source_image_embeddings": int(len(source_embedding_array)),
            "evaluation_samples": int(len(source_embedding_array)),
            "aggregation": "none",
        }
        evaluation_embeddings = source_embedding_array
        evaluation_metadata = metadata_df
        evaluation_paths = image_paths
        evaluation_sample_ids = source_sample_ids
        if dataset_name == "odir":
            (
                evaluation_embeddings,
                evaluation_metadata,
                evaluation_sample_ids,
                evaluation_audit,
            ) = pool_odir_patient_embeddings(
                source_embedding_array,
                metadata_df,
                image_paths,
                normalize=normalize_embeddings,
            )
            # Evaluation functions use explicit sample IDs; these references are
            # never opened as files.
            evaluation_paths = list(evaluation_sample_ids)

        evaluation_metadata, fold_assignments, manifest_info = (
            prepare_experiment_manifest(
                evaluation_metadata,
                model_id=model_id,
                dataset_name=dataloader.dataset_name,
                id_col=id_col,
                label_col=label_col,
                manifest_dir=manifest_dir,
                group_col=group_col,
                max_samples=max_samples,
                sample_seed=sample_seed,
                n_splits=outer_folds,
                fold_seed=fold_seed,
                available_sample_ids=evaluation_sample_ids,
            )
        )
        print(
            f"Extracted {len(source_embedding_array)} source-image embeddings; "
            f"evaluating {len(evaluation_metadata)} manifest samples: "
            f"{manifest_info['path']}"
        )

        # Run benchmark suite
        results = run_benchmark(
            dataset_name,
            evaluation_embeddings,
            evaluation_metadata,
            evaluation_paths,
            id_col = id_col,
            label_col = label_col,
            outer_folds = fold_assignments,
            n_splits = outer_folds,
            random_state = evaluation_seed,
            group_col = group_col,
            sample_ids = evaluation_sample_ids,
            oof_output_dir = os.path.join(run_folder, "oof", dataset_name),
            oof_path_prefix = os.path.join("oof", dataset_name),
            retrieval_output_dir = os.path.join(
                run_folder, "retrieval", dataset_name
            ),
            retrieval_path_prefix = os.path.join("retrieval", dataset_name),
            clustering_output_dir = os.path.join(
                run_folder, "clustering", dataset_name
            ),
            clustering_path_prefix = os.path.join("clustering", dataset_name),
        )

        emb_array = np.asarray(evaluation_embeddings)
        results["embedding_info"] = {
            "model_id": model_id,
            "embedding_dim": int(emb_array.shape[1]),
            "normalized": normalize_embeddings,
            "n_source_image_embeddings": int(source_embedding_array.shape[0]),
            "n_evaluation_embeddings": int(emb_array.shape[0]),
            "n_evaluated_embeddings": int(manifest_info["n_samples"]),
            "ordered_source_image_ids_sha256": ordered_ids_sha256(source_sample_ids),
            "ordered_evaluation_ids_sha256": ordered_ids_sha256(
                evaluation_sample_ids
            ),
            "evaluation_unit": contract.evaluation_unit,
            "extraction_scope": "all_eligible_source_images_before_manifest_selection",
            "source": source,
            "preprocessing": backend.preprocessing_spec,
            "preprocessing_sha256": preprocessing_fingerprint(backend),
        }
        results["data_audit"] = dataloader.data_audit
        results["data_audit"]["evaluation"] = evaluation_audit
        results["reproducibility"] = manifest_info
        applicable_cohort_sensitivity_seeds = (
            list(cohort_sensitivity_seeds)
            if (
                dataset_name in COHORT_SENSITIVITY_DATASETS
                and max_samples is not None
                and manifest_info["n_eligible_samples"] > manifest_info["n_samples"]
            )
            else []
        )
        results["cohort_selection"] = {
            "schema_version": COHORT_SENSITIVITY_SCHEMA_VERSION,
            "analysis_type": "cohort_sampling",
            "role": "primary",
            "primary_sample_seed": int(sample_seed),
            "sample_seed": int(sample_seed),
            "fold_seed": int(fold_seed),
            "evaluation_seed": int(evaluation_seed),
            "outer_folds": int(outer_folds),
            "sensitivity_sample_seeds": applicable_cohort_sensitivity_seeds,
            "max_samples": max_samples,
            "n_eligible_samples": int(manifest_info["n_eligible_samples"]),
            "n_selected_samples": int(manifest_info["n_samples"]),
            "manifest_sha256": manifest_info["sha256"],
            "manifest_applied_after_full_embedding_extraction": True,
        }
        validate_cohort_selection_record(results["cohort_selection"])
        results["run_provenance"] = run_provenance
        results["execution"] = collect_backend_execution_metadata(backend)
        
        results["runtime"]["stages"]["dataset_loading"] = dataset_loading_seconds
        results["runtime"]["stages"]["embedding_extraction"] = embedding_seconds
        results["runtime"]["stages"][
            "model_loading_once_per_model"
        ] = backend_loading_seconds
        results["runtime"]["stages"]["total_dataset_run"] = perf_counter() - dataset_start
        results["runtime"]["embedding"] = build_embedding_runtime(
            embedding_seconds,
            int(source_embedding_array.shape[0]),
            batch_size,
            num_workers,
            source,
            timing_components=embedding_timing,
        )
        results["runtime"]["embedding"]["extraction_scope"] = (
            "all_eligible_source_images_before_manifest_selection"
        )
        results["runtime"]["peak_memory"] = collect_peak_memory(backend)

        # Save results
        results_path = os.path.join(run_folder, f"{dataset_name}.json")
        write_json(results, results_path)

        if dataset_name == "chexpert":
            primary_ids = evaluation_metadata[id_col].astype(str).tolist()
            primary_groups = evaluation_metadata[group_col].astype(str).tolist()
            for policy in chexpert_sensitivity_policies:
                sensitivity_start = perf_counter()
                sensitivity_metadata, policy_audit = (
                    _apply_chexpert_uncertainty_policy(
                        evaluation_metadata, policy
                    )
                )
                sensitivity_ids = sensitivity_metadata[id_col].astype(str).tolist()
                sensitivity_groups = sensitivity_metadata[group_col].astype(str).tolist()
                if sensitivity_ids != primary_ids:
                    raise ValueError(
                        f"CheXpert sensitivity {policy!r} changed the cohort or its order."
                    )
                if sensitivity_groups != primary_groups:
                    raise ValueError(
                        f"CheXpert sensitivity {policy!r} changed patient groups."
                    )

                sensitivity_results = run_benchmark(
                    dataset_name,
                    evaluation_embeddings,
                    sensitivity_metadata,
                    evaluation_paths,
                    id_col=id_col,
                    label_col=label_col,
                    outer_folds=fold_assignments,
                    n_splits=outer_folds,
                    random_state=evaluation_seed,
                    group_col=group_col,
                    sample_ids=evaluation_sample_ids,
                    oof_output_dir=os.path.join(
                        run_folder, "oof", "chexpert_sensitivity", policy
                    ),
                    oof_path_prefix=os.path.join(
                        "oof", "chexpert_sensitivity", policy
                    ),
                    retrieval_output_dir=os.path.join(
                        run_folder, "retrieval", "chexpert_sensitivity", policy
                    ),
                    retrieval_path_prefix=os.path.join(
                        "retrieval", "chexpert_sensitivity", policy
                    ),
                    clustering_output_dir=os.path.join(
                        run_folder, "clustering", "chexpert_sensitivity", policy
                    ),
                    clustering_path_prefix=os.path.join(
                        "clustering", "chexpert_sensitivity", policy
                    ),
                )
                label_fingerprint = build_dataset_fingerprint(
                    sensitivity_ids,
                    sensitivity_groups,
                    sensitivity_metadata[label_col].tolist(),
                )
                sensitivity_results["embedding_info"] = deepcopy(
                    results["embedding_info"]
                )
                sensitivity_results["data_audit"] = deepcopy(results["data_audit"])
                sensitivity_results["data_audit"]["sensitivity_evaluation"] = {
                    "policy": policy_audit,
                    "fingerprint": label_fingerprint,
                }
                sensitivity_results["reproducibility"] = deepcopy(manifest_info)
                sensitivity_results["cohort_selection"] = deepcopy(
                    results["cohort_selection"]
                )
                sensitivity_results["run_provenance"] = run_provenance
                sensitivity_results["execution"] = deepcopy(results["execution"])
                sensitivity_results["sensitivity_analysis"] = {
                    "primary_uncertainty_policy": chexpert_uncertainty_policy,
                    "uncertainty_policy": policy,
                    "matched_cohort": True,
                    "matched_patient_groups": True,
                    "matched_outer_folds": True,
                    "reused_embeddings": True,
                    "label_fingerprint": label_fingerprint,
                }
                sensitivity_results["runtime"]["stages"][
                    "total_sensitivity_run"
                ] = perf_counter() - sensitivity_start

                sensitivity_dir = os.path.join(
                    run_folder, "chexpert_sensitivity"
                )
                os.makedirs(sensitivity_dir, exist_ok=True)
                write_json(
                    sensitivity_results,
                    os.path.join(sensitivity_dir, f"{policy}.json"),
                )

        if applicable_cohort_sensitivity_seeds:
            primary_manifest_ids = evaluation_metadata[id_col].astype(str).tolist()
            cohort_sensitivity_dir = os.path.join(
                run_folder, "cohort_sensitivity", dataset_name
            )
            os.makedirs(cohort_sensitivity_dir, exist_ok=True)
            cohort_index = {
                "schema_version": COHORT_SENSITIVITY_SCHEMA_VERSION,
                "analysis_type": "cohort_sampling_sensitivity",
                "dataset": dataset_name,
                "model_id": model_id,
                "primary_sample_seed": int(sample_seed),
                "fold_seed": int(fold_seed),
                "evaluation_seed": int(evaluation_seed),
                "outer_folds": int(outer_folds),
                "sensitivity_sample_seeds": applicable_cohort_sensitivity_seeds,
                "max_samples": max_samples,
                "embedding_bank": {
                    "reused_for_every_cohort": True,
                    "additional_embedding_extraction_per_seed": False,
                    "n_source_image_embeddings": int(source_embedding_array.shape[0]),
                    "n_available_evaluation_embeddings": int(
                        np.asarray(evaluation_embeddings).shape[0]
                    ),
                    "embedding_dim": int(emb_array.shape[1]),
                    "ordered_source_image_ids_sha256": results["embedding_info"][
                        "ordered_source_image_ids_sha256"
                    ],
                    "ordered_evaluation_ids_sha256": results["embedding_info"][
                        "ordered_evaluation_ids_sha256"
                    ],
                    "preprocessing_sha256": results["embedding_info"][
                        "preprocessing_sha256"
                    ],
                },
                "primary": {
                    "sample_seed": int(sample_seed),
                    "result_path": f"{dataset_name}.json",
                    "manifest_sha256": manifest_info["sha256"],
                    "n_samples": int(manifest_info["n_samples"]),
                },
                "sensitivities": [],
            }

            for sensitivity_seed in applicable_cohort_sensitivity_seeds:
                cohort_start = perf_counter()
                (
                    cohort_metadata,
                    cohort_folds,
                    cohort_manifest_info,
                ) = prepare_experiment_manifest(
                    metadata_df,
                    model_id=model_id,
                    dataset_name=dataloader.dataset_name,
                    id_col=id_col,
                    label_col=label_col,
                    manifest_dir=manifest_dir,
                    group_col=group_col,
                    max_samples=max_samples,
                    sample_seed=sensitivity_seed,
                    n_splits=outer_folds,
                    fold_seed=fold_seed,
                    available_sample_ids=evaluation_sample_ids,
                )
                cohort_ids = cohort_metadata[id_col].astype(str).tolist()
                overlap = cohort_overlap_summary(primary_manifest_ids, cohort_ids)
                seed_tag = cohort_seed_tag(sensitivity_seed)
                artifact_parts = ("cohort_sensitivity", dataset_name, seed_tag)

                cohort_results = run_benchmark(
                    dataset_name,
                    evaluation_embeddings,
                    cohort_metadata,
                    evaluation_paths,
                    id_col=id_col,
                    label_col=label_col,
                    outer_folds=cohort_folds,
                    n_splits=outer_folds,
                    random_state=evaluation_seed,
                    group_col=group_col,
                    sample_ids=evaluation_sample_ids,
                    oof_output_dir=os.path.join(run_folder, "oof", *artifact_parts),
                    oof_path_prefix=os.path.join("oof", *artifact_parts),
                    retrieval_output_dir=os.path.join(
                        run_folder, "retrieval", *artifact_parts
                    ),
                    retrieval_path_prefix=os.path.join("retrieval", *artifact_parts),
                    clustering_output_dir=os.path.join(
                        run_folder, "clustering", *artifact_parts
                    ),
                    clustering_path_prefix=os.path.join("clustering", *artifact_parts),
                )
                cohort_results["embedding_info"] = deepcopy(results["embedding_info"])
                cohort_results["embedding_info"]["n_evaluated_embeddings"] = int(
                    cohort_manifest_info["n_samples"]
                )
                cohort_results["data_audit"] = deepcopy(results["data_audit"])
                cohort_results["reproducibility"] = cohort_manifest_info
                cohort_results["cohort_selection"] = {
                    "schema_version": COHORT_SENSITIVITY_SCHEMA_VERSION,
                    "analysis_type": "cohort_sampling",
                    "role": "sensitivity",
                    "primary_sample_seed": int(sample_seed),
                    "sample_seed": int(sensitivity_seed),
                    "fold_seed": int(fold_seed),
                    "evaluation_seed": int(evaluation_seed),
                    "outer_folds": int(outer_folds),
                    "sensitivity_sample_seeds": applicable_cohort_sensitivity_seeds,
                    "max_samples": max_samples,
                    "n_eligible_samples": int(
                        cohort_manifest_info["n_eligible_samples"]
                    ),
                    "n_selected_samples": int(cohort_manifest_info["n_samples"]),
                    "manifest_sha256": cohort_manifest_info["sha256"],
                    "primary_manifest_sha256": manifest_info["sha256"],
                    "primary_n_selected_samples": int(manifest_info["n_samples"]),
                    "manifest_applied_after_full_embedding_extraction": True,
                    "full_embedding_bank_reused": True,
                    "additional_embedding_extraction": False,
                    "overlap_with_primary": overlap,
                }
                validate_cohort_selection_record(
                    cohort_results["cohort_selection"]
                )
                cohort_results["run_provenance"] = run_provenance
                cohort_results["execution"] = deepcopy(results["execution"])
                cohort_results["runtime"]["embedding_reuse"] = {
                    "full_embedding_bank_reused": True,
                    "additional_embedding_extraction_seconds": 0.0,
                    "source_primary_sample_seed": int(sample_seed),
                }
                cohort_results["runtime"]["stages"][
                    "total_cohort_sensitivity_run"
                ] = perf_counter() - cohort_start

                result_filename = f"{seed_tag}.json"
                write_json(
                    cohort_results,
                    os.path.join(cohort_sensitivity_dir, result_filename),
                )
                cohort_index["sensitivities"].append({
                    "sample_seed": int(sensitivity_seed),
                    "result_path": os.path.join(
                        "cohort_sensitivity", dataset_name, result_filename
                    ),
                    "manifest_sha256": cohort_manifest_info["sha256"],
                    "n_samples": int(cohort_manifest_info["n_samples"]),
                    "overlap_with_primary": overlap,
                })

            validate_cohort_sensitivity_index(cohort_index)
            write_json(
                cohort_index,
                os.path.join(cohort_sensitivity_dir, "index.json"),
            )

        if run_random_embeddings:
            run_random_baseline(
                dataset_name,
                embedding_shape = emb_array.shape,
                metadata_df = evaluation_metadata,
                image_paths = evaluation_paths,
                id_col = id_col,
                label_col = label_col,
                output_dir = os.path.join(run_folder, "random_baseline"),
                matched_model_id = model_id,
                normalize = normalize_embeddings,
                repeats = random_baseline_repeats,
                seed_start = random_baseline_seed,
                overwrite = overwrite_random_baseline,
                outer_folds = fold_assignments,
                n_splits = outer_folds,
                evaluation_seed = evaluation_seed,
                manifest_info = manifest_info,
                group_col = group_col,
                sample_ids = evaluation_sample_ids
            )

        if run_label_prior:
            run_label_prior_baselines(
                dataset_name,
                metadata_df = evaluation_metadata,
                id_col = id_col,
                label_col = label_col,
                output_dir = os.path.join(run_folder, "label_prior_baseline"),
                outer_folds = fold_assignments,
                n_splits = outer_folds,
                random_state = evaluation_seed,
                manifest_info = manifest_info
            )

        if run_permuted_embeddings:
            run_permuted_baseline(
                dataset_name,
                embeddings = emb_array,
                metadata_df = evaluation_metadata,
                image_paths = evaluation_paths,
                id_col = id_col,
                label_col = label_col,
                output_dir = os.path.join(run_folder, "permuted_baseline"),
                matched_model_id = model_id,
                repeats = permuted_baseline_repeats,
                seed_start = permuted_baseline_seed,
                overwrite = overwrite_permuted_baseline,
                outer_folds = fold_assignments,
                n_splits = outer_folds,
                evaluation_seed = evaluation_seed,
                manifest_info = manifest_info,
                group_col = group_col,
                sample_ids = evaluation_sample_ids
            )

        print(f"\n=== Dataset Complete ===\n\n")
    
    print(f"=== Benchmark Complete ===")
    print(f"Results saved to: {os.path.abspath(run_folder)}\n")

if __name__ == "__main__":
    main()
