"""
main.py

Executes the testbench.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from time import perf_counter

PROHIBITED_CHARS = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", "_"]

# Format: (id_col, label_col)
DATASET_COL_MAP = {"pad_ufes": ("img_id", "diagnostic"), "cbis_ddsm": ("cropped image file path", "pathology"),
                   "odir": ("filename", "target"), "ham10000": ("image_id", "dx"),
                   "chexpert": ("Path", "Diagnosis")}
GROUP_COL_MAP = {"pad_ufes": "patient_id", "cbis_ddsm": "patient_id",
                 "odir": "ID", "ham10000": "lesion_id", "chexpert": "patient_id"}

ID_COL_IDX = 0
LABEL_COL_IDX = 1

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
    batch_size = cfg["dataset"].get("batch_size", 32)
    shuffle = cfg["dataset"].get("shuffle", False)

    normalize_embeddings = cfg["embeddings"].get("normalize", True)
    cache_embeddings = cfg["embeddings"].get("cache", False)
    random_baseline_cfg = cfg.get("random_baseline", {})
    run_random_embeddings = random_baseline_cfg.get("enabled", False)
    random_baseline_repeats = int(random_baseline_cfg.get("repeats", 20))
    random_baseline_seed = int(random_baseline_cfg.get("seed", 42))
    overwrite_random_baseline = random_baseline_cfg.get("overwrite", False)
    label_prior_cfg = cfg.get("label_prior_baseline", {})
    run_label_prior = label_prior_cfg.get("enabled", False)
    permuted_baseline_cfg = cfg.get("permuted_baseline", {})
    run_permuted_embeddings = permuted_baseline_cfg.get("enabled", False)
    permuted_baseline_repeats = int(permuted_baseline_cfg.get("repeats", 20))
    permuted_baseline_seed = int(permuted_baseline_cfg.get("seed", 42))
    overwrite_permuted_baseline = permuted_baseline_cfg.get("overwrite", False)
    reproducibility_cfg = cfg.get("reproducibility", {})
    max_samples = cfg["dataset"].get("max_samples", 5000)
    if max_samples is not None:
        max_samples = int(max_samples)
    sample_seed = int(reproducibility_cfg.get("sample_seed", 42))
    fold_seed = int(reproducibility_cfg.get("fold_seed", 42))
    evaluation_seed = int(reproducibility_cfg.get("evaluation_seed", 42))
    outer_folds = int(reproducibility_cfg.get("outer_folds", 5))
    manifest_dir = reproducibility_cfg.get("manifest_dir", f".{os.sep}manifests")

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
    from scripts.dataloading import load_dataset
    from scripts.models import build_backend
    from scripts.extraction import extract_embeddings
    from scripts.label_prior_baseline import run_label_prior_baselines
    from scripts.permuted_baseline import run_permuted_baseline
    from scripts.random_baseline import run_random_baseline
    from scripts.reproducibility import prepare_experiment_manifest, sample_ids_from_paths
    from scripts.run_benchmark import run_benchmark

    print(f"\n=== Running Benchmark: {model_name} ===")
    print(f"Model: {model_id}")
    print(f"Output folder: {os.path.abspath(run_folder)}\n")

    for dataset_name in datasets:
        dataset_start = perf_counter()
        print(f"Dataset: {dataset_name}")

        id_col = DATASET_COL_MAP[dataset_name][ID_COL_IDX]
        label_col = DATASET_COL_MAP[dataset_name][LABEL_COL_IDX]

        # Build backend
        backend = build_backend(model_id)

        # Load dataset and corresponding backend transform
        transform = backend.get_transform()

        load_start = perf_counter()
        dataloader, metadata_df = load_dataset(
            dataset_name,
            transform = transform,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )
        dataset_loading_seconds = perf_counter() - load_start

        metadata_df, fold_assignments, manifest_info = prepare_experiment_manifest(
            metadata_df,
            dataset_name = dataloader.protocol_name,
            id_col = id_col,
            label_col = label_col,
            manifest_dir = manifest_dir,
            group_col = GROUP_COL_MAP[dataset_name],
            max_samples = max_samples,
            sample_seed = sample_seed,
            n_splits = outer_folds,
            fold_seed = fold_seed,
            available_sample_ids = sample_ids_from_paths(
                dataset_name, dataloader.dataset.image_paths
            )
        )
        print(
            f"Using {len(metadata_df)} samples from reproducibility manifest: "
            f"{manifest_info['path']}"
        )
        if hasattr(dataloader, "dataset_audit"):
            manifest_info["dataset_preparation"] = dataloader.dataset_audit

        # Extract embeddings
        embedding_start = perf_counter()
        embeddings, image_paths, source = extract_embeddings(
            dataloader,
            backend,
            normalize = normalize_embeddings,
            cache = cache_embeddings
        )
        embedding_seconds = perf_counter() - embedding_start
        if len(embeddings) != len(image_paths):
            raise ValueError("Embedding and image-path counts do not match.")

        # Run benchmark suite
        results = run_benchmark(
            dataset_name,
            embeddings,
            metadata_df,
            image_paths,
            id_col = id_col,
            label_col = label_col,
            outer_folds = fold_assignments,
            n_splits = outer_folds,
            random_state = evaluation_seed,
            group_col = GROUP_COL_MAP[dataset_name]
        )

        emb_array = np.asarray(embeddings)
        results["embedding_info"] = {
            "model_id": model_id,
            "dataset_protocol": dataloader.protocol_name,
            "embedding_dim": int(emb_array.shape[1]),
            "normalized": normalize_embeddings,
            "n_embeddings": int(emb_array.shape[0]),
            "n_evaluated_embeddings": int(manifest_info["n_samples"]),
            "source": source
        }
        results["reproducibility"] = manifest_info
        
        results["runtime"]["stages"]["dataset_loading"] = dataset_loading_seconds
        results["runtime"]["stages"]["embedding_extraction"] = embedding_seconds
        results["runtime"]["stages"]["total_dataset_run"] = perf_counter() - dataset_start

        # Save results
        results_path = os.path.join(run_folder, f"{dataset_name}.json")
        write_json(results, results_path)

        if run_random_embeddings:
            run_random_baseline(
                dataset_name,
                embedding_shape = emb_array.shape,
                metadata_df = metadata_df,
                image_paths = image_paths,
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
                group_col = GROUP_COL_MAP[dataset_name]
            )

        if run_label_prior:
            run_label_prior_baselines(
                dataset_name,
                metadata_df = metadata_df,
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
                metadata_df = metadata_df,
                image_paths = image_paths,
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
                group_col = GROUP_COL_MAP[dataset_name]
            )

        print(f"\n=== Dataset Complete ===\n\n")
    
    print(f"=== Benchmark Complete ===")
    print(f"Results saved to: {os.path.abspath(run_folder)}\n")

if __name__ == "__main__":
    main()
