"""
main.py

Executes the testbench.
"""

import os
import gc
import sys
import json
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from scripts.model_interface import TorchvisionBackend, HuggingFaceVisionBackend
import torch
import tensorflow as tf

PROHIBITED_CHARS = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", "_"]

# Format: (id_col, label_col)
DATASET_COL_MAP = {"pad_ufes": ("img_id", "diagnostic"), "cbis_ddsm": ("image file path", "pathology"),
                   "odir": ("filename", "target"), "ham10000": ("image_id", "dx"),
                   "chexpert": ("Path", "Diagnosis")}

ID_COL_IDX = 0
LABEL_COL_IDX = 1

# Make the current directory (for the subprocess) relative to the testbench
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

python = sys.executable

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
    Runs the testbench application.
    """
    parser = argparse.ArgumentParser(description = "Run medical FM embedding benchmark.")
    parser.add_argument("--config", required = True, help = "Path to benchmark config JSON file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    model_name = clean_name(model_id, "-")
    output_dir = cfg.get("output_dir", f".{os.sep}results")

    datasets = cfg["dataset"]["datasets"]
    batch_size = cfg["dataset"].get("batch_size", 32)
    shuffle = cfg["dataset"].get("shuffle", False)

    normalize_embeddings = cfg["embeddings"].get("normalize", True)
    cache_embeddings = cfg["embeddings"].get("cache", False)

    os.makedirs(output_dir, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # This dictates the path for the results directory for a model
    run_folder = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_folder)

    write_json(cfg, os.path.join(run_folder, "config_used.json"))

    # Load dependecies once JSON file is parsed
    from scripts.dataloading import load_dataset
    from scripts.models import build_backend
    from scripts.extraction import extract_embeddings
    from scripts.run_benchmark import run_benchmark

    print(f"\n=== Running Benchmark: {model_name} ===")
    print(f"Model: {model_id}")
    print(f"Output folder: {os.path.abspath(run_folder)}\n")

    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")

        id_col = DATASET_COL_MAP[dataset_name][ID_COL_IDX]
        label_col = DATASET_COL_MAP[dataset_name][LABEL_COL_IDX]

        # Build backend
        backend = build_backend(model_id)

        # Load dataset with the backend transform (Torchvision models)
        # HF backend returns None here; dataset will return PIL images (still works)
        transform = backend.get_transform()

        dataloader, metadata_df = load_dataset(
            dataset_name,
            transform = transform,
            batch_size = batch_size,
            shuffle = shuffle
        )

        # FIXME: Can change if using better compute resources
        train_size = 5000
        
        if len(metadata_df) > train_size:
            metadata_df, _ = train_test_split(metadata_df, train_size = train_size, \
            stratify = metadata_df[label_col], random_state = 42)
            print(f"Dataset stratified and downsampled to {train_size} examples." + \
                  " Adjust as needed by editing main.py.")

        # Extract embeddings
        embeddings, image_paths = extract_embeddings(
            dataloader,
            backend,
            normalize = normalize_embeddings,
            cache = cache_embeddings
        )

        # Run benchmark suite
        results = run_benchmark(
            dataset_name,
            embeddings,
            metadata_df,
            image_paths,
            id_col = id_col,
            label_col = label_col
        )

        # Save results
        results_path = os.path.join(run_folder, f"{dataset_name}.json")
        write_json(results, results_path)

        # Clean up RAM
        gc.collect()

        # FIXME: Clean up VRAM (GPUs)
        if isinstance(backend, (TorchvisionBackend, HuggingFaceVisionBackend)):
            torch.cuda.empty_cache()
        else:
            tf.keras.backend.clear_session()

        print(f"\n=== Dataset Complete ===\n\n")
    
    print(f"=== Benchmark Complete ===")
    print(f"Results saved to: {os.path.abspath(run_folder)}\n")

if __name__ == "__main__":
    main()