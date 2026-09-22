"""
run_benchmark.py

This file executes the testbench.
"""

import os
from time import perf_counter

from scripts.oof import OOF_SCHEMA_VERSION, write_oof_artifact
from scripts.retrieval_artifacts import (
    RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
    write_retrieval_artifact,
)
from scripts.tests import *

def timed_call(fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - start

def run_benchmark(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                  outer_folds = None, n_splits = 5, random_state = 42,
                  group_col = None, sample_ids = None, oof_output_dir = None,
                  oof_path_prefix = None, retrieval_output_dir = None,
                  retrieval_path_prefix = None):
    """
    Runs the testbench and returns the results for all adapters (for a given model on a given dataset).

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image  
    
    Returns:
        results : A map containing the formatted results (JSON-ready) for each adapter
    """
    # MLP
    mlp_tuple, mlp_seconds = timed_call(
        MLP_cv, dataset_name, embeddings, metadata_df,
        image_paths, id_col = id_col, label_col = label_col, n_splits = n_splits,
        random_state = random_state, outer_folds = outer_folds, group_col = group_col,
        sample_ids = sample_ids, return_oof = True
    )
    print(f"Completed MLP CV benchmark on {dataset_name}.\n")

    mlp_summary, dataset_info, mlp_oof = mlp_tuple

    # KNN
    knn_tuple, knn_seconds = timed_call(
        KNN_cv, dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, n_splits = n_splits,
        random_state = random_state, outer_folds = outer_folds, group_col = group_col,
        sample_ids = sample_ids, return_oof = True
    )
    knn_summary, knn_oof = knn_tuple

    print(f"Completed KNN CV benchmark on {dataset_name}.\n")

    # LR
    logreg_tuple, logreg_seconds = timed_call(
        logistic_regression_cv, dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, n_splits = n_splits,
        random_state = random_state, outer_folds = outer_folds, group_col = group_col,
        sample_ids = sample_ids, return_oof = True
    )
    logreg_summary, logreg_oof = logreg_tuple
    print(f"Completed Logistic Regression CV benchmarks on {dataset_name}.\n")

    oof_started = perf_counter()
    oof_metadata = {
        "schema_version": OOF_SCHEMA_VERSION,
        "enabled": oof_output_dir is not None,
        "artifacts": {},
    }
    if oof_output_dir is not None:
        for adapter, artifact in (
            ("mlp", mlp_oof),
            ("knn", knn_oof),
            ("logistic_regression", logreg_oof),
        ):
            filename = f"{adapter}.npz"
            relative_path = (
                os.path.join(oof_path_prefix, filename)
                if oof_path_prefix is not None else None
            )
            oof_metadata["artifacts"][adapter] = write_oof_artifact(
                artifact,
                oof_output_dir,
                filename,
                relative_path=relative_path,
            )
    oof_seconds = perf_counter() - oof_started

    # Retrieval
    retrieval_tuple, retrieval_seconds = timed_call(
        retrieval_eval, dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, ks = (1,5,10), per_class=True,
        random_state = random_state, group_col = group_col,
        sample_ids = sample_ids, return_artifact = True
    )
    ret_results, retrieval_artifact = retrieval_tuple
    retrieval_artifact_started = perf_counter()
    retrieval_artifact_metadata = {
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "enabled": retrieval_output_dir is not None,
        "artifact": None,
    }
    if retrieval_output_dir is not None:
        filename = "queries.npz"
        relative_path = (
            os.path.join(retrieval_path_prefix, filename)
            if retrieval_path_prefix is not None else None
        )
        retrieval_artifact_metadata["artifact"] = write_retrieval_artifact(
            retrieval_artifact,
            ret_results,
            retrieval_output_dir,
            filename=filename,
            relative_path=relative_path,
        )
    retrieval_artifact_seconds = perf_counter() - retrieval_artifact_started
    print(f"Completed retrieval evaluation on {dataset_name}.\n")

    # Clustering
    clustering_results, clustering_seconds = timed_call(
        clustering_eval,
        dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col,
        k_range = range(2, 12), random_state = random_state,
        sample_ids = sample_ids
    )
    print(f"Completed clustering evaluation on {dataset_name}.\n\n")

    # Compile the results
    results = {
        "result_schema_version": 4,
        "dataset_info": dataset_info,
        "mlp_cv": mlp_summary,
        "knn_cv": knn_summary,
        "logreg_cv": logreg_summary,
        "oof_predictions": oof_metadata,
        "retrieval": ret_results,
        "retrieval_queries": retrieval_artifact_metadata,
        "clustering": clustering_results,
        "runtime": {
            "unit": "seconds",
            "timer": "time.perf_counter",
            "scope": "single_process_wall_clock",
            "stages": {
                "mlp_cv": float(mlp_seconds),
                "knn_cv": float(knn_seconds),
                "logreg_cv": float(logreg_seconds),
                "oof_serialization": float(oof_seconds),
                "retrieval": float(retrieval_seconds),
                "retrieval_serialization": float(retrieval_artifact_seconds),
                "clustering": float(clustering_seconds)
            }
        }
    }

    # Results are ready to be formatted into a JSON file (json.dump)
    return results
