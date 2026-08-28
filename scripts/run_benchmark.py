"""
run_benchmark.py

This file executes the testbench.
"""

from time import perf_counter
from scripts.tests import *

def timed_call(fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - start

def run_benchmark(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                  outer_folds = None, n_splits = 5, random_state = 42,
                  group_col = None, sample_ids = None):
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
        sample_ids = sample_ids
    )
    print(f"Completed MLP CV benchmark on {dataset_name}.\n")

    mlp_summary, dataset_info = mlp_tuple

    # KNN
    knn_summary, knn_seconds = timed_call(
        KNN_cv, dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, n_splits = n_splits,
        random_state = random_state, outer_folds = outer_folds, group_col = group_col,
        sample_ids = sample_ids
    )

    print(f"Completed KNN CV benchmark on {dataset_name}.\n")

    # LR
    logreg_summary, logreg_seconds = timed_call(
        logistic_regression_cv, dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, n_splits = n_splits,
        random_state = random_state, outer_folds = outer_folds, group_col = group_col,
        sample_ids = sample_ids
    )
    print(f"Completed Logistic Regression CV benchmarks on {dataset_name}.\n")

    # Retrieval
    ret_results, retrieval_seconds = timed_call(
        retrieval_eval, dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, ks = (1,5,10), per_class=True,
        random_state = random_state, group_col = group_col,
        sample_ids = sample_ids
    )
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
        "result_schema_version": 2,
        "dataset_info": dataset_info,
        "mlp_cv": mlp_summary,
        "knn_cv": knn_summary,
        "logreg_cv": logreg_summary,
        "retrieval": ret_results,
        "clustering": clustering_results,
        "runtime": {
            "unit": "seconds",
            "timer": "time.perf_counter",
            "scope": "single_process_wall_clock",
            "stages": {
                "mlp_cv": float(mlp_seconds),
                "knn_cv": float(knn_seconds),
                "logreg_cv": float(logreg_seconds),
                "retrieval": float(retrieval_seconds),
                "clustering": float(clustering_seconds)
            }
        }
    }

    # Results are ready to be formatted into a JSON file (json.dump)
    return results
