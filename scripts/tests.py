"""
tests.py

This file contains the functions for testing the quality of the embeddings.
"""

import os
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Silence Optuna's extensive logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def prepare_data_multilabel(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col):
    """
    Prepares the data for adapters. Allows for multilabel data to be labeled appropriately.
    Embeddings are ordered to be "paired" up with their associated image and target.

    i.e. X[1343] is the embedding associated with the image whose target is y[1343]

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
    
    Returns:
        X : The embeddings computed by the model
        y : The target diagnosis(es)
        classes : The target classes in the dataset
        is_multilabel : Boolean flag for multilabel data
    """
    # Prepare image column for embedding DataFrame
    if dataset_name == "chexpert":
        image_names = [os.sep.join(path.split(os.sep)[-3:]) for path in image_paths]
    elif dataset_name == "cbis_ddsm":
        image_names = [os.sep.join(path.split(os.sep)[-2:]) for path in image_paths]
    else:
        image_names = [os.path.basename(path) for path in image_paths]

    emb = np.asarray(embeddings, dtype = np.float32)
    emb_df = pd.DataFrame(emb)
    emb_df[id_col] = np.asarray(image_names)

    # Create a DataFrame via merging to prepare samples and labels
    df = pd.merge(
        metadata_df[[id_col, label_col]],
        emb_df,
        on = id_col,
        how = "inner"
    ).dropna(subset = [label_col])

    X = df.drop(columns = [id_col, label_col]).values

    # Label type detection: multiclass v.s. multilabel
    is_multilabel = False
    first_label = df[label_col].iloc[0]

    is_vector = isinstance(first_label, (list, tuple, np.ndarray))

    if is_vector:
        label_matrix = np.stack(df[label_col].values)
        label_matrix = label_matrix.astype(int)
        row_sums = label_matrix.sum(axis = 1)

        if np.any(row_sums >= 2):
            is_multilabel = True

    if is_multilabel:
        # Stack the labels into a matrix
        y = np.vstack(df[label_col].to_numpy()).astype(int)
        classes = [f"Label {i}" for i in range(y.shape[1])]
    else:
        # Standard multiclass encoding
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].astype(str).to_numpy())
        classes = list(le.classes_)

    return X, y, classes, is_multilabel

def prepare_data_multiclass(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col):
    """
    Prepares the data for adapters. Treats multilabel data as multiclass data.
    Embeddings are ordered to be "paired" up with their associated image and target.

    i.e. X[1343] is the embedding associated with the image whose target is y[1343]

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
    
    Returns:
        X : The embeddings computed by the model
        y : The target diagnosis(es)
        classes : The target classes in the dataset
    """
    # Prepare image column for embedding DataFrame
    if dataset_name == "chexpert":
        image_names = [os.sep.join(path.split(os.sep)[-3:]) for path in image_paths]
    elif dataset_name == "cbis_ddsm":
        image_names = [os.sep.join(path.split(os.sep)[-2:]) for path in image_paths]
    else:
        image_names = [os.path.basename(path) for path in image_paths]

    emb = np.asarray(embeddings, dtype = np.float32)
    emb_df = pd.DataFrame(emb)
    emb_df[id_col] = np.asarray(image_names)

    # Create a DataFrame via merging to prepare samples and labels
    df = pd.merge(
        metadata_df[[id_col, label_col]],
        emb_df,
        on = id_col,
        how = "inner"
    ).dropna(subset = [label_col])

    X = df.drop(columns = [id_col, label_col]).values

    # Treat the exact label combination as the "class"
    # Convert list [1, 0, 1] -> string "[1, 0, 1]" so we can use LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str).to_numpy())
    classes = list(le.classes_)

    return X, y, classes


### Adapter test functions ###


def MLP_cv(dataset_name, embeddings, metadata_df, image_paths, id_col,
           label_col, n_splits = 5, random_state = 42, n_trials = 20):
    """
    Tests the embeddings on an MLP adapter.

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
        n_splits : Number of folds for train/test splits 
        random_state : Seed used for random operations
        n_trials : Number of trials used for Optuna
    
    Returns:
        summary : JSON-compatible summary of the MLP test
    """
    X, y, classes, is_multilabel = prepare_data_multilabel(dataset_name, embeddings, metadata_df,
                                                image_paths, id_col, label_col)

    if is_multilabel:
        cv = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        # "roc_auc" in sklearn defaults to macro average for multilabel in modern versions
        scoring_auc = "roc_auc"
    else:
        cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        scoring_auc = "roc_auc" if len(classes) == 2 else "roc_auc_ovr"
    
    print(f"--- Optimizing MLP with Optuna ({n_trials} trials) ---")

    def objective(trial):
        """
        Function used by Optuna to maximize MLP performance.
        """
        # 1. Suggest Hyperparameters
        hidden_layer_choice = trial.suggest_categorical("hidden_layers", ["small", "medium", "large"])
        if hidden_layer_choice == "small":
            layers = (64,)
        elif hidden_layer_choice == "medium":
            layers = (128, 64)
        else:
            layers = (256, 128, 64)

        lr_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log = True)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log = True)
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])

        # 2. Build Pipeline
        clf = MLPClassifier(
            hidden_layer_sizes = layers,
            activation = activation,
            learning_rate_init = lr_init,
            alpha = alpha,
            batch_size = 256,
            max_iter = 300, # Slightly lower for tuning speed
            early_stopping = True,
            n_iter_no_change = 10,
            random_state = random_state
        )
        
        pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", clf)])

        # 3. Fast Evaluation (accuracy is usually faster/stable for tuning, but AUC is better quality)
        # Using accuracy for speed in loop, or main_metric if feasible.
        scores = cross_val_score(pipeline, X, y, cv = 3, scoring = "roc_auc" if is_multilabel else scoring_auc, n_jobs = -1)
        return scores.mean()
    
     # Run Optimization
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = n_trials)

    # Reconstruct Best Model
    best_layers_map = {"small": (64,), "medium": (128, 64), "large": (256, 128, 64)}
    best_clf = MLPClassifier(
        hidden_layer_sizes = best_layers_map[study.best_params["hidden_layers"]],
        activation = study.best_params["activation"],
        learning_rate_init = study.best_params["learning_rate_init"],
        alpha = study.best_params["alpha"],
        batch_size = 256,
        max_iter = 500, # Full training
        early_stopping = True,
        random_state = random_state
    )

    final_pipe = Pipeline([("scaler", StandardScaler()), ("mlp", best_clf)])

    # use built-in scorer names
    scorers = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "roc_auc": scoring_auc
    }

    cv_res = cross_validate(final_pipe, X, y, cv = cv, scoring = scorers, n_jobs = -1, return_train_score = False)

    # Aggregate results (JSON compatible)
    summary = {
        "accuracy": [float(cv_res["test_accuracy"].mean()), float(cv_res["test_accuracy"].std())],
        "f1_weighted": [float(cv_res["test_f1_weighted"].mean()), float(cv_res["test_f1_weighted"].std())],
        "precision_weighted": [float(cv_res["test_precision_weighted"].mean()), float(cv_res["test_precision_weighted"].std())],
        "roc_auc": [float(cv_res["test_roc_auc"].mean()), float(cv_res["test_roc_auc"].std())],
        "classes": classes,
        "best_params": study.best_params
    }

    print(f"CV {n_splits}-fold results (mean ± std):")
    for k in ["accuracy", "f1_weighted", "precision_weighted", "roc_auc"]:
        m, s = summary[k]
        print(f"  {k:18s}: {m:.4f} ± {s:.4f}")

    return summary

def KNN_cv(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
           n_splits = 5, random_state = 42, n_trials = 15):
    """
    Tests the embeddings on a KNN adapter.

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
        n_splits : Number of folds for train/test splits 
        random_state : Seed used for random operations
        n_trials : Number of trials used for Optuna
    
    Returns:
        summary : JSON-compatible summary of the KNN test
    """
    X, y, classes, is_multilabel = prepare_data_multilabel(dataset_name, embeddings, metadata_df,
                                                image_paths, id_col, label_col)

    def auc_scorer(est, X, y):
        """
        Function to handle appropriate AUC scoring based on dataset type (multilabel vs multiclass). 

        Args:
            est : A KNeighborsClassifier object
            X : The embeddings computed by the model
            y : The target diagnosis(es)
        
        Returns:
            The appropriate AUC scorer
        """
        proba = est.predict_proba(X)
        
        if is_multilabel:
            # len(proba) == num of classes = L; p is an array of (N x 2), N == num of samples
            # np.transpose flips a list of (N x 1) arrays into a (N x L) array
            proba_stacked = np.transpose([p[:, 1] for p in proba])
            return roc_auc_score(y, proba_stacked, average = "macro")
        else:
            # Multiclass/Binary
            if proba.shape[1] == 2:
                return roc_auc_score(y, proba[:, 1])
            else:
                return roc_auc_score(y, proba, multi_class = "ovr", average = "macro")
    
    def objective(trial):
        """
        Function used by Optuna to maximize KNN performance.
        """
        k = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "manhattan"])
        
        clf = KNeighborsClassifier(n_neighbors = k, weights = weights, metric = metric)
        pipe = Pipeline([("norm", Normalizer(norm = "l2")), ("knn", clf)])
        
        # Use F1 weighted for tuning KNN (handling class imbalance better than accuracy)
        score = cross_val_score(pipe, X, y, cv = 3, scoring = "f1_weighted", n_jobs = -1).mean()
        return score

    print(f"--- Optimizing KNN with Optuna ({n_trials} trials) ---")
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = n_trials)

    # Final Evaluation with Best Params
    best_clf = KNeighborsClassifier(
        n_neighbors = study.best_params["n_neighbors"],
        weights = study.best_params["weights"],
        metric = study.best_params["metric"]
    )
    
    final_pipe = Pipeline([("norm", Normalizer(norm="l2")), ("knn", best_clf)])

    if is_multilabel:
        cv = KFold(n_splits = 5, shuffle = True, random_state = random_state)
    else:
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)

    scorers = {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average = "weighted", zero_division = 0),
        "precision_weighted": make_scorer(precision_score, average = "weighted", zero_division = 0),
        "roc_auc": auc_scorer
    }

    res = cross_validate(final_pipe, X, y, cv = cv, scoring = scorers, n_jobs = -1, return_train_score = False)

    # JSON compatible
    summary = {
        "best_k": study.best_params["n_neighbors"],
        "best_scores": {
            "accuracy": float(res["test_accuracy"].mean()),
            "f1_weighted": float(res["test_f1_weighted"].mean()),
            "precision_weighted": [float(np.mean(res["test_precision_weighted"])),
                                   float(np.std(res["test_precision_weighted"]))],
            "roc_auc": [float(res["test_roc_auc"].mean()), float(res["test_roc_auc"].std())]
        },
        "classes": classes,
        "best_params": study.best_params
    }

    print(f"KNN CV (n_splits={n_splits}) — best k = {summary['best_k']}")
    for k, v in summary["best_scores"].items():
        if isinstance(v, list):
            continue
        else:
            print(f"  {k:18s}: {v:.4f}")

    return summary

def logistic_regression_cv(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                           n_splits = 5, random_state = 42, n_trials = 15):
    """
    Tests the embeddings on a LR adapter.

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
        n_splits : Number of folds for train/test splits 
        random_state : Seed used for random operations
        n_trials : Number of trials used for Optuna
    
    Returns:
        summary : JSON-compatible summary of the LR test
    """
    X, y, classes, is_multilabel = prepare_data_multilabel(dataset_name, embeddings, metadata_df,
                                                image_paths, id_col, label_col)

    if is_multilabel:
        cv = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        scoring_auc = "roc_auc" # sklearn handles macro average
    else:
        cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        scoring_auc = "roc_auc" if len(classes) == 2 else "roc_auc_ovr"

    print(f"--- Optimizing Logistic Regression with Optuna ({n_trials} trials) ---")

    def objective(trial):
        """
        Function used by Optuna to maximize LR performance.
        """
        c_value = trial.suggest_float("C", 1e-3, 1e2, log = True)

        # FIXME: Edit as necessary
        if len(X) > 20000:
            solver = "saga"
        else:
            solver = "lbfgs"
        
        base_clf = LogisticRegression(C = c_value, max_iter = 2000, class_weight = "balanced", solver = solver)
        
        if is_multilabel:
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = base_clf
            
        pipe = Pipeline([("std", StandardScaler()), ("clf", clf)])
        
        # Use 3-fold for speed during tuning
        return cross_val_score(pipe, X, y, cv = 3, scoring = scoring_auc, n_jobs = -1).mean()
    
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = n_trials)

    # Final Model
    final_base = LogisticRegression(C=study.best_params["C"], max_iter = 3000, class_weight = "balanced")
    if is_multilabel:
        final_clf = MultiOutputClassifier(final_base)
    else:
        final_clf = final_base

    final_pipe = Pipeline([("std", StandardScaler()), ("clf", final_clf)])
    
    scorers = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "roc_auc": scoring_auc,
    }

    res = cross_validate(final_pipe, X, y, cv = cv, scoring = scorers, n_jobs = -1, return_train_score = False)

    # Summarize results (JSON compatible)
    summary = {
        "accuracy": [float(res["test_accuracy"].mean()), float(res["test_accuracy"].std())],
        "f1_weighted": [float(res["test_f1_weighted"].mean()), float(res["test_f1_weighted"].std())],
        "precision_weighted": [float(res["test_precision_weighted"].mean()), float(res["test_precision_weighted"].std())],
        "roc_auc": [float(res["test_roc_auc"].mean()), float(res["test_roc_auc"].std())],
        "classes": classes,
        "best_params": study.best_params 
    }

    # Print out the results
    print(f"Logistic Regression CV Results (n_splits={n_splits}):")
    for metric in ["accuracy", "f1_weighted", "precision_weighted", "roc_auc"]:
        mean, std = summary[metric]
        print(f"  {metric:18s}: {mean:.4f} ± {std:.4f}")

    return summary

def retrieval_eval(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                   ks = (1, 5, 10), normalize = True, per_class = False):
    """
    All-vs-all retrieval on embeddings using cosine similarity (via L2-normalization).
    Returns Recall@K and mAP. Queries from singleton classes are skipped for metrics.

    For Multilabel data, this test treats unique label vectors as distinct "classes"
    for the purpose of determining "Same Class" vs "Different Class" in retrieval.

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
        ks : Iterable of ints, K values for Recall@K
        normalize : Boolean flag to L2-normalize embeddings
        per_class : Boolean flag to return per-class Recall@K

    Returns:
        results : JSON-compatible summary of the recall test
    """
    X, y, classes = prepare_data_multiclass(dataset_name, embeddings, metadata_df, image_paths,
            id_col, label_col)

    message = ""
    for idx, k in enumerate(ks):
        message += f"{k}, " if idx != len(ks) - 1 else f"and {k}"

    print(f"--- Retrieval evaluation with Recall@{message} ---")

    # Normalize to make cosine == dot
    if normalize:
        X = Normalizer(norm = "l2").fit_transform(X)

    N = X.shape[0]

    # Cosine similarity and ranking
    S = X @ X.T                            # cosine similarity
    np.fill_diagonal(S, -np.inf)           # don't retrieve yourself
    ranks = np.argsort(-S, axis = 1)         # highest sim first

    # Mask out singleton classes
    counts = Counter(y.tolist())
    valid = np.array([counts[c] > 1 for c in y])  # queries with at least 1 other same-class item
    idx_valid = np.where(valid)[0]
    n_eval = int(valid.sum())

    if n_eval == 0:
        return {
            "n_total": N,
            "n_eval": 0,
            "recall_at_k": {int(k): np.nan for k in ks},
            "map": np.nan,
            "note": "No classes have more than one sample; retrieval undefined."
        }

    # Metrics helpers
    def recall_at_k_for_query(i, K):
        """
        Tests if the top K neighbors share the same label as the query.

        Args:
            i : The index of the query sample in the dataset
            K : The number of neighbors to consider
        Returns:
            True if any of the K neighbors share the same label as the query
        """
        # true if any of top-K neighbors share the label
        topk = ranks[i, :K]
        return np.any(y[topk] == y[i])

    def average_precision_for_query(i):
        """
        Computes the average precision for a single query.

        Args:
            i : The index of the query sample in the dataset
        
        Returns:
            The average precision for the query
        """
        rel = (y[ranks[i]] == y[i])   # boolean vector over all candidates
        if not np.any(rel):
            return np.nan  # shouldn't happen for valid queries, but safe-guard
        # precision at each rank where rel is True
        hits = np.flatnonzero(rel)        # positions where we hit the class
        precisions = []
        for r in hits:
            # ranks are 0-indexed; +1 is the rank position
            top_r = ranks[i, :r + 1]
            precisions.append(np.mean(y[top_r] == y[i]))
        # AP = mean of precisions at relevant ranks
        return float(np.mean(precisions))

    # Compute Recall@K and mAP
    recall_at_k = {}
    for K in ks:
        vals = [recall_at_k_for_query(i, K) for i in idx_valid]
        recall_at_k[int(K)] = float(np.mean(vals))

    aps = [average_precision_for_query(i) for i in idx_valid]
    mAP = float(np.nanmean(aps)) if len(aps) else np.nan

    # JSON compatible
    results = {
        "n_total": N,
        "n_eval": n_eval,
        "recall_at_k": recall_at_k,
        "map": mAP,
        "classes": classes
    }

    # Optional per-class Recall@K
    if per_class:
        per_cls = {}
        for c_idx, c_name in enumerate(classes):
            idx_c = np.where((y == c_idx) & valid)[0]
            if len(idx_c) == 0:
                per_cls[c_name] = {int(K): np.nan for K in ks}
                continue
            per_cls[c_name] = {
                int(K): float(np.mean([recall_at_k_for_query(i, K) for i in idx_c])) for K in ks
            }
        results["recall_at_k_per_class"] = per_cls

    # Print summary
    print(f"Retrieval (all-vs-all, cosine) — evaluated {n_eval}/{N} queries (non-singleton classes).")
    for K in ks:
        print(f"  Recall@{K}: {recall_at_k[int(K)]:.4f}")
    print(f"  mAP      : {mAP:.4f}")

    return results

def clustering_eval(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                    k_range = range(2, 15), random_state = 42, compute_silhouette = True):
    """
    Run KMeans for multiple numbers of clusters (k_range) and compute ARI/NMI (and silhouette if requested).

    For Multilabel data, Ground Truth (GT) is defined as the unique 
    vector combination (stringified) for ARI/NMI calculation.

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image
        k_range : List of the number of potential clusters
        random_state : Seed used for random operations
        compute_silhouette : Boolean flag for computing silhouette score
    Returns:
        results_dict : JSON-compatible summary of the clustering test
    """
    # classes is unused by design
    X, y, classes = prepare_data_multiclass(dataset_name, embeddings, metadata_df, image_paths,
            id_col, label_col)
    
    print(f"--- Clustering evaluation with k_min={min(k_range)} through k_max={max(k_range)} ---")

    # Run KMeans for each k
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters = k, n_init = "auto", random_state = random_state)
        cluster_labels = kmeans.fit_predict(X)

        ari = adjusted_rand_score(y, cluster_labels)
        nmi = normalized_mutual_info_score(y, cluster_labels)

        if compute_silhouette and len(np.unique(cluster_labels)) > 1:
            sil = silhouette_score(X, cluster_labels)
        else:
            sil = np.nan

        results.append({"k": k, "ARI": ari, "NMI": nmi, "Silhouette": sil})

    results_df = pd.DataFrame(results)

    # Display summary
    best_k_ari = results_df.loc[results_df["ARI"].idxmax()]
    best_k_nmi = results_df.loc[results_df["NMI"].idxmax()]
    best_k_sil = results_df.loc[results_df["Silhouette"].idxmax()] if compute_silhouette else None

    print("KMeans Clustering Evaluation (variable k):")
    print(results_df.round(4))
    print()
    print(f"Best ARI:  k={int(best_k_ari['k'])}, score={best_k_ari['ARI']:.4f}")
    print(f"Best NMI:  k={int(best_k_nmi['k'])}, score={best_k_nmi['NMI']:.4f}")
    if compute_silhouette:
        print(f"Best Silhouette: k = {int(best_k_sil['k'])}, score = {best_k_sil['Silhouette']:.4f}")

    # JSON compatible
    results_dict = {
        "best_ari": [int(best_k_ari['k']), best_k_ari['ARI']],
        "best_nmi": [int(best_k_nmi['k']), best_k_nmi['NMI']]
    }

    if compute_silhouette:
        results_dict["best_silhouette"] = [int(best_k_sil['k']), best_k_sil['Silhouette']]

    return results_dict