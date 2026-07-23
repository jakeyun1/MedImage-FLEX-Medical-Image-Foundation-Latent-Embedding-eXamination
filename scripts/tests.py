"""
tests.py

This file contains the functions for testing the quality of the embeddings.
"""

import os
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
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


def _cv_splitter(is_multilabel, n_splits, random_state):
    if is_multilabel:
        return KFold(n_splits = n_splits, shuffle = True, random_state = random_state)

    return StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)

def _inner_cv_splitter(is_multilabel, y, n_splits, random_state):
    if is_multilabel:
        return _cv_splitter(is_multilabel, n_splits, random_state)

    _, counts = np.unique(y, return_counts = True)
    safe_splits = min(n_splits, int(counts.min()))
    if safe_splits < 2:
        raise ValueError("At least two samples per class are required for inner cross-validation.")

    return _cv_splitter(is_multilabel, safe_splits, random_state)

def _split_iter(cv, X, y, is_multilabel):
    if is_multilabel:
        return cv.split(X)

    return cv.split(X, y)

def _extract_positive_proba(proba):
    if isinstance(proba, list):
        return np.transpose([p[:, 1] for p in proba])

    return proba

def _dataset_info(dataset_name, y, classes, label_type):
    if label_type == "multilabel":
        counts = np.asarray(y).sum(axis = 0)
    else:
        counts = np.bincount(np.asarray(y), minlength = len(classes))

    return {
        "dataset_name": dataset_name,
        "n_samples": int(len(y)),
        "n_classes": int(len(classes)),
        "classes": [str(cls) for cls in classes],
        "class_counts": {str(cls): int(counts[idx]) for idx, cls in enumerate(classes)},
        "label_type": label_type
    }

def _per_class_metrics(y_true, y_pred, classes, is_multilabel):
    labels = None if is_multilabel else np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels = labels,
        average = None,
        zero_division = 0
    )

    if is_multilabel:
        support = np.asarray(y_true).sum(axis = 0)

    return {
        str(cls): {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx])
        }
        for idx, cls in enumerate(classes)
    }

def _evaluate_classifier(pipe, X_test, y_test, is_multilabel, y_pred = None):
    if y_pred is None:
        y_pred = pipe.predict(X_test)

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average = "weighted", zero_division = 0),
        "precision_weighted": precision_score(y_test, y_pred, average = "weighted", zero_division = 0),
    }

    proba = _extract_positive_proba(pipe.predict_proba(X_test))

    if is_multilabel:
        scores["roc_auc"] = roc_auc_score(y_test, proba, average = "macro")
    elif proba.shape[1] == 2:
        scores["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
    else:
        scores["roc_auc"] = roc_auc_score(y_test, proba, multi_class = "ovr", average = "macro")

    return scores

def _fold_summary(fold_idx, scores):
    fold_result = {"fold": int(fold_idx)}
    fold_result.update({metric: float(score) for metric, score in scores.items()})
    return fold_result

def _summarize_fold_scores(fold_scores):
    return {
        metric: [float(np.mean(values)), float(np.std(values))]
        for metric, values in fold_scores.items()
    }

def _most_common_params(params_by_fold):
    counts = Counter(tuple(sorted(params.items())) for params in params_by_fold)
    return dict(counts.most_common(1)[0][0])


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

    cv = _cv_splitter(is_multilabel, n_splits, random_state)
    scoring_auc = "roc_auc" if is_multilabel or len(classes) == 2 else "roc_auc_ovr"

    print(f"--- Optimizing MLP with Optuna ({n_trials} trials) ---")

    def build_pipe(params, max_iter):
        best_layers_map = {"small": (64,), "medium": (128, 64), "large": (256, 128, 64)}
        clf = MLPClassifier(
            hidden_layer_sizes = best_layers_map[params["hidden_layers"]],
            activation = params["activation"],
            learning_rate_init = params["learning_rate_init"],
            alpha = params["alpha"],
            batch_size = 256,
            max_iter = max_iter,
            early_stopping = True,
            n_iter_no_change = 10,
            random_state = random_state
        )

        return Pipeline([("scaler", StandardScaler()), ("mlp", clf)])

    fold_scores = {"accuracy": [], "f1_weighted": [], "precision_weighted": [], "roc_auc": []}
    fold_score_rows = []
    best_params_by_fold = []
    inner_folds_by_fold = []
    y_true_parts = []
    y_pred_parts = []

    for fold_idx, (train_idx, test_idx) in enumerate(_split_iter(cv, X, y, is_multilabel), start = 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_cv = _inner_cv_splitter(is_multilabel, y_train, 3, random_state)
        inner_folds_by_fold.append(int(inner_cv.get_n_splits()))

        def objective(trial):
            """
            Function used by Optuna to maximize MLP performance.
            """
            # 1. Suggest Hyperparameters
            params = {
                "hidden_layers": trial.suggest_categorical("hidden_layers", ["small", "medium", "large"]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log = True),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log = True),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"])
            }

            pipeline = build_pipe(params, max_iter = 300)
            scores = cross_val_score(
                pipeline, X_train, y_train,
                cv = inner_cv,
                scoring = scoring_auc,
                n_jobs = -1
            )
            return scores.mean()

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials = n_trials)
        best_params_by_fold.append(study.best_params)

        final_pipe = build_pipe(study.best_params, max_iter = 500)
        final_pipe.fit(X_train, y_train)
        y_pred = final_pipe.predict(X_test)
        scores = _evaluate_classifier(final_pipe, X_test, y_test, is_multilabel, y_pred = y_pred)

        for metric, score in scores.items():
            fold_scores[metric].append(float(score))

        fold_score_rows.append(_fold_summary(fold_idx, scores))
        y_true_parts.append(y_test)
        y_pred_parts.append(y_pred)

        print(f"  Fold {fold_idx}/{n_splits} best params: {study.best_params}")

    summary = _summarize_fold_scores(fold_scores)
    summary["classes"] = classes
    summary["best_params"] = _most_common_params(best_params_by_fold)
    summary["best_params_by_fold"] = best_params_by_fold
    summary["fold_scores"] = fold_score_rows
    summary["per_class"] = _per_class_metrics(
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        classes,
        is_multilabel
    )
    summary["evaluation_protocol"] = {
        "adapter": "mlp",
        "cv_type": "nested",
        "outer_folds": int(n_splits),
        "inner_folds": inner_folds_by_fold,
        "random_state": int(random_state),
        "tuning_metric": scoring_auc,
        "n_trials": int(n_trials)
    }

    dataset_info = _dataset_info(
            dataset_name,
            y,
            classes,
            "multilabel" if is_multilabel else "multiclass"
    )

    print(f"Nested CV {n_splits}-fold results (mean ± std):")
    for k in ["accuracy", "f1_weighted", "precision_weighted", "roc_auc"]:
        m, s = summary[k]
        print(f"  {k:18s}: {m:.4f} ± {s:.4f}")

    return summary, dataset_info

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

    def build_pipe(params):
        clf = KNeighborsClassifier(
            n_neighbors = params["n_neighbors"],
            weights = params["weights"],
            metric = params["metric"]
        )
        pipe = Pipeline([("norm", Normalizer(norm = "l2")), ("knn", clf)])

        return pipe

    print(f"--- Optimizing KNN with Optuna ({n_trials} trials) ---")
    cv = _cv_splitter(is_multilabel, n_splits, random_state)
    fold_scores = {"accuracy": [], "f1_weighted": [], "precision_weighted": [], "roc_auc": []}
    fold_score_rows = []
    best_params_by_fold = []
    inner_folds_by_fold = []
    y_true_parts = []
    y_pred_parts = []

    for fold_idx, (train_idx, test_idx) in enumerate(_split_iter(cv, X, y, is_multilabel), start = 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_cv = _inner_cv_splitter(is_multilabel, y_train, 3, random_state)
        inner_folds_by_fold.append(int(inner_cv.get_n_splits()))

        def objective(trial):
            """
            Function used by Optuna to maximize KNN performance.
            """
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "cosine", "manhattan"])
            }

            # Use F1 weighted for tuning KNN (handling class imbalance better than accuracy)
            score = cross_val_score(
                build_pipe(params), X_train, y_train,
                cv = inner_cv,
                scoring = "f1_weighted",
                n_jobs = -1
            ).mean()
            return score

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials = n_trials)
        best_params_by_fold.append(study.best_params)

        final_pipe = build_pipe(study.best_params)
        final_pipe.fit(X_train, y_train)
        y_pred = final_pipe.predict(X_test)
        scores = _evaluate_classifier(final_pipe, X_test, y_test, is_multilabel, y_pred = y_pred)

        for metric, score in scores.items():
            fold_scores[metric].append(float(score))

        fold_score_rows.append(_fold_summary(fold_idx, scores))
        y_true_parts.append(y_test)
        y_pred_parts.append(y_pred)

        print(f"  Fold {fold_idx}/{n_splits} best params: {study.best_params}")

    score_summary = _summarize_fold_scores(fold_scores)
    best_params = _most_common_params(best_params_by_fold)

    # JSON compatible
    summary = {
        "best_k": best_params["n_neighbors"],
        "best_scores": {
            "accuracy": score_summary["accuracy"][0],
            "f1_weighted": score_summary["f1_weighted"][0],
            "precision_weighted": score_summary["precision_weighted"],
            "roc_auc": score_summary["roc_auc"]
        },
        "classes": classes,
        "best_params": best_params,
        "best_params_by_fold": best_params_by_fold,
        "fold_scores": fold_score_rows,
        "per_class": _per_class_metrics(
            np.concatenate(y_true_parts),
            np.concatenate(y_pred_parts),
            classes,
            is_multilabel
        ),
        "evaluation_protocol": {
            "adapter": "knn",
            "cv_type": "nested",
            "outer_folds": int(n_splits),
            "inner_folds": inner_folds_by_fold,
            "random_state": int(random_state),
            "tuning_metric": "f1_weighted",
            "n_trials": int(n_trials)
        }
    }

    print(f"KNN nested CV (n_splits={n_splits}) — most common best k = {summary['best_k']}")
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

    cv = _cv_splitter(is_multilabel, n_splits, random_state)
    scoring_auc = "roc_auc" if is_multilabel or len(classes) == 2 else "roc_auc_ovr"

    print(f"--- Optimizing Logistic Regression with Optuna ({n_trials} trials) ---")

    def build_pipe(params, n_rows, max_iter):
        if n_rows > 20000:
            solver = "saga"
        else:
            solver = "lbfgs"

        base_clf = LogisticRegression(
            C = params["C"],
            max_iter = max_iter,
            class_weight = "balanced",
            solver = solver
        )

        if is_multilabel:
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = base_clf

        return Pipeline([("std", StandardScaler()), ("clf", clf)])

    fold_scores = {"accuracy": [], "f1_weighted": [], "precision_weighted": [], "roc_auc": []}
    fold_score_rows = []
    best_params_by_fold = []
    inner_folds_by_fold = []
    y_true_parts = []
    y_pred_parts = []

    for fold_idx, (train_idx, test_idx) in enumerate(_split_iter(cv, X, y, is_multilabel), start = 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_cv = _inner_cv_splitter(is_multilabel, y_train, 3, random_state)
        inner_folds_by_fold.append(int(inner_cv.get_n_splits()))

        def objective(trial):
            """
            Function used by Optuna to maximize LR performance.
            """
            params = {"C": trial.suggest_float("C", 1e-3, 10, log = True)}

            # Use 3-fold inner CV for tuning without touching the outer test fold.
            return cross_val_score(
                build_pipe(params, len(X_train), max_iter = 2000),
                X_train,
                y_train,
                cv = inner_cv,
                scoring = scoring_auc,
                n_jobs = -1
            ).mean()

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials = n_trials)
        best_params_by_fold.append(study.best_params)

        final_pipe = build_pipe(study.best_params, len(X_train), max_iter = 3000)
        final_pipe.fit(X_train, y_train)
        y_pred = final_pipe.predict(X_test)
        scores = _evaluate_classifier(final_pipe, X_test, y_test, is_multilabel, y_pred = y_pred)

        for metric, score in scores.items():
            fold_scores[metric].append(float(score))

        fold_score_rows.append(_fold_summary(fold_idx, scores))
        y_true_parts.append(y_test)
        y_pred_parts.append(y_pred)

        print(f"  Fold {fold_idx}/{n_splits} best params: {study.best_params}")

    summary = _summarize_fold_scores(fold_scores)
    summary["classes"] = classes
    summary["best_params"] = _most_common_params(best_params_by_fold)
    summary["best_params_by_fold"] = best_params_by_fold
    summary["fold_scores"] = fold_score_rows
    summary["per_class"] = _per_class_metrics(
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        classes,
        is_multilabel
    )
    summary["evaluation_protocol"] = {
        "adapter": "logistic_regression",
        "cv_type": "nested",
        "outer_folds": int(n_splits),
        "inner_folds": inner_folds_by_fold,
        "random_state": int(random_state),
        "tuning_metric": scoring_auc,
        "n_trials": int(n_trials)
    }

    # Print out the results
    print(f"Logistic Regression Nested CV Results (n_splits={n_splits}):")
    for metric in ["accuracy", "f1_weighted", "precision_weighted", "roc_auc"]:
        mean, std = summary[metric]
        print(f"  {metric:18s}: {mean:.4f} ± {std:.4f}")

    return summary

def retrieval_eval(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                   ks = (1, 5, 10), normalize = True, per_class = False,
                   bootstrap = True, n_bootstrap = 1000, ci = 95, random_state = 42):
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
        bootstrap : Boolean flag to return bootstrap confidence intervals
        n_bootstrap : Number of bootstrap resamples over evaluated queries
        ci : Confidence interval width
        random_state : Seed used for bootstrap resampling

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
            "evaluation_protocol": {
                "type": "all_vs_all_retrieval",
                "ks": [int(k) for k in ks],
                "similarity": "cosine",
                "normalized": normalize,
                "bootstrap": bootstrap,
                "n_bootstrap": int(n_bootstrap),
                "ci": int(ci),
                "random_state": int(random_state),
                "tuned": False
            },
            "protocol": {"similarity": "cosine", "normalized": normalize, "tuned": False},
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

    # Compute query-level metrics, then aggregate.
    per_query = []
    for i in idx_valid:
        recall_values = {int(K): bool(recall_at_k_for_query(i, K)) for K in ks}
        per_query.append({
            "query_index": int(i),
            "class": str(classes[y[i]]),
            "average_precision": average_precision_for_query(i),
            "recall_at_k": recall_values
        })

    recall_at_k = {
        int(K): float(np.mean([q["recall_at_k"][int(K)] for q in per_query]))
        for K in ks
    }
    aps = np.array([q["average_precision"] for q in per_query], dtype = float)
    mAP = float(np.nanmean(aps)) if len(aps) else np.nan

    # JSON compatible
    results = {
        "n_total": N,
        "n_eval": n_eval,
        "recall_at_k": recall_at_k,
        "map": mAP,
        "classes": classes,
        "evaluation_protocol": {
            "type": "all_vs_all_retrieval",
            "ks": [int(k) for k in ks],
            "similarity": "cosine",
            "normalized": normalize,
            "bootstrap": bootstrap,
            "n_bootstrap": int(n_bootstrap),
            "ci": int(ci),
            "random_state": int(random_state),
            "tuned": False
        },
        "protocol": {"similarity": "cosine", "normalized": normalize, "tuned": False}
    }

    if bootstrap:
        rng = np.random.default_rng(random_state)
        alpha = (100 - ci) / 2
        boot_recall = {int(K): [] for K in ks}
        boot_map = []

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, len(per_query), size = len(per_query))
            sample = [per_query[i] for i in sample_idx]
            for K in ks:
                boot_recall[int(K)].append(np.mean([q["recall_at_k"][int(K)] for q in sample]))
            boot_map.append(np.nanmean([q["average_precision"] for q in sample]))

        results["confidence_intervals"] = {
            "recall_at_k": {
                int(K): [
                    float(np.nanpercentile(boot_recall[int(K)], alpha)),
                    float(np.nanpercentile(boot_recall[int(K)], 100 - alpha))
                ]
                for K in ks
            },
            "map": [
                float(np.nanpercentile(boot_map, alpha)),
                float(np.nanpercentile(boot_map, 100 - alpha))
            ]
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
    X, y, classes = prepare_data_multiclass(dataset_name, embeddings, metadata_df, image_paths,
            id_col, label_col)

    class_count_k = len(classes)
    eval_k_values = sorted(set(k_range) | {class_count_k})

    print(f"--- Clustering evaluation with k_min={min(eval_k_values)} through k_max={max(eval_k_values)} ---")

    # Run KMeans for each k
    results = []
    for k in eval_k_values:
        kmeans = KMeans(n_clusters = k, n_init = "auto", random_state = random_state)
        cluster_labels = kmeans.fit_predict(X)

        ari = adjusted_rand_score(y, cluster_labels)
        nmi = normalized_mutual_info_score(y, cluster_labels)

        if compute_silhouette and 1 < len(np.unique(cluster_labels)) < X.shape[0]:
            sil = silhouette_score(X, cluster_labels)
        else:
            sil = np.nan

        results.append({"k": k, "ARI": ari, "NMI": nmi, "Silhouette": sil})

    results_df = pd.DataFrame(results)

    def row_summary(row):
        return {
            "k": int(row["k"]),
            "ARI": float(row["ARI"]),
            "NMI": float(row["NMI"]),
            "Silhouette": float(row["Silhouette"])
        }

    # Display summary
    best_k_ari = results_df.loc[results_df["ARI"].idxmax()]
    best_k_nmi = results_df.loc[results_df["NMI"].idxmax()]
    best_k_sil = results_df.loc[results_df["Silhouette"].idxmax()] if compute_silhouette else None
    class_count_row = results_df.loc[results_df["k"] == class_count_k].iloc[0]

    print("KMeans Clustering Evaluation (variable k):")
    print(results_df.round(4))
    print()
    print(f"Best ARI:  k={int(best_k_ari['k'])}, score={best_k_ari['ARI']:.4f}")
    print(f"Best NMI:  k={int(best_k_nmi['k'])}, score={best_k_nmi['NMI']:.4f}")
    print(f"Class-count k: k={class_count_k}, NMI={class_count_row['NMI']:.4f}")
    if compute_silhouette:
        print(f"Best Silhouette: k = {int(best_k_sil['k'])}, score = {best_k_sil['Silhouette']:.4f}")

    # JSON compatible
    results_dict = {
        "best_ari": [int(best_k_ari['k']), float(best_k_ari['ARI'])],
        "best_nmi": [int(best_k_nmi['k']), float(best_k_nmi['NMI'])],
        "oracle_best_ari": row_summary(best_k_ari),
        "oracle_best_nmi": row_summary(best_k_nmi),
        "class_count_k": row_summary(class_count_row),
        "k_sweep": [row_summary(row) for _, row in results_df.iterrows()],
        "evaluation_protocol": {
            "algorithm": "kmeans",
            "k_range": [int(k) for k in eval_k_values],
            "primary_k_rule": "number_of_classes",
            "random_state": int(random_state),
            "compute_silhouette": compute_silhouette,
            "label_tuned_primary": False
        }
    }

    if compute_silhouette:
        results_dict["best_silhouette"] = [int(best_k_sil['k']), float(best_k_sil['Silhouette'])]
        results_dict["silhouette_selected_k"] = row_summary(best_k_sil)

    if dataset_name == "chexpert":
        results_dict["note"] = ("For CheXpert, clustering classes are unique multilabel "
                                "combinations and should be interpreted as exploratory.")

    return results_dict
