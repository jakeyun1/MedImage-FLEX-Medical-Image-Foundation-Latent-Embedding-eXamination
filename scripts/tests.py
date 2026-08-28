"""
tests.py

This file contains the functions for testing the quality of the embeddings.
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from scripts.dataset_contracts import DATASET_CONTRACTS, get_dataset_contract

# Silence Optuna's extensive logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

CLASSIFICATION_TUNING_METRIC = "f1_macro"

MULTILABEL_CLASS_NAMES = {
    name: list(contract.label_names)
    for name, contract in DATASET_CONTRACTS.items()
    if contract.label_type == "multilabel"
}

def encode_labels(labels, dataset_name = None):
    """
    Encode a label series using the benchmark's multiclass/multilabel rules.
    """
    first_label = labels.iloc[0]
    is_vector = isinstance(first_label, (list, tuple, np.ndarray))
    
    if is_vector and len(first_label) > 0:
        # Lists of strings
        if isinstance(first_label[0], str):
            classes_to_use = MULTILABEL_CLASS_NAMES.get(dataset_name)
            mlb = MultiLabelBinarizer(classes=classes_to_use)
            y = mlb.fit_transform(labels)
            classes = list(mlb.classes_)
            
            # Force multilabel if the dataset is defined in the dictionary
            is_multilabel = dataset_name in MULTILABEL_CLASS_NAMES
            
            if is_multilabel:
                return y, classes, True
                
        # Already numeric vectors
        elif isinstance(first_label[0], (int, float, bool, np.number)):
            y = np.stack(labels.to_numpy()).astype(int)
            is_multilabel = dataset_name in MULTILABEL_CLASS_NAMES
            
            if is_multilabel:
                classes = MULTILABEL_CLASS_NAMES.get(
                    dataset_name, [f"Label {i}" for i in range(y.shape[1])]
                )
                if len(classes) != y.shape[1]:
                    raise ValueError("Multilabel class names do not match the label width.")
                return y, classes, True

    # Multiclass
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str).to_numpy())
    classes = list(le.classes_)

    return y, classes, False

def _embedding_sample_ids(dataset_name, image_paths, sample_ids=None):
    if sample_ids is None:
        contract = get_dataset_contract(dataset_name)
        sample_ids = [contract.sample_id_from_path(path) for path in image_paths]
    else:
        sample_ids = [str(sample_id) for sample_id in sample_ids]
    if len(sample_ids) != len(image_paths):
        raise ValueError("Sample IDs and sample references have different lengths.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Embedding sample IDs must be unique.")
    return sample_ids


def prepare_data_multilabel(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                            return_sample_ids = False, sample_ids = None):
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
    emb = np.asarray(embeddings, dtype = np.float32)
    image_names = _embedding_sample_ids(dataset_name, image_paths, sample_ids)
    if emb.ndim != 2 or len(emb) != len(image_names):
        raise ValueError("Embedding rows must match the explicit sample IDs.")
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

    y, classes, is_multilabel = encode_labels(df[label_col], dataset_name)

    if return_sample_ids:
        return X, y, classes, is_multilabel, df[id_col].astype(str).to_numpy()

    return X, y, classes, is_multilabel

def prepare_data_multiclass(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                            sample_ids = None):
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
    emb = np.asarray(embeddings, dtype = np.float32)
    image_names = _embedding_sample_ids(dataset_name, image_paths, sample_ids)
    if emb.ndim != 2 or len(emb) != len(image_names):
        raise ValueError("Embedding rows must match the explicit sample IDs.")
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

def _group_ids_for_samples(metadata_df, id_col, group_col, sample_ids):
    if group_col is None:
        return None

    identifiers = metadata_df[[id_col, group_col]].dropna().copy()
    identifiers[id_col] = identifiers[id_col].astype(str)
    identifiers[group_col] = identifiers[group_col].astype(str)
    if identifiers[id_col].duplicated().any():
        raise ValueError("Grouping metadata contains duplicate sample IDs.")

    group_by_sample = dict(zip(identifiers[id_col], identifiers[group_col]))
    missing = [sample_id for sample_id in sample_ids if sample_id not in group_by_sample]
    if missing:
        raise ValueError(f"Grouping metadata is missing sample IDs: {missing[:5]}")
    return np.asarray([group_by_sample[sample_id] for sample_id in sample_ids])

def _inner_cv_splits(is_multilabel, y, groups, n_splits, random_state):
    if groups is None:
        cv = _inner_cv_splitter(is_multilabel, y, n_splits, random_state)
        return list(_split_iter(cv, np.empty((len(y), 0)), y, is_multilabel))

    groups = np.asarray(groups).astype(str)
    safe_splits = min(n_splits, len(np.unique(groups)))
    if is_multilabel:
        for label_idx in range(y.shape[1]):
            for value in (0, 1):
                safe_splits = min(
                    safe_splits, len(np.unique(groups[y[:, label_idx] == value]))
                )
        signatures = pd.Series(["|".join(row.astype(str)) for row in y])
    else:
        for class_id in np.unique(y):
            safe_splits = min(safe_splits, len(np.unique(groups[y == class_id])))
        signatures = pd.Series(y.astype(str))

    if safe_splits < 2:
        raise ValueError(
            "At least two groups per class are required for inner cross-validation."
        )

    counts = signatures.value_counts()
    strata = signatures.where(
        signatures.map(counts) >= safe_splits, "__rare_label_combination__"
    )
    if strata.value_counts().min() < safe_splits:
        strata = pd.Series("__all_samples__", index=signatures.index)

    cv = StratifiedGroupKFold(
        n_splits=safe_splits, shuffle=True, random_state=random_state
    )
    splits = list(cv.split(np.empty((len(y), 0)), strata, groups=groups))
    for train_idx, validation_idx in splits:
        if set(groups[train_idx]) & set(groups[validation_idx]):
            raise RuntimeError("A group appears in both sides of an inner CV split.")
    return splits

def _split_iter(cv, X, y, is_multilabel):
    if is_multilabel:
        return cv.split(X)

    return cv.split(X, y)

def _outer_split_iter(X, y, is_multilabel, n_splits, random_state,
                      sample_ids, outer_folds, group_ids=None):
    if outer_folds is None:
        if group_ids is not None:
            raise ValueError(
                "A stored outer-fold manifest is required for grouped evaluation."
            )
        cv = _cv_splitter(is_multilabel, n_splits, random_state)
        return _split_iter(cv, X, y, is_multilabel)

    missing = [sample_id for sample_id in sample_ids if sample_id not in outer_folds]
    if missing:
        raise ValueError(f"Outer-fold manifest is missing sample IDs: {missing[:5]}")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Prepared benchmark data contains duplicate sample IDs.")
    unused = sorted(set(outer_folds) - set(sample_ids))
    if unused:
        raise ValueError(f"Manifest samples are missing embeddings: {unused[:5]}")

    fold_values = np.asarray([outer_folds[sample_id] for sample_id in sample_ids], dtype=int)
    observed_folds = sorted(np.unique(fold_values).tolist())
    if observed_folds != list(range(n_splits)):
        raise ValueError(
            f"Expected outer folds 0 through {n_splits - 1}; observed {observed_folds}."
        )
    if group_ids is not None:
        group_folds = pd.DataFrame({"group": group_ids, "fold": fold_values})
        if group_folds.groupby("group")["fold"].nunique().max() != 1:
            raise ValueError("A group appears in more than one outer fold.")

    indices = np.arange(len(X))
    return [
        (indices[fold_values != fold], indices[fold_values == fold])
        for fold in observed_folds
    ]

def _extract_positive_proba(proba):
    if isinstance(proba, list):
        return np.transpose([p[:, 1] for p in proba])

    return proba

def _dataset_info(dataset_name, y, classes, label_type):
    contract = get_dataset_contract(dataset_name)
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
        "label_type": label_type,
        "evaluation_unit": contract.evaluation_unit,
        "accuracy_metric": (
            "exact_match_accuracy" if label_type == "multilabel" else "accuracy"
        ),
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
        "f1_macro": f1_score(y_test, y_pred, average = "macro", zero_division = 0),
        "f1_weighted": f1_score(y_test, y_pred, average = "weighted", zero_division = 0),
        "precision_weighted": precision_score(y_test, y_pred, average = "weighted", zero_division = 0),
    }
    if is_multilabel:
        scores["exact_match_accuracy"] = accuracy_score(y_test, y_pred)
    else:
        scores["accuracy"] = accuracy_score(y_test, y_pred)
        scores["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)

    proba = _extract_positive_proba(pipe.predict_proba(X_test))

    if is_multilabel:
        # Only calculate ROC-AUC for columns that contain both 0s and 1s in y_test
        valid_cols = [i for i in range(y_test.shape[1]) if len(np.unique(y_test[:, i])) > 1]
        
        if len(valid_cols) > 0:
            scores["roc_auc"] = roc_auc_score(y_test[:, valid_cols], proba[:, valid_cols], average = "macro")
        else:
            scores["roc_auc"] = np.nan
            
    elif proba.shape[1] == 2:
        scores["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
    else:
        scores["roc_auc"] = roc_auc_score(y_test, proba, multi_class = "ovr", average = "macro")

    return scores


def _classification_metric_names(is_multilabel):
    if is_multilabel:
        return (
            "exact_match_accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_weighted",
            "roc_auc",
        )
    return (
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_weighted",
        "roc_auc",
    )

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
           label_col, n_splits = 5, random_state = 42, n_trials = 20,
           outer_folds = None, group_col = None, sample_ids = None):
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
    X, y, classes, is_multilabel, sample_ids = prepare_data_multilabel(
        dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
        return_sample_ids = True, sample_ids = sample_ids
    )
    group_ids = _group_ids_for_samples(
        metadata_df, id_col, group_col, sample_ids
    )

    outer_splits = _outer_split_iter(
        X, y, is_multilabel, n_splits, random_state, sample_ids, outer_folds,
        group_ids
    )
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

    metric_names = _classification_metric_names(is_multilabel)
    fold_scores = {metric: [] for metric in metric_names}
    fold_score_rows = []
    best_params_by_fold = []
    inner_folds_by_fold = []
    y_true_parts = []
    y_pred_parts = []

    optuna_seeds = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start = 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = None if group_ids is None else group_ids[train_idx]
        inner_cv = _inner_cv_splits(
            is_multilabel, y_train, train_groups, 3, random_state
        )
        inner_folds_by_fold.append(len(inner_cv))

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
                scoring = CLASSIFICATION_TUNING_METRIC,
                n_jobs = -1
            )
            return scores.mean()

        optuna_seed = random_state + fold_idx - 1
        optuna_seeds.append(int(optuna_seed))
        study = optuna.create_study(
            direction = "maximize",
            sampler = optuna.samplers.TPESampler(seed = optuna_seed)
        )
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
        "optuna_seeds": optuna_seeds,
        "outer_folds_source": "manifest" if outer_folds is not None else "generated",
        "inner_fold_strategy": (
            "stratified_group_kfold" if group_col is not None else "sample_level"
        ),
        "inner_fold_unit": group_col if group_col is not None else "sample_id",
        "tuning_metric": CLASSIFICATION_TUNING_METRIC,
        "n_trials": int(n_trials)
    }

    dataset_info = _dataset_info(
            dataset_name,
            y,
            classes,
            "multilabel" if is_multilabel else "multiclass"
    )

    print(f"Nested CV {n_splits}-fold results (mean ± std):")
    for k in metric_names:
        m, s = summary[k]
        print(f"  {k:18s}: {m:.4f} ± {s:.4f}")

    return summary, dataset_info

def KNN_cv(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
           n_splits = 5, random_state = 42, n_trials = 15, outer_folds = None,
           group_col = None, sample_ids = None):
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
    X, y, classes, is_multilabel, sample_ids = prepare_data_multilabel(
        dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
        return_sample_ids = True, sample_ids = sample_ids
    )
    group_ids = _group_ids_for_samples(
        metadata_df, id_col, group_col, sample_ids
    )

    def build_pipe(params):
        clf = KNeighborsClassifier(
            n_neighbors = params["n_neighbors"],
            weights = params["weights"],
            metric = params["metric"]
        )
        pipe = Pipeline([("norm", Normalizer(norm = "l2")), ("knn", clf)])

        return pipe

    print(f"--- Optimizing KNN with Optuna ({n_trials} trials) ---")
    outer_splits = _outer_split_iter(
        X, y, is_multilabel, n_splits, random_state, sample_ids, outer_folds,
        group_ids
    )
    metric_names = _classification_metric_names(is_multilabel)
    fold_scores = {metric: [] for metric in metric_names}
    fold_score_rows = []
    best_params_by_fold = []
    inner_folds_by_fold = []
    y_true_parts = []
    y_pred_parts = []

    optuna_seeds = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start = 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = None if group_ids is None else group_ids[train_idx]
        inner_cv = _inner_cv_splits(
            is_multilabel, y_train, train_groups, 3, random_state
        )
        inner_folds_by_fold.append(len(inner_cv))

        def objective(trial):
            """
            Function used by Optuna to maximize KNN performance.
            """
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "cosine", "manhattan"])
            }

            # Macro F1 prevents majority findings/classes from dominating selection.
            score = cross_val_score(
                build_pipe(params), X_train, y_train,
                cv = inner_cv,
                scoring = CLASSIFICATION_TUNING_METRIC,
                n_jobs = -1
            ).mean()
            return score

        optuna_seed = random_state + fold_idx - 1
        optuna_seeds.append(int(optuna_seed))
        study = optuna.create_study(
            direction = "maximize",
            sampler = optuna.samplers.TPESampler(seed = optuna_seed)
        )
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

    summary = dict(score_summary)
    summary.update({
        "best_k": best_params["n_neighbors"],
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
            "optuna_seeds": optuna_seeds,
            "outer_folds_source": "manifest" if outer_folds is not None else "generated",
            "inner_fold_strategy": (
                "stratified_group_kfold" if group_col is not None else "sample_level"
            ),
            "inner_fold_unit": group_col if group_col is not None else "sample_id",
            "tuning_metric": CLASSIFICATION_TUNING_METRIC,
            "n_trials": int(n_trials)
        },
    })

    print(f"KNN nested CV (n_splits={n_splits}) — most common best k = {summary['best_k']}")
    for metric in metric_names:
        mean, std = summary[metric]
        print(f"  {metric:18s}: {mean:.4f} ± {std:.4f}")

    return summary

def logistic_regression_cv(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                           n_splits = 5, random_state = 42, n_trials = 15,
                           outer_folds = None, group_col = None, sample_ids = None):
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
    X, y, classes, is_multilabel, sample_ids = prepare_data_multilabel(
        dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
        return_sample_ids = True, sample_ids = sample_ids
    )
    group_ids = _group_ids_for_samples(
        metadata_df, id_col, group_col, sample_ids
    )

    outer_splits = _outer_split_iter(
        X, y, is_multilabel, n_splits, random_state, sample_ids, outer_folds,
        group_ids
    )
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
            solver = solver,
            random_state = random_state
        )

        if is_multilabel:
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = base_clf

        return Pipeline([("std", StandardScaler()), ("clf", clf)])

    metric_names = _classification_metric_names(is_multilabel)
    fold_scores = {metric: [] for metric in metric_names}
    fold_score_rows = []
    best_params_by_fold = []
    inner_folds_by_fold = []
    y_true_parts = []
    y_pred_parts = []

    optuna_seeds = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start = 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = None if group_ids is None else group_ids[train_idx]
        inner_cv = _inner_cv_splits(
            is_multilabel, y_train, train_groups, 3, random_state
        )
        inner_folds_by_fold.append(len(inner_cv))

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
                scoring = CLASSIFICATION_TUNING_METRIC,
                n_jobs = -1
            ).mean()

        optuna_seed = random_state + fold_idx - 1
        optuna_seeds.append(int(optuna_seed))
        study = optuna.create_study(
            direction = "maximize",
            sampler = optuna.samplers.TPESampler(seed = optuna_seed)
        )
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
        "optuna_seeds": optuna_seeds,
        "outer_folds_source": "manifest" if outer_folds is not None else "generated",
        "inner_fold_strategy": (
            "stratified_group_kfold" if group_col is not None else "sample_level"
        ),
        "inner_fold_unit": group_col if group_col is not None else "sample_id",
        "tuning_metric": CLASSIFICATION_TUNING_METRIC,
        "n_trials": int(n_trials)
    }

    # Print out the results
    print(f"Logistic Regression Nested CV Results (n_splits={n_splits}):")
    for metric in metric_names:
        mean, std = summary[metric]
        print(f"  {metric:18s}: {mean:.4f} ± {std:.4f}")

    return summary

def retrieval_eval(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                   ks = (1, 5, 10), normalize = True, per_class = False,
                   bootstrap = True, n_bootstrap = 1000, ci = 95, random_state = 42,
                   group_col = None, sample_ids = None):
    """
    Cross-group retrieval using exact-class or per-finding relevance.

    Multiclass candidates are relevant when they share the query class. Multilabel
    retrieval evaluates each positive query finding separately. The query itself and,
    when available, every sample from the same patient/lesion group are excluded.

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
        group_col : Metadata column identifying patient/lesion groups

    Returns:
        results : JSON-compatible summary of the recall test
    """
    X, y, classes, is_multilabel, sample_ids = prepare_data_multilabel(
        dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
        return_sample_ids = True, sample_ids = sample_ids
    )
    group_ids = _group_ids_for_samples(
        metadata_df, id_col, group_col, sample_ids
    )

    ks = tuple(int(k) for k in ks)
    if not ks or any(k < 1 for k in ks):
        raise ValueError("Retrieval K values must be positive integers.")
    if bootstrap and n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1 when bootstrap is enabled.")
    if not 0 < ci < 100:
        raise ValueError("ci must be between 0 and 100.")

    if len(ks) == 1:
        message = str(ks[0])
    else:
        message = f"{', '.join(str(k) for k in ks[:-1])}, and {ks[-1]}"

    print(f"--- Retrieval evaluation with Recall@{message} ---")

    # Normalize to make cosine == dot
    if normalize:
        X = Normalizer(norm = "l2").fit_transform(X)

    N = X.shape[0]

    protocol = {
        "type": (
            "group_excluded_all_vs_all_retrieval"
            if group_ids is not None else "self_excluded_all_vs_all_retrieval"
        ),
        "ks": [int(k) for k in ks],
        "similarity": "cosine",
        "normalized": normalize,
        "relevance": "per_finding" if is_multilabel else "exact_class",
        "aggregation": (
            "micro_over_query_finding_pairs" if is_multilabel else "queries"
        ),
        "candidate_exclusion": (
            "self_and_same_group" if group_ids is not None else "self"
        ),
        "group_column": group_col,
        "bootstrap": bootstrap,
        "bootstrap_unit": group_col if group_ids is not None else "query",
        "n_bootstrap": int(n_bootstrap),
        "ci": int(ci),
        "random_state": int(random_state),
        "tuned": False,
    }

    similarities = X @ X.T

    def ranked_candidates(query_index):
        candidate_mask = np.ones(N, dtype=bool)
        candidate_mask[query_index] = False
        if group_ids is not None:
            candidate_mask &= group_ids != group_ids[query_index]
        candidates = np.flatnonzero(candidate_mask)
        order = np.argsort(-similarities[query_index, candidates], kind="stable")
        return candidates[order]

    def metrics_for_relevance(relevance):
        hits = np.flatnonzero(relevance)
        precision_at_hits = np.cumsum(relevance)[hits] / (hits + 1)
        return {
            "average_precision": float(np.mean(precision_at_hits)),
            "recall_at_k": {
                int(k): bool(np.any(relevance[:k])) for k in ks
            },
        }

    evaluation_units = []
    for query_index in range(N):
        candidates = ranked_candidates(query_index)
        if is_multilabel:
            positive_findings = np.flatnonzero(y[query_index] == 1)
            for finding_index in positive_findings:
                relevance = y[candidates, finding_index] == 1
                if not np.any(relevance):
                    continue
                evaluation_units.append({
                    "query_index": int(query_index),
                    "group_id": (
                        str(group_ids[query_index])
                        if group_ids is not None else str(query_index)
                    ),
                    "label_index": int(finding_index),
                    **metrics_for_relevance(relevance),
                })
        else:
            relevance = y[candidates] == y[query_index]
            if not np.any(relevance):
                continue
            evaluation_units.append({
                "query_index": int(query_index),
                "group_id": (
                    str(group_ids[query_index])
                    if group_ids is not None else str(query_index)
                ),
                "label_index": int(y[query_index]),
                **metrics_for_relevance(relevance),
            })

    valid_query_indices = {unit["query_index"] for unit in evaluation_units}
    n_eval = len(valid_query_indices)

    def aggregate_units(units):
        return {
            "recall_at_k": {
                int(k): float(np.mean([
                    unit["recall_at_k"][int(k)] for unit in units
                ]))
                for k in ks
            },
            "map": float(np.mean([
                unit["average_precision"] for unit in units
            ])),
        }

    if not evaluation_units:
        return {
            "n_total": N,
            "n_eval": 0,
            "n_excluded_queries": N,
            "n_evaluation_units": 0,
            "recall_at_k": {int(k): np.nan for k in ks},
            "map": np.nan,
            "classes": classes,
            "evaluation_protocol": protocol,
            "protocol": {
                "similarity": "cosine",
                "normalized": normalize,
                "relevance": protocol["relevance"],
                "candidate_exclusion": protocol["candidate_exclusion"],
                "tuned": False,
            },
            "note": "No query has a relevant candidate outside its excluded group.",
        }

    aggregate = aggregate_units(evaluation_units)
    recall_at_k = aggregate["recall_at_k"]
    mAP = aggregate["map"]

    # JSON compatible
    results = {
        "n_total": N,
        "n_eval": n_eval,
        "n_excluded_queries": int(N - n_eval),
        "n_evaluation_units": int(len(evaluation_units)),
        "recall_at_k": recall_at_k,
        "map": mAP,
        "classes": classes,
        "evaluation_protocol": protocol,
        "protocol": {
            "similarity": "cosine",
            "normalized": normalize,
            "relevance": protocol["relevance"],
            "candidate_exclusion": protocol["candidate_exclusion"],
            "tuned": False,
        },
    }

    units_by_label = {
        label_index: [
            unit for unit in evaluation_units
            if unit["label_index"] == label_index
        ]
        for label_index in range(len(classes))
    }
    if is_multilabel:
        per_finding = {}
        for label_index, label_name in enumerate(classes):
            finding_units = units_by_label[label_index]
            total_queries = int(np.sum(y[:, label_index] == 1))
            if not finding_units:
                per_finding[str(label_name)] = {
                    "n_total_queries": total_queries,
                    "n_queries": 0,
                    "n_excluded_queries": total_queries,
                    "recall_at_k": {int(k): np.nan for k in ks},
                    "map": np.nan,
                }
                continue
            finding_metrics = aggregate_units(finding_units)
            per_finding[str(label_name)] = {
                "n_total_queries": total_queries,
                "n_queries": len(finding_units),
                "n_excluded_queries": total_queries - len(finding_units),
                **finding_metrics,
            }
        populated_findings = [
            metrics for metrics in per_finding.values()
            if metrics["n_queries"] > 0
        ]
        results["per_finding"] = per_finding
        results["macro_recall_at_k"] = {
            int(k): float(np.mean([
                metrics["recall_at_k"][int(k)] for metrics in populated_findings
            ]))
            for k in ks
        }
        results["macro_map"] = float(np.mean([
            metrics["map"] for metrics in populated_findings
        ]))

    if bootstrap:
        rng = np.random.default_rng(random_state)
        alpha = (100 - ci) / 2
        boot_recall = {int(K): [] for K in ks}
        boot_map = []

        units_by_bootstrap_key = {}
        for unit in evaluation_units:
            bootstrap_key = (
                unit["group_id"] if group_ids is not None
                else str(unit["query_index"])
            )
            units_by_bootstrap_key.setdefault(bootstrap_key, []).append(unit)
        bootstrap_keys = sorted(units_by_bootstrap_key)

        for _ in range(n_bootstrap):
            sampled_keys = rng.choice(
                bootstrap_keys, size=len(bootstrap_keys), replace=True
            )
            sampled_units = [
                unit
                for bootstrap_key in sampled_keys
                for unit in units_by_bootstrap_key[bootstrap_key]
            ]
            sampled_metrics = aggregate_units(sampled_units)
            for k in ks:
                boot_recall[int(k)].append(
                    sampled_metrics["recall_at_k"][int(k)]
                )
            boot_map.append(sampled_metrics["map"])

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

    # Optional per-class Recall@K for multiclass datasets.
    if per_class and not is_multilabel:
        per_cls = {}
        for class_index, class_name in enumerate(classes):
            class_units = units_by_label[class_index]
            if not class_units:
                per_cls[str(class_name)] = {int(k): np.nan for k in ks}
                continue
            per_cls[str(class_name)] = {
                int(k): float(np.mean([
                    unit["recall_at_k"][int(k)] for unit in class_units
                ]))
                for k in ks
            }
        results["recall_at_k_per_class"] = per_cls

    # Print summary
    print(
        f"Retrieval (cross-group, cosine) — evaluated {n_eval}/{N} queries "
        f"across {len(evaluation_units)} relevance units."
    )
    for K in ks:
        print(f"  Recall@{K}: {recall_at_k[int(K)]:.4f}")
    print(f"  mAP      : {mAP:.4f}")

    return results

def clustering_eval(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col,
                    k_range = range(2, 15), random_state = 42,
                    compute_silhouette = True, sample_ids = None):
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
            id_col, label_col, sample_ids = sample_ids)

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

    if get_dataset_contract(dataset_name).label_type == "multilabel":
        results_dict["note"] = (
            "Clustering classes are unique multilabel combinations and should "
            "be interpreted as exploratory."
        )

    return results_dict
