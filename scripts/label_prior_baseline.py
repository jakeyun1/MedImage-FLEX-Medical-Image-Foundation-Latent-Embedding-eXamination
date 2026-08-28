"""
label_prior_baseline.py

Classification baselines that use training-label frequencies only.
"""

import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)

from scripts.tests import (
    _dataset_info,
    _outer_split_iter,
    _per_class_metrics,
    encode_labels,
)


def _score_predictions(y_true, y_pred, scores, is_multilabel):
    result = {
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    if is_multilabel:
        result["exact_match_accuracy"] = float(accuracy_score(y_true, y_pred))
    else:
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
        result["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true, y_pred)
        )

    try:
        if is_multilabel:
            auc = roc_auc_score(y_true, scores, average="macro")
        elif scores.shape[1] == 2:
            auc = roc_auc_score(y_true, scores[:, 1])
        else:
            auc = roc_auc_score(
                y_true, scores, multi_class="ovr", average="macro"
            )
        result["roc_auc"] = float(auc)
    except ValueError:
        result["roc_auc"] = None

    return result


def _summarize_fold_scores(fold_scores):
    summary = {}
    ignored = {"fold", "training_label_priors"}
    metrics = sorted(set().union(*(fold.keys() for fold in fold_scores)) - ignored)
    for metric in metrics:
        values = [fold[metric] for fold in fold_scores if fold[metric] is not None]
        summary[metric] = [
            float(np.mean(values)) if values else None,
            float(np.std(values)) if values else None,
        ]
    return summary


def _multiclass_predictions(y_train, n_test, strategy, rng, n_classes):
    counts = np.bincount(y_train, minlength=n_classes)
    priors = counts / counts.sum()

    if strategy == "majority":
        predictions = np.full(n_test, int(np.argmax(counts)), dtype=int)
        scores = np.zeros((n_test, n_classes), dtype=float)
        scores[:, int(np.argmax(counts))] = 1.0
    else:
        predictions = rng.choice(n_classes, size=n_test, p=priors)
        scores = np.tile(priors, (n_test, 1))

    return predictions, scores, priors


def _multilabel_predictions(y_train, n_test, strategy, rng):
    priors = np.mean(y_train, axis=0)

    if strategy == "majority":
        majority = (priors >= 0.5).astype(int)
        predictions = np.tile(majority, (n_test, 1))
        scores = predictions.astype(float)
    else:
        predictions = rng.binomial(1, priors, size=(n_test, y_train.shape[1]))
        scores = np.tile(priors, (n_test, 1))

    return predictions, scores, priors


def _format_priors(priors, classes):
    return {
        str(class_name): float(prior)
        for class_name, prior in zip(classes, priors)
    }


def run_label_prior_baselines(
    dataset_name,
    metadata_df,
    id_col,
    label_col,
    output_dir,
    outer_folds,
    n_splits=5,
    random_state=42,
    manifest_info=None,
):
    """
    Evaluate majority and stratified-random predictors on stored outer folds.
    """
    sample_ids = metadata_df[id_col].astype(str).to_numpy()
    y, classes, is_multilabel = encode_labels(
        metadata_df[label_col], dataset_name
    )
    X = np.empty((len(y), 0))
    splits = list(_outer_split_iter(
        X, y, is_multilabel, n_splits, random_state, sample_ids, outer_folds
    ))

    results = {}
    for strategy in ("majority", "stratified"):
        fold_scores = []
        y_true_parts = []
        y_pred_parts = []
        prediction_seeds = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            y_train, y_test = y[train_idx], y[test_idx]
            prediction_seed = random_state + fold_idx
            rng = np.random.default_rng(prediction_seed)

            if is_multilabel:
                y_pred, scores, priors = _multilabel_predictions(
                    y_train, len(test_idx), strategy, rng
                )
            else:
                y_pred, scores, priors = _multiclass_predictions(
                    y_train, len(test_idx), strategy, rng, len(classes)
                )

            scores_for_fold = _score_predictions(
                y_test, y_pred, scores, is_multilabel
            )
            scores_for_fold["fold"] = int(fold_idx + 1)
            scores_for_fold["training_label_priors"] = _format_priors(
                priors, classes
            )
            fold_scores.append(scores_for_fold)
            y_true_parts.append(y_test)
            y_pred_parts.append(y_pred)
            prediction_seeds.append(int(prediction_seed))

        strategy_result = _summarize_fold_scores(fold_scores)
        strategy_result["fold_scores"] = fold_scores
        strategy_result["per_class"] = _per_class_metrics(
            np.concatenate(y_true_parts),
            np.concatenate(y_pred_parts),
            classes,
            is_multilabel,
        )
        strategy_result["evaluation_protocol"] = {
            "strategy": strategy,
            "uses_embeddings": False,
            "class_priors_source": "outer_training_fold_only",
            "outer_folds": int(n_splits),
            "outer_folds_source": "manifest",
            "random_state": int(random_state),
            "prediction_seeds": prediction_seeds if strategy == "stratified" else [],
            "tuned": False,
        }
        results[strategy] = strategy_result

    output = {
        "baseline_type": "label_prior",
        "dataset_info": _dataset_info(
            dataset_name,
            y,
            classes,
            "multilabel" if is_multilabel else "multiclass",
        ),
        "baselines": results,
    }
    if manifest_info is not None:
        output["reproducibility"] = manifest_info

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Label-prior baseline saved to: {os.path.abspath(output_path)}")
    return output
