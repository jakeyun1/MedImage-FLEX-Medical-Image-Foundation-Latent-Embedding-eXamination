import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


if "optuna" not in sys.modules:
    optuna = types.ModuleType("optuna")
    optuna.logging = SimpleNamespace(WARNING=30, set_verbosity=lambda level: None)
    sys.modules["optuna"] = optuna

from scripts.random_baseline import aggregate_benchmark_results
from scripts.random_baseline import _validate_cached_result as validate_random_cache
from scripts.permuted_baseline import (
    _validate_cached_result as validate_permuted_cache,
)
from scripts.tests import (
    CLASSIFICATION_TUNING_METRIC,
    _classification_scores,
    _evaluate_classifier,
    _outer_split_iter,
    _validate_oof_metric_consistency,
)
from scripts.oof import OOFAccumulator


class _Predictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, _):
        return self.probabilities


class ClassificationMetricTests(unittest.TestCase):
    def test_all_adapters_share_macro_f1_tuning_metric(self):
        self.assertEqual(CLASSIFICATION_TUNING_METRIC, "f1_macro")

    def test_multiclass_reports_macro_and_balanced_metrics(self):
        y_true = np.asarray([0, 0, 0, 1])
        y_pred = np.asarray([0, 0, 0, 0])
        proba = np.asarray([[0.8, 0.2]] * 4)
        scores = _evaluate_classifier(
            _Predictor(proba), np.zeros((4, 1)), y_true, False, y_pred
        )
        self.assertIn("f1_macro", scores)
        self.assertIn("balanced_accuracy", scores)
        self.assertIn("accuracy", scores)
        self.assertNotIn("exact_match_accuracy", scores)

    def test_multilabel_names_exact_match_accuracy(self):
        y_true = np.asarray([[1, 0], [0, 1]])
        y_pred = np.asarray([[1, 0], [1, 0]])
        proba = [
            np.asarray([[0.1, 0.9], [0.1, 0.9]]),
            np.asarray([[0.9, 0.1], [0.9, 0.1]]),
        ]
        scores = _evaluate_classifier(
            _Predictor(proba), np.zeros((2, 1)), y_true, True, y_pred
        )
        self.assertIn("exact_match_accuracy", scores)
        self.assertIn("f1_macro", scores)
        self.assertNotIn("accuracy", scores)

    def test_grouped_outer_cv_requires_a_manifest(self):
        with self.assertRaisesRegex(ValueError, "manifest"):
            _outer_split_iter(
                np.zeros((4, 1)),
                np.asarray([0, 0, 1, 1]),
                False,
                2,
                42,
                ["a", "b", "c", "d"],
                None,
                np.asarray(["g1", "g1", "g2", "g2"]),
            )

    def test_oof_artifact_exactly_reproduces_fold_metrics(self):
        accumulator = OOFAccumulator(
            "fixture", "logistic_regression",
            ["s0", "s1", "s2", "s3"],
            ["g0", "g1", "g2", "g3"],
            np.asarray([0, 1, 0, 1]),
            ["negative", "positive"],
            False,
        )
        score_rows = []
        for fold, indices, scores in (
            (0, [0, 1], [[0.9, 0.1], [0.2, 0.8]]),
            (1, [2, 3], [[0.7, 0.3], [0.1, 0.9]]),
        ):
            y_true = np.asarray([0, 1])
            y_pred = np.asarray([0, 1])
            score_array = np.asarray(scores)
            accumulator.record(
                indices, fold, y_true, y_pred, score_array
            )
            row = {"fold": fold + 1}
            row.update(
                _classification_scores(y_true, y_pred, score_array, False)
            )
            score_rows.append(row)

        artifact = accumulator.finalize()
        self.assertTrue(
            _validate_oof_metric_consistency(artifact, score_rows, False)
        )

        score_rows[0]["f1_macro"] -= 0.01
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_oof_metric_consistency(artifact, score_rows, False)

    def test_random_baseline_reads_standardized_knn_score_shape(self):
        result = {
            "mlp_cv": {"f1_macro": [0.4, 0.1]},
            "knn_cv": {"f1_macro": [0.5, 0.2]},
            "logreg_cv": {"f1_macro": [0.6, 0.3]},
            "retrieval": {},
            "clustering": {},
        }
        summary = aggregate_benchmark_results([result], [42])

        self.assertEqual(summary["metrics"]["knn_cv.f1_macro"]["mean"], 0.5)

    def test_legacy_baseline_result_schemas_are_rejected(self):
        for validator in (validate_random_cache, validate_permuted_cache):
            with self.subTest(validator=validator.__module__):
                with self.assertRaisesRegex(ValueError, "result schema"):
                    validator({}, {}, None)


if __name__ == "__main__":
    unittest.main()
