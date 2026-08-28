import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


if "optuna" not in sys.modules:
    optuna = types.ModuleType("optuna")
    optuna.logging = SimpleNamespace(WARNING=30, set_verbosity=lambda level: None)
    sys.modules["optuna"] = optuna

from scripts.tests import retrieval_eval


def _paths(sample_ids):
    return [f"/dataset/{sample_id}" for sample_id in sample_ids]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_average_precision_does_not_count_the_query_itself(self):
        sample_ids = ["a0.jpg", "a1.jpg", "b0.jpg"]
        metadata = pd.DataFrame({
            "image_id": sample_ids,
            "dx": ["A", "A", "B"],
            "lesion_id": ["g0", "g1", "g2"],
        })
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.9],
        ])

        results = retrieval_eval(
            "ham10000", embeddings, metadata, _paths(sample_ids),
            "image_id", "dx", ks=(1,), bootstrap=False,
        )

        self.assertEqual(results["n_eval"], 2)
        self.assertAlmostEqual(results["map"], 0.5)
        self.assertEqual(
            results["evaluation_protocol"]["candidate_exclusion"], "self"
        )

    def test_same_group_candidates_are_completely_excluded(self):
        sample_ids = ["a0.jpg", "a1.jpg", "a2.jpg", "b0.jpg"]
        metadata = pd.DataFrame({
            "image_id": sample_ids,
            "dx": ["A", "A", "A", "B"],
            "lesion_id": ["shared", "shared", "other-a", "other-b"],
        })
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.99, 0.01],
            [-1.0, 0.0],
            [0.0, 1.0],
        ])

        unrestricted = retrieval_eval(
            "ham10000", embeddings, metadata, _paths(sample_ids),
            "image_id", "dx", ks=(1,), bootstrap=False,
        )
        group_excluded = retrieval_eval(
            "ham10000", embeddings, metadata, _paths(sample_ids),
            "image_id", "dx", ks=(1,), bootstrap=False,
            group_col="lesion_id",
        )

        self.assertAlmostEqual(unrestricted["recall_at_k"][1], 2 / 3)
        self.assertEqual(group_excluded["recall_at_k"][1], 0.0)
        self.assertEqual(group_excluded["n_eval"], 3)
        self.assertEqual(
            group_excluded["evaluation_protocol"]["candidate_exclusion"],
            "self_and_same_group",
        )

    def test_multilabel_retrieval_uses_per_finding_relevance(self):
        sample_ids = ["s0.jpg", "s1.jpg", "s2.jpg", "s3.jpg"]
        metadata = pd.DataFrame({
            "image path": [f"dataset/{sample_id}" for sample_id in sample_ids],
            "pathology": [
                ["mass_BENIGN", "calcification_BENIGN"],
                ["mass_BENIGN"],
                ["calcification_BENIGN"],
                [],
            ],
            "patient_id": ["p0", "p1", "p2", "p3"],
        })
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.99, 0.1],
            [0.8, 0.6],
            [-1.0, 0.0],
        ])

        results = retrieval_eval(
            "cbis_ddsm", embeddings, metadata, _paths(sample_ids),
            "image path", "pathology", ks=(1, 2), bootstrap=False,
            group_col="patient_id",
        )

        self.assertEqual(results["n_eval"], 3)
        self.assertEqual(results["n_evaluation_units"], 4)
        self.assertEqual(results["evaluation_protocol"]["relevance"], "per_finding")
        self.assertEqual(results["per_finding"]["mass_BENIGN"]["n_queries"], 2)
        self.assertEqual(
            results["per_finding"]["mass_BENIGN"]["n_excluded_queries"], 0
        )
        self.assertEqual(
            results["per_finding"]["calcification_BENIGN"]["n_queries"], 2
        )
        self.assertEqual(results["per_finding"]["mass_MALIGNANT"]["n_queries"], 0)
        self.assertEqual(
            results["per_finding"]["mass_MALIGNANT"]["n_total_queries"], 0
        )
        self.assertIn("macro_map", results)

    def test_group_bootstrap_is_deterministic(self):
        sample_ids = ["a0.jpg", "a1.jpg", "a2.jpg", "a3.jpg"]
        metadata = pd.DataFrame({
            "image_id": sample_ids,
            "dx": ["A", "A", "A", "A"],
            "lesion_id": ["g0", "g1", "g2", "g3"],
        })
        embeddings = np.eye(4, dtype=float)
        kwargs = {
            "ks": (1,),
            "bootstrap": True,
            "n_bootstrap": 25,
            "random_state": 19,
            "group_col": "lesion_id",
        }

        first = retrieval_eval(
            "ham10000", embeddings, metadata, _paths(sample_ids),
            "image_id", "dx", **kwargs,
        )
        second = retrieval_eval(
            "ham10000", embeddings, metadata, _paths(sample_ids),
            "image_id", "dx", **kwargs,
        )

        self.assertEqual(first["confidence_intervals"], second["confidence_intervals"])
        self.assertEqual(
            first["evaluation_protocol"]["bootstrap_unit"], "lesion_id"
        )

    def test_rejects_invalid_k(self):
        metadata = pd.DataFrame({
            "image_id": ["a.jpg", "b.jpg"],
            "dx": ["A", "A"],
        })
        with self.assertRaisesRegex(ValueError, "positive integers"):
            retrieval_eval(
                "ham10000", np.eye(2), metadata, _paths(metadata["image_id"]),
                "image_id", "dx", ks=(0,), bootstrap=False,
            )


class RetrievalWiringTests(unittest.TestCase):
    def test_run_benchmark_passes_group_column_to_retrieval(self):
        import scripts.run_benchmark as benchmark_module

        captured = {}

        def fake_retrieval(*args, **kwargs):
            captured.update(kwargs)
            return {"recall_at_k": {1: 0.0, 5: 0.0, 10: 0.0}, "map": 0.0}

        with (
            patch.object(benchmark_module, "MLP_cv", return_value=({}, {})),
            patch.object(benchmark_module, "KNN_cv", return_value={}),
            patch.object(benchmark_module, "logistic_regression_cv", return_value={}),
            patch.object(benchmark_module, "retrieval_eval", side_effect=fake_retrieval),
            patch.object(benchmark_module, "clustering_eval", return_value={}),
        ):
            benchmark_module.run_benchmark(
                "ham10000", np.eye(2), pd.DataFrame(), [],
                "image_id", "dx", group_col="lesion_id",
                sample_ids=["a.jpg", "b.jpg"],
            )

        self.assertEqual(captured["group_col"], "lesion_id")
        self.assertEqual(captured["sample_ids"], ["a.jpg", "b.jpg"])


if __name__ == "__main__":
    unittest.main()
