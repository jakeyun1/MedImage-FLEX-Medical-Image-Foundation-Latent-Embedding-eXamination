import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.tests import clustering_eval
from scripts.clustering_artifacts import (
    validate_clustering_artifact,
    write_clustering_artifact,
)


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = np.vstack((
            rng.normal(-2, 0.2, size=(12, 3)),
            rng.normal(2, 0.2, size=(12, 3)),
        ))
        self.y = np.asarray([0] * 12 + [1] * 12)

    @patch("scripts.tests.get_dataset_contract")
    @patch("scripts.tests.prepare_data_multiclass")
    def test_repeated_kmeans_records_protocol_and_stability(
        self, prepare_data, get_contract
    ):
        prepare_data.return_value = (self.X, self.y, ["a", "b"])
        get_contract.return_value = SimpleNamespace(label_type="multiclass")

        result = clustering_eval(
            "dataset",
            self.X,
            None,
            [],
            "id",
            "label",
            k_range=range(2, 4),
            random_state=42,
            n_init=3,
            seed_repeats=4,
            max_iter=100,
        )

        protocol = result["evaluation_protocol"]
        self.assertEqual(protocol["n_init"], 3)
        self.assertEqual(protocol["max_iter"], 100)
        self.assertEqual(protocol["stability_seeds"], [42, 43, 44, 45])
        self.assertEqual(len(result["seed_stability"]["per_seed"]), 4)
        self.assertEqual(
            result["seed_stability"]["pairwise_assignment_comparisons"], 6
        )
        self.assertEqual(result["class_count_k"]["k"], 2)
        self.assertIn("n_iter", result["class_count_k"])

    @patch("scripts.tests.prepare_data_multiclass")
    def test_repeated_kmeans_rejects_invalid_controls(self, prepare_data):
        prepare_data.return_value = (self.X, self.y, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "n_init"):
            clustering_eval(
                "dataset", self.X, None, [], "id", "label", n_init=0
            )
        with self.assertRaisesRegex(ValueError, "seed_repeats"):
            clustering_eval(
                "dataset", self.X, None, [], "id", "label", seed_repeats=1
            )

    @patch("scripts.tests.get_dataset_contract")
    @patch("scripts.tests.prepare_data_multiclass")
    def test_assignment_artifact_is_aligned_and_round_trips(
        self, prepare_data, get_contract
    ):
        sample_ids = np.asarray([f"s{i}" for i in range(len(self.X))])
        prepare_data.return_value = (self.X, self.y, ["a", "b"], sample_ids)
        get_contract.return_value = SimpleNamespace(label_type="multiclass")
        _, artifact = clustering_eval(
            "dataset", self.X, None, [], "id", "label",
            k_range=range(2, 4), seed_repeats=3, n_init=2,
            compute_silhouette=False, return_artifact=True,
        )
        self.assertTrue(validate_clustering_artifact(artifact))
        self.assertEqual(artifact["assignments"].shape, (3, len(self.X)))
        with tempfile.TemporaryDirectory() as output_dir:
            metadata = write_clustering_artifact(artifact, output_dir)
            self.assertEqual(metadata["n_samples"], len(self.X))
            self.assertEqual(metadata["n_seeds"], 3)

        tampered = dict(artifact)
        tampered["sample_ids"] = artifact["sample_ids"][::-1]
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_clustering_artifact(tampered)


if __name__ == "__main__":
    unittest.main()
