import os
import tempfile
import unittest

import numpy as np

from scripts.oof import (
    OOFAccumulator,
    validate_oof_artifact,
    write_oof_artifact,
)


class OOFArtifactTests(unittest.TestCase):
    def _multiclass_accumulator(self):
        return OOFAccumulator(
            "fixture",
            "logistic_regression",
            ["s0", "s1", "s2", "s3"],
            ["g0", "g1", "g2", "g3"],
            np.asarray([0, 1, 0, 1]),
            ["negative", "positive"],
            False,
        )

    def test_complete_artifact_round_trips_without_pickle(self):
        accumulator = self._multiclass_accumulator()
        accumulator.record(
            [2, 3], 1, [0, 1], [0, 1], [[0.8, 0.2], [0.1, 0.9]]
        )
        accumulator.record(
            [0, 1], 0, [0, 1], [0, 1], [[0.9, 0.1], [0.2, 0.8]]
        )
        artifact = accumulator.finalize()

        np.testing.assert_array_equal(artifact["outer_folds"], [0, 0, 1, 1])
        np.testing.assert_array_equal(artifact["sample_ids"], ["s0", "s1", "s2", "s3"])

        with tempfile.TemporaryDirectory() as directory:
            metadata = write_oof_artifact(
                artifact,
                directory,
                "logistic_regression.npz",
                relative_path="oof/fixture/logistic_regression.npz",
            )
            path = os.path.join(directory, "logistic_regression.npz")
            with np.load(path, allow_pickle=False) as loaded:
                reloaded = {key: loaded[key] for key in loaded.files}

        self.assertTrue(validate_oof_artifact(reloaded))
        self.assertEqual(metadata["n_samples"], 4)
        self.assertEqual(metadata["n_groups"], 4)
        self.assertEqual(metadata["n_outputs"], 2)
        self.assertEqual(len(metadata["sha256"]), 64)
        self.assertEqual(metadata["path"], "oof/fixture/logistic_regression.npz")

    def test_duplicate_prediction_is_rejected(self):
        accumulator = self._multiclass_accumulator()
        accumulator.record([0], 0, [0], [0], [[0.9, 0.1]])
        with self.assertRaisesRegex(ValueError, "more than once"):
            accumulator.record([0], 0, [0], [0], [[0.9, 0.1]])

    def test_missing_prediction_is_rejected(self):
        accumulator = self._multiclass_accumulator()
        accumulator.record(
            [0, 1], 0, [0, 1], [0, 1], [[0.9, 0.1], [0.2, 0.8]]
        )
        with self.assertRaisesRegex(ValueError, "missing samples"):
            accumulator.finalize()

    def test_misaligned_target_is_rejected(self):
        accumulator = self._multiclass_accumulator()
        with self.assertRaisesRegex(ValueError, "misaligned"):
            accumulator.record([0], 0, [1], [0], [[0.9, 0.1]])

    def test_invalid_probability_rows_are_rejected(self):
        accumulator = self._multiclass_accumulator()
        with self.assertRaisesRegex(ValueError, "sum to one"):
            accumulator.record([0], 0, [0], [0], [[0.9, 0.9]])

    def test_group_spanning_outer_folds_is_rejected(self):
        accumulator = OOFAccumulator(
            "fixture", "mlp", ["s0", "s1"], ["patient", "patient"],
            np.asarray([0, 1]), ["negative", "positive"], False,
        )
        accumulator.record([0], 0, [0], [0], [[0.9, 0.1]])
        accumulator.record([1], 1, [1], [1], [[0.1, 0.9]])
        with self.assertRaisesRegex(ValueError, "multiple outer folds"):
            accumulator.finalize()

    def test_multilabel_shapes_and_binary_outputs_are_validated(self):
        accumulator = OOFAccumulator(
            "fixture", "knn", ["s0", "s1"], ["g0", "g1"],
            np.asarray([[1, 0], [0, 1]]), ["finding_a", "finding_b"], True,
        )
        accumulator.record(
            [0, 1], 0,
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
            [[0.8, 0.2], [0.3, 0.7]],
        )
        self.assertTrue(validate_oof_artifact(accumulator.finalize()))

    def test_tampered_ordered_id_hash_is_rejected(self):
        accumulator = self._multiclass_accumulator()
        accumulator.record(
            [0, 1], 0, [0, 1], [0, 1], [[0.9, 0.1], [0.2, 0.8]]
        )
        accumulator.record(
            [2, 3], 1, [0, 1], [0, 1], [[0.8, 0.2], [0.1, 0.9]]
        )
        artifact = accumulator.finalize()
        artifact["ordered_sample_ids_sha256"] = np.asarray("incorrect")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_oof_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
