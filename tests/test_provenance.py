import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from scripts.provenance import (
    _git_provenance,
    build_dataset_fingerprint,
    build_run_provenance,
    canonical_json_sha256,
)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_json_hash_ignores_dictionary_order(self):
        first = canonical_json_sha256({"b": 2, "a": [1, 3]})
        second = canonical_json_sha256({"a": [1, 3], "b": 2})
        self.assertEqual(first, second)

    def test_dataset_fingerprint_is_row_order_independent(self):
        first = build_dataset_fingerprint(
            ["b.jpg", "a.jpg"],
            ["patient-2", "patient-1"],
            [[0, 1], [1, 0]],
        )
        second = build_dataset_fingerprint(
            ["a.jpg", "b.jpg"],
            ["patient-1", "patient-2"],
            [[1, 0], [0, 1]],
        )
        self.assertEqual(first, second)
        self.assertFalse(first["image_bytes_hashed"])

    def test_dataset_fingerprint_changes_with_semantics(self):
        baseline = build_dataset_fingerprint(
            ["a.jpg"], ["patient-1"], [np.int64(1)]
        )
        changed_label = build_dataset_fingerprint(
            ["a.jpg"], ["patient-1"], [np.int64(0)]
        )
        changed_group = build_dataset_fingerprint(
            ["a.jpg"], ["patient-2"], [np.int64(1)]
        )
        self.assertNotEqual(
            baseline["retained_sample_group_labels_sha256"],
            changed_label["retained_sample_group_labels_sha256"],
        )
        self.assertNotEqual(
            baseline["retained_sample_group_labels_sha256"],
            changed_group["retained_sample_group_labels_sha256"],
        )

    def test_dataset_fingerprint_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            build_dataset_fingerprint(["a.jpg"], [], [1])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_dataset_fingerprint(
                ["a.jpg", "a.jpg"], ["p1", "p2"], [0, 1]
            )

    @patch("scripts.provenance.subprocess.run")
    def test_git_provenance_separates_tracked_and_untracked_state(self, run):
        run.side_effect = [
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout="paper_readiness_check\n"),
            SimpleNamespace(stdout=" M main.py\n?? output/\n"),
        ]
        result = _git_provenance("/repo")

        self.assertEqual(result["commit"], "abc123")
        self.assertTrue(result["tracked_changes"])
        self.assertTrue(result["untracked_files"])

    @patch("scripts.provenance._git_provenance")
    @patch("scripts.provenance._distribution_version")
    def test_run_provenance_documents_config_code_and_runtime(
        self, distribution_version, git_provenance
    ):
        distribution_version.return_value = "1.2.3"
        git_provenance.return_value = {
            "commit": "abc123",
            "branch": "paper_readiness_check",
            "tracked_changes": False,
            "untracked_files": True,
        }
        result = build_run_provenance({"model_id": "resnet50"}, "/repo")

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(result["config_canonical_sha256"]), 64)
        self.assertEqual(result["git"]["commit"], "abc123")
        self.assertIn("python_version", result["runtime"])
        self.assertEqual(
            result["runtime"]["package_versions"]["torch"], "1.2.3"
        )


if __name__ == "__main__":
    unittest.main()
