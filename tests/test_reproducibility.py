import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.reproducibility import (
    cohort_overlap_summary,
    cohort_seed_tag,
    prepare_experiment_manifest,
    validate_cohort_selection_record,
    validate_cohort_sensitivity_index,
    validate_cohort_sensitivity_seeds,
)
from scripts.tests import prepare_data_multiclass


def _metadata(group_sizes):
    rows = []
    for group_index, group_size in enumerate(group_sizes):
        group_id = f"patient_{group_index:02d}"
        label = "A" if group_index % 2 == 0 else "B"
        for sample_index in range(group_size):
            rows.append({
                "sample_id": f"{group_id}_image_{sample_index:02d}",
                "patient_id": group_id,
                "label": label,
            })
    return pd.DataFrame(rows)


class ExperimentManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.metadata = _metadata([2, 3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4])

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _prepare(self, manifest_dir, max_samples=12, sample_seed=17):
        return prepare_experiment_manifest(
            self.metadata,
            model_id="fixture/model",
            dataset_name="pad_ufes",
            id_col="sample_id",
            label_col="label",
            group_col="patient_id",
            manifest_dir=manifest_dir,
            max_samples=max_samples,
            sample_seed=sample_seed,
            n_splits=3,
            fold_seed=23,
            available_sample_ids=self.metadata["sample_id"],
        )

    def test_selection_preserves_complete_groups_near_sample_target(self):
        selected, folds, info = self._prepare(self.temporary_directory.name)

        source_sizes = self.metadata.groupby("patient_id").size()
        selected_sizes = selected.groupby("patient_id").size()
        self.assertTrue((selected_sizes == source_sizes.loc[selected_sizes.index]).all())
        self.assertLessEqual(abs(len(selected) - 12), source_sizes.max())
        self.assertEqual(set(selected["sample_id"]), set(folds))
        self.assertTrue(info["complete_groups"])
        self.assertEqual(info["selection_unit"], "patient_id")
        self.assertEqual(info["sample_target_deviation"], len(selected) - 12)
        self.assertEqual(info["n_eligible_samples"], len(self.metadata))
        self.assertEqual(info["n_eligible_groups"], source_sizes.size)
        self.assertIn("_v4_", os.path.basename(info["path"]))

        group_folds = pd.DataFrame({
            "group": selected["patient_id"],
            "fold": selected["sample_id"].map(folds),
        })
        self.assertEqual(group_folds.groupby("group")["fold"].nunique().max(), 1)

    def test_selection_and_folds_are_deterministic(self):
        first_dir = os.path.join(self.temporary_directory.name, "first")
        second_dir = os.path.join(self.temporary_directory.name, "second")
        first_selected, first_folds, first_info = self._prepare(first_dir)
        second_selected, second_folds, second_info = self._prepare(second_dir)

        self.assertEqual(
            first_selected["sample_id"].tolist(),
            second_selected["sample_id"].tolist(),
        )
        self.assertEqual(first_folds, second_folds)
        self.assertEqual(first_info["sha256"], second_info["sha256"])

    def test_reuse_rejects_a_changed_manifest_cohort(self):
        _, _, info = self._prepare(self.temporary_directory.name)
        manifest = pd.read_csv(info["path"])
        manifest.iloc[1:].to_csv(info["path"], index=False)

        with self.assertRaisesRegex(ValueError, "deterministic group selection"):
            self._prepare(self.temporary_directory.name)

    def test_null_target_uses_every_complete_group(self):
        selected, _, info = self._prepare(
            self.temporary_directory.name, max_samples=None
        )

        self.assertEqual(len(selected), len(self.metadata))
        self.assertEqual(info["selection_method"], "all_eligible_samples")
        self.assertIsNone(info["sample_target_deviation"])

    def test_different_seeds_reuse_one_full_embedding_bank(self):
        first, _, _ = self._prepare(
            os.path.join(self.temporary_directory.name, "seed-17"),
            sample_seed=17,
        )
        second, _, _ = self._prepare(
            os.path.join(self.temporary_directory.name, "seed-18"),
            sample_seed=18,
        )
        self.assertNotEqual(set(first["sample_id"]), set(second["sample_id"]))

        full_ids = self.metadata["sample_id"].tolist()
        full_embeddings = np.arange(
            len(full_ids) * 2, dtype=np.float32
        ).reshape(len(full_ids), 2)
        id_to_row = {sample_id: index for index, sample_id in enumerate(full_ids)}
        for selected in (first, second):
            X, _, _ = prepare_data_multiclass(
                "pad_ufes",
                full_embeddings,
                selected,
                full_ids,
                "sample_id",
                "label",
                sample_ids=full_ids,
            )
            expected = full_embeddings[
                [id_to_row[sample_id] for sample_id in selected["sample_id"]]
            ]
            np.testing.assert_array_equal(X, expected)

    def test_cohort_sensitivity_seed_configuration_is_strict(self):
        self.assertEqual(
            validate_cohort_sensitivity_seeds(42, [43, 44, 45]),
            (43, 44, 45),
        )
        with self.assertRaisesRegex(ValueError, "cannot also"):
            validate_cohort_sensitivity_seeds(42, [42, 43])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_cohort_sensitivity_seeds(42, [43, 43])
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            validate_cohort_sensitivity_seeds(42, [43, -1])
        with self.assertRaisesRegex(ValueError, "JSON list"):
            validate_cohort_sensitivity_seeds(42, (43, 44))

    def test_cohort_overlap_schema_is_exact(self):
        summary = cohort_overlap_summary(
            ["a", "b", "c", "d"], ["b", "c", "d", "e"]
        )
        self.assertEqual(summary["intersection_n_samples"], 3)
        self.assertEqual(summary["union_n_samples"], 5)
        self.assertEqual(summary["primary_retained_fraction"], 0.75)
        self.assertEqual(summary["jaccard_index"], 0.6)
        self.assertFalse(summary["identical_sample_set"])
        self.assertEqual(cohort_seed_tag(43), "seed_0000000043")

    def test_cohort_sensitivity_output_schema_is_strict(self):
        primary = {
            "schema_version": 1,
            "analysis_type": "cohort_sampling",
            "role": "primary",
            "primary_sample_seed": 42,
            "sample_seed": 42,
            "fold_seed": 42,
            "evaluation_seed": 42,
            "outer_folds": 5,
            "sensitivity_sample_seeds": [43],
            "max_samples": 5000,
            "n_eligible_samples": 10,
            "n_selected_samples": 4,
            "manifest_sha256": "a" * 64,
            "manifest_applied_after_full_embedding_extraction": True,
        }
        self.assertTrue(validate_cohort_selection_record(primary))

        overlap = cohort_overlap_summary(["a", "b", "c", "d"], ["b", "c", "e"])
        sensitivity = {
            **primary,
            "role": "sensitivity",
            "sample_seed": 43,
            "n_selected_samples": 3,
            "manifest_sha256": "b" * 64,
            "primary_manifest_sha256": "a" * 64,
            "primary_n_selected_samples": 4,
            "full_embedding_bank_reused": True,
            "additional_embedding_extraction": False,
            "overlap_with_primary": overlap,
        }
        self.assertTrue(validate_cohort_selection_record(sensitivity))

        index = {
            "schema_version": 1,
            "analysis_type": "cohort_sampling_sensitivity",
            "dataset": "chexpert",
            "model_id": "fixture/model",
            "primary_sample_seed": 42,
            "fold_seed": 42,
            "evaluation_seed": 42,
            "outer_folds": 5,
            "sensitivity_sample_seeds": [43],
            "max_samples": 5000,
            "embedding_bank": {
                "reused_for_every_cohort": True,
                "additional_embedding_extraction_per_seed": False,
            },
            "primary": {"sample_seed": 42},
            "sensitivities": [{
                "sample_seed": 43,
                "result_path": (
                    "cohort_sensitivity/chexpert/seed_0000000043.json"
                ),
                "manifest_sha256": "b" * 64,
            }],
        }
        self.assertTrue(validate_cohort_sensitivity_index(index))
        sensitivity["additional_embedding_extraction"] = True
        with self.assertRaisesRegex(ValueError, "cannot report"):
            validate_cohort_selection_record(sensitivity)


if __name__ == "__main__":
    unittest.main()
