import os
import tempfile
import unittest

import pandas as pd

from scripts.reproducibility import prepare_experiment_manifest


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


if __name__ == "__main__":
    unittest.main()
