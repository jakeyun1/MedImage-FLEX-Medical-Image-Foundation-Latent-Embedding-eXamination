import os
import tempfile
import unittest

import pandas as pd

from scripts.cbis_ddsm import CBIS_LABELS, prepare_cbis_ddsm_full_mammograms


def _row(patient, laterality, view, abnormality, pathology, source_path):
    return {
        "patient_id": patient,
        "left or right breast": laterality,
        "image view": view,
        "abnormality type": abnormality,
        "pathology": pathology,
        "jpg_fullMammo_img_path": source_path,
    }


class CbisDdsmPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _image(self, folder, content=b"same-image"):
        directory = os.path.join(self.temporary_directory.name, folder)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "full-mammogram-images-img_0-1.jpg")
        with open(path, "wb") as file:
            file.write(content)
        return path

    def test_aggregates_labels_and_collapses_verified_duplicate_aliases(self):
        calc_path = self._image("Calc_Training_P_00001_LEFT_CC-series")
        mass_path = self._image("Mass_Test_P_00001_LEFT_CC-series")
        metadata = pd.DataFrame([
            _row(
                "P_00001", "LEFT", "CC", "calcification", "MALIGNANT",
                f"jpg_img/{os.path.basename(os.path.dirname(calc_path))}/{os.path.basename(calc_path)}",
            ),
            _row(
                "P_00001", "LEFT", "CC", "mass", "BENIGN_WITHOUT_CALLBACK",
                f"jpg_img/{os.path.basename(os.path.dirname(mass_path))}/{os.path.basename(mass_path)}",
            ),
            _row(
                "P_00001", "LEFT", "CC", "mass", "MALIGNANT",
                f"jpg_img/{os.path.basename(os.path.dirname(mass_path))}/{os.path.basename(mass_path)}",
            ),
        ])

        result, selected_paths, audit = prepare_cbis_ddsm_full_mammograms(
            metadata, [mass_path, calc_path]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.loc[0, "pathology"],
            ["calcification_MALIGNANT", "mass_BENIGN", "mass_MALIGNANT"],
        )
        self.assertEqual(selected_paths, [calc_path])
        self.assertEqual(result.loc[0, "source_partitions"], ["Test", "Training"])
        self.assertEqual(audit["duplicate_image_aliases_collapsed"], 1)
        self.assertEqual(audit["benign_without_callback_rows_collapsed"], 1)
        self.assertEqual(audit["representation"], "third_party_jpeg_derivative")
        self.assertEqual(
            audit["source_partition_policy"],
            "pooled_for_patient_grouped_cross_validation",
        )
        self.assertTrue(set(result.loc[0, "pathology"]).issubset(CBIS_LABELS))

    def test_rejects_different_image_contents_for_one_scan(self):
        calc_path = self._image("Calc_Training_P_00001_LEFT_CC-series", b"calc")
        mass_path = self._image("Mass_Training_P_00001_LEFT_CC-series", b"mass")
        metadata = pd.DataFrame([
            _row(
                "P_00001", "LEFT", "CC", "calcification", "BENIGN",
                f"jpg_img/{os.path.basename(os.path.dirname(calc_path))}/{os.path.basename(calc_path)}",
            ),
            _row(
                "P_00001", "LEFT", "CC", "mass", "BENIGN",
                f"jpg_img/{os.path.basename(os.path.dirname(mass_path))}/{os.path.basename(mass_path)}",
            ),
        ])

        with self.assertRaisesRegex(ValueError, "different image contents"):
            prepare_cbis_ddsm_full_mammograms(metadata, [calc_path, mass_path])

    def test_rejects_missing_image(self):
        metadata = pd.DataFrame([
            _row(
                "P_00001", "LEFT", "CC", "mass", "BENIGN",
                "jpg_img/Mass_Training_P_00001_LEFT_CC-series/full-mammogram-images-img_0-1.jpg",
            )
        ])
        with self.assertRaisesRegex(ValueError, "missing full mammograms"):
            prepare_cbis_ddsm_full_mammograms(metadata, [])

    def test_rejects_scan_id_disagreement(self):
        image_path = self._image("Mass_Training_P_00002_LEFT_CC-series")
        metadata = pd.DataFrame([
            _row(
                "P_00001", "LEFT", "CC", "mass", "BENIGN",
                f"jpg_img/{os.path.basename(os.path.dirname(image_path))}/{os.path.basename(image_path)}",
            )
        ])
        with self.assertRaisesRegex(ValueError, "do not match image-path scan IDs"):
            prepare_cbis_ddsm_full_mammograms(metadata, [image_path])


if __name__ == "__main__":
    unittest.main()
