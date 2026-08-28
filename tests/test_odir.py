import unittest

import numpy as np
import pandas as pd

from scripts.odir import ODIR_LABELS, pool_odir_patient_embeddings, prepare_odir_eye_records
from scripts.tests import prepare_data_multilabel


def _patient(patient_id, labels):
    return {
        "ID": patient_id,
        "Patient Age": 60,
        "Patient Sex": "Male",
        "Left-Fundus": f"{patient_id}_left.jpg",
        "Right-Fundus": f"{patient_id}_right.jpg",
        "Left-Diagnostic Keywords": "normal fundus",
        "Right-Diagnostic Keywords": "normal fundus",
        **dict(zip(ODIR_LABELS, labels)),
    }


class OdirPreparationTests(unittest.TestCase):
    def test_expands_and_pools_one_multilabel_patient_sample(self):
        source = pd.DataFrame([
            _patient(1, [1, 0, 0, 0, 0, 0, 0, 0]),
            _patient(2, [0, 1, 0, 0, 0, 0, 0, 1]),
        ])
        eyes, source_audit = prepare_odir_eye_records(source)
        self.assertEqual(len(eyes), 4)
        self.assertEqual(source_audit["multilabel_patients"], 1)

        paths = ["/images/2_right.jpg", "/images/1_left.jpg",
                 "/images/2_left.jpg", "/images/1_right.jpg"]
        embeddings = np.asarray([[0, 2], [2, 0], [0, 2], [2, 0]], dtype=float)
        pooled, metadata, patient_ids, audit = pool_odir_patient_embeddings(
            embeddings, eyes, paths, normalize=True
        )

        self.assertEqual(patient_ids, ["1", "2"])
        np.testing.assert_allclose(pooled, [[1, 0], [0, 1]])
        self.assertEqual(metadata.loc[1, "Diagnosis"], [0, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(audit["evaluation_samples"], 2)

        X, y, classes, is_multilabel, sample_ids = prepare_data_multilabel(
            "odir",
            pooled,
            metadata,
            patient_ids,
            "ID",
            "Diagnosis",
            return_sample_ids=True,
            sample_ids=patient_ids,
        )
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2, 8))
        self.assertEqual(classes, list(ODIR_LABELS))
        self.assertTrue(is_multilabel)
        self.assertEqual(sample_ids.tolist(), ["1", "2"])

    def test_rejects_nonbinary_labels(self):
        source = pd.DataFrame([_patient(1, [2, 0, 0, 0, 0, 0, 0, 0])])
        with self.assertRaisesRegex(ValueError, "binary"):
            prepare_odir_eye_records(source)

    def test_rejects_missing_eye_embedding(self):
        eyes, _ = prepare_odir_eye_records(pd.DataFrame([
            _patient(1, [1, 0, 0, 0, 0, 0, 0, 0])
        ]))
        with self.assertRaisesRegex(ValueError, "do not match metadata"):
            pool_odir_patient_embeddings(
                np.asarray([[1.0, 0.0]]), eyes, ["/images/1_left.jpg"]
            )


if __name__ == "__main__":
    unittest.main()
