import unittest

import numpy as np
import pandas as pd

from scripts.dataloading import (
    CHEXPERT_LABELS,
    _apply_chexpert_uncertainty_policy,
    validate_chexpert_uncertainty_policies,
)


class ChexpertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({
            "Cardiomegaly": [-1, np.nan],
            "Pleural Effusion": [-1, 0],
            "Edema": [-1, 1],
            "Consolidation": [-1, np.nan],
            "Atelectasis": [-1, 0],
            "Frontal/Lateral": ["Frontal", "Lateral"],
        })

    def test_finding_specific_policy(self):
        result, audit = _apply_chexpert_uncertainty_policy(
            self.metadata, "finding_specific"
        )
        self.assertEqual(result.loc[0, "Diagnosis"], [0, 1, 1, 0, 1])
        self.assertEqual(result.loc[1, "Diagnosis"], [0, 0, 1, 0, 0])
        self.assertEqual(audit["source_raw_uncertain_counts"]["Edema"], 1)
        self.assertEqual(
            audit["source_view_counts"], {"Frontal": 1, "Lateral": 1}
        )
        self.assertEqual(result.loc[0, "Cardiomegaly"], -1)
        self.assertTrue(np.isnan(result.loc[1, "Cardiomegaly"]))

    def test_sensitivity_policies(self):
        zeros, _ = _apply_chexpert_uncertainty_policy(self.metadata, "u_zeros")
        ones, _ = _apply_chexpert_uncertainty_policy(self.metadata, "u_ones")
        self.assertEqual(zeros.loc[0, "Diagnosis"], [0] * len(CHEXPERT_LABELS))
        self.assertEqual(ones.loc[0, "Diagnosis"], [1] * len(CHEXPERT_LABELS))

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uncertainty policy"):
            _apply_chexpert_uncertainty_policy(self.metadata, "guess")

    def test_sensitivity_policy_configuration_is_strict(self):
        self.assertEqual(
            validate_chexpert_uncertainty_policies(
                "finding_specific", ["u_zeros", "u_ones"]
            ),
            ("finding_specific", ("u_zeros", "u_ones")),
        )
        with self.assertRaisesRegex(ValueError, "JSON list"):
            validate_chexpert_uncertainty_policies("finding_specific", "u_zeros")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_chexpert_uncertainty_policies(
                "finding_specific", ["u_zeros", "u_zeros"]
            )
        with self.assertRaisesRegex(ValueError, "cannot also"):
            validate_chexpert_uncertainty_policies(
                "finding_specific", ["finding_specific"]
            )


if __name__ == "__main__":
    unittest.main()
