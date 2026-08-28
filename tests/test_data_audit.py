import unittest

from scripts.data_audit import (
    AuditValidationError,
    IdentityAudit,
    ordered_ids_sha256,
    read_dataset_protocol,
    read_exclusion_policy,
    require_columns,
    resolve_required_files,
)


class DataAuditTests(unittest.TestCase):
    def test_required_columns_pass(self):
        require_columns(
            ["sample_id", "label", "patient_id"],
            ["sample_id", "label"],
            "fixture",
        )

    def test_required_columns_report_all_missing_columns(self):
        with self.assertRaisesRegex(
            AuditValidationError,
            r"\['label', 'patient_id'\]",
        ):
            require_columns(["sample_id"], ["patient_id", "label"], "fixture")

    def test_required_files_resolve_case_insensitively_in_declared_order(self):
        resolved = resolve_required_files(
            ["/data/MASS.CSV", "/data/Calc.csv"],
            ["calc.csv", "mass.csv"],
            "fixture",
        )
        self.assertEqual(resolved, ("/data/Calc.csv", "/data/MASS.CSV"))

    def test_required_files_reject_missing_or_duplicate_copies(self):
        with self.assertRaisesRegex(AuditValidationError, "missing mass.csv"):
            resolve_required_files(["/data/calc.csv"], ["calc.csv", "mass.csv"], "fixture")
        with self.assertRaisesRegex(AuditValidationError, "multiple copies of calc.csv"):
            resolve_required_files(
                ["/data/a/calc.csv", "/data/b/CALC.CSV"],
                ["calc.csv"],
                "fixture",
            )

    def test_exact_identity_match_passes(self):
        audit = IdentityAudit.from_ids(
            ["a.jpg", "b.jpg"],
            ["b.jpg", "a.jpg"],
        ).validate()
        self.assertEqual(audit.matched_unique_ids, 2)
        self.assertEqual(audit.metadata_only_ids, ())
        self.assertEqual(audit.image_only_ids, ())

    def test_duplicate_metadata_id_fails(self):
        audit = IdentityAudit.from_ids(
            ["a.jpg", "a.jpg"],
            ["a.jpg"],
        )
        with self.assertRaisesRegex(AuditValidationError, "duplicate metadata IDs"):
            audit.validate()

    def test_duplicate_image_id_fails(self):
        audit = IdentityAudit.from_ids(
            ["a.jpg"],
            ["a.jpg", "a.jpg"],
        )
        with self.assertRaisesRegex(AuditValidationError, "duplicate image IDs"):
            audit.validate()

    def test_unmatched_ids_fail_by_default(self):
        audit = IdentityAudit.from_ids(
            ["a.jpg", "metadata_only.jpg"],
            ["a.jpg", "image_only.jpg"],
        )
        with self.assertRaisesRegex(AuditValidationError, "metadata IDs without images"):
            audit.validate()

    def test_declared_image_only_records_can_pass(self):
        audit = IdentityAudit.from_ids(
            ["a.jpg"],
            ["a.jpg", "unreferenced.jpg"],
        ).validate(allow_image_only=True)
        self.assertEqual(audit.image_only_ids, ("unreferenced.jpg",))

    def test_audit_dictionary_contains_explicit_counts(self):
        result = IdentityAudit.from_ids(
            ["a.jpg"],
            ["a.jpg", "extra.jpg"],
        ).to_dict()
        self.assertEqual(result["matched_unique_ids"], 1)
        self.assertEqual(result["image_only_count"], 1)
        self.assertEqual(result["duplicate_image_count"], 0)

    def test_ordered_hash_is_stable_and_order_sensitive(self):
        first = ordered_ids_sha256(["a", "b"])
        self.assertEqual(first, ordered_ids_sha256(["a", "b"]))
        self.assertNotEqual(first, ordered_ids_sha256(["b", "a"]))
        self.assertNotEqual(
            ordered_ids_sha256(["ab", "c"]),
            ordered_ids_sha256(["a", "bc"]),
        )

    def test_committed_exclusion_policies_are_strict_and_counted(self):
        expected = {
            "pad_ufes_exclusions.csv": 27,
            "ham10000_exclusions.csv": 2,
            "chexpert_exclusions.csv": 28,
        }
        for filename, count in expected.items():
            with self.subTest(filename=filename):
                policy, audit = read_exclusion_policy(filename)
                self.assertEqual(len(policy), count)
                self.assertEqual(audit["excluded_samples"], count)
                self.assertEqual(len(audit["sha256"]), 64)

    def test_every_dataset_has_an_audited_protocol(self):
        for dataset_name in (
            "pad_ufes", "ham10000", "chexpert", "cbis_ddsm", "odir"
        ):
            with self.subTest(dataset=dataset_name):
                protocol, manifest = read_dataset_protocol(dataset_name)
                self.assertGreater(protocol["retained_image_files"], 0)
                self.assertIn("/versions/", protocol["dataset_handle"])
                self.assertEqual(len(manifest["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
