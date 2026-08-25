import unittest

from scripts.dataset_contracts import (
    DATASET_CONTRACTS,
    DatasetContract,
    get_dataset_contract,
)


class DatasetContractTests(unittest.TestCase):
    def test_all_supported_datasets_have_complete_contracts(self):
        self.assertEqual(
            set(DATASET_CONTRACTS),
            {"pad_ufes", "chexpert", "cbis_ddsm", "odir", "ham10000"},
        )
        for name, contract in DATASET_CONTRACTS.items():
            with self.subTest(dataset=name):
                self.assertEqual(contract.name, name)
                self.assertTrue(contract.dataset_handle)
                self.assertTrue(contract.metadata_filenames)
                self.assertTrue(contract.id_column)
                self.assertTrue(contract.label_column)
                self.assertTrue(contract.group_column)
                self.assertTrue(contract.required_source_columns)

    def test_sample_ids_are_platform_independent(self):
        chexpert = get_dataset_contract("chexpert")
        expected = "patient00001/study1/view1_frontal.jpg"
        self.assertEqual(
            chexpert.sample_id_from_path(
                "/data/train/patient00001/study1/view1_frontal.jpg"
            ),
            expected,
        )
        self.assertEqual(
            chexpert.sample_id_from_path(
                r"C:\data\train\patient00001\study1\view1_frontal.jpg"
            ),
            expected,
        )

    def test_cbis_uses_two_path_components(self):
        cbis = get_dataset_contract("cbis_ddsm")
        self.assertEqual(
            cbis.sample_id_from_path("/data/case-folder/1-1.jpg"),
            "case-folder/1-1.jpg",
        )

    def test_basename_datasets_use_one_path_component(self):
        for dataset_name in ("pad_ufes", "odir", "ham10000"):
            with self.subTest(dataset=dataset_name):
                contract = get_dataset_contract(dataset_name)
                self.assertEqual(
                    contract.sample_id_from_path(r"C:\images\sample.jpg"),
                    "sample.jpg",
                )

    def test_ham_image_directory_matching_is_case_insensitive(self):
        ham = get_dataset_contract("ham10000")
        self.assertTrue(
            ham.path_is_in_image_directory(
                "/data/HAM10000_IMAGES_PART_1/ISIC_0000001.jpg"
            )
        )
        self.assertFalse(
            ham.path_is_in_image_directory("/data/unrelated/ISIC_0000001.jpg")
        )

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            get_dataset_contract("not_a_dataset")

    def test_invalid_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported label type"):
            DatasetContract(
                name="invalid",
                dataset_handle="owner/dataset",
                metadata_filenames=("metadata.csv",),
                id_column="id",
                label_column="label",
                group_column="group",
                label_type="other",
                required_source_columns=("id", "label", "group"),
            )


if __name__ == "__main__":
    unittest.main()
