import unittest
from types import SimpleNamespace

import pandas as pd

from scripts.dataloading import GeneralDataset, subset_dataloader


class DataloaderSubsetTests(unittest.TestCase):
    def setUp(self):
        dataset = GeneralDataset(
            pd.DataFrame({"image_id": ["a.jpg", "b.jpg", "c.jpg"]}),
            ["/images/a.jpg", "/images/b.jpg", "/images/c.jpg"],
        )
        self.loader = SimpleNamespace(
            dataset_name="ham10000",
            dataset=dataset,
            data_audit={"dataset": "ham10000"},
        )

    def test_rebuilds_exact_ordered_cohort_before_iteration(self):
        selected = subset_dataloader(
            self.loader, ["c.jpg", "a.jpg"], batch_size=2
        )
        self.assertEqual(
            selected.dataset.image_paths,
            ["/images/c.jpg", "/images/a.jpg"],
        )
        self.assertEqual(
            selected.data_audit["extraction_cohort"],
            {
                "selection_applied_before_image_decoding": True,
                "source_image_count": 3,
                "selected_image_count": 2,
            },
        )
        self.assertNotIn("extraction_cohort", self.loader.data_audit)

    def test_rejects_missing_or_duplicate_selected_ids(self):
        with self.assertRaisesRegex(ValueError, "missing image files"):
            subset_dataloader(self.loader, ["missing.jpg"], batch_size=1)
        with self.assertRaisesRegex(ValueError, "nonempty and unique"):
            subset_dataloader(self.loader, ["a.jpg", "a.jpg"], batch_size=1)


if __name__ == "__main__":
    unittest.main()
