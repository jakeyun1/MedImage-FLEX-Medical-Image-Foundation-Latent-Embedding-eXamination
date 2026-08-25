import os
import tempfile
import unittest

import numpy as np

from scripts.extraction import CACHE_SCHEMA_VERSION, _cache_paths, extract_embeddings


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeDataset:
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)


class FakeDataLoader:
    def __init__(self, dataset_name, image_paths, batches):
        self.dataset_name = dataset_name
        self.dataset = FakeDataset(image_paths)
        self.batches = list(batches)

    def __iter__(self):
        return iter(self.batches)


class FakeBackend:
    model_id = "research/model.v1"

    def __init__(self, batch_embeddings):
        self.batch_embeddings = iter(batch_embeddings)
        self.calls = 0

    def encode_batch(self, images):
        self.calls += 1
        return FakeTensor(next(self.batch_embeddings))


class ExtractionCacheTests(unittest.TestCase):
    def setUp(self):
        self.previous_directory = os.getcwd()
        self.temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self.temporary_directory.name)

    def tearDown(self):
        os.chdir(self.previous_directory)
        self.temporary_directory.cleanup()

    def test_cache_preserves_shuffled_embedding_alignment(self):
        current_paths = ["/data/a.jpg", "/data/b.jpg", "/data/c.jpg"]
        extraction_order = [current_paths[1], current_paths[0], current_paths[2]]
        first_loader = FakeDataLoader(
            "ham10000",
            current_paths,
            [(["b", "a", "c"], extraction_order)],
        )
        first_backend = FakeBackend([[[2.0], [1.0], [3.0]]])

        embeddings, paths, source = extract_embeddings(
            first_loader, first_backend, normalize=False, cache=True
        )
        self.assertEqual(source, "computed")
        self.assertEqual(paths, extraction_order)
        np.testing.assert_array_equal(embeddings[:, 0], [2.0, 1.0, 3.0])

        relocated_paths = ["/relocated/a.jpg", "/relocated/b.jpg", "/relocated/c.jpg"]
        second_loader = FakeDataLoader("ham10000", relocated_paths, [])
        second_backend = FakeBackend([])
        cached_embeddings, cached_paths, source = extract_embeddings(
            second_loader, second_backend, normalize=False, cache=True
        )

        self.assertEqual(source, "cache")
        self.assertEqual(second_backend.calls, 0)
        self.assertEqual(
            cached_paths,
            [relocated_paths[1], relocated_paths[0], relocated_paths[2]],
        )
        np.testing.assert_array_equal(cached_embeddings, embeddings)

    def test_cache_rejects_changed_dataset_identity(self):
        original_paths = ["/data/a.jpg", "/data/b.jpg"]
        loader = FakeDataLoader(
            "ham10000",
            original_paths,
            [(["a", "b"], original_paths)],
        )
        extract_embeddings(loader, FakeBackend([[[1.0], [2.0]]]), False, True)

        changed_loader = FakeDataLoader(
            "ham10000", ["/data/a.jpg", "/data/c.jpg"], []
        )
        with self.assertRaisesRegex(ValueError, "sample IDs do not match"):
            extract_embeddings(changed_loader, FakeBackend([]), False, True)

    def test_cache_rejects_tampered_sample_id_hash(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        loader = FakeDataLoader("ham10000", paths, [])
        backend = FakeBackend([])
        filepath, _ = _cache_paths(loader, backend, normalize=False)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            schema_version=np.asarray(CACHE_SCHEMA_VERSION),
            dataset_name=np.asarray("ham10000"),
            model_id=np.asarray(backend.model_id),
            normalized=np.asarray(False),
            sample_ids=np.asarray(["a.jpg", "b.jpg"]),
            ordered_ids_sha256=np.asarray("incorrect"),
            embeddings=np.asarray([[1.0], [2.0]]),
        )

        with self.assertRaisesRegex(ValueError, "sample-ID hash mismatch"):
            extract_embeddings(loader, backend, normalize=False, cache=True)

    def test_normalized_and_unnormalized_caches_are_distinct(self):
        loader = FakeDataLoader("ham10000", ["/data/a.jpg"], [])
        backend = FakeBackend([])
        normalized, _ = _cache_paths(loader, backend, normalize=True)
        unnormalized, _ = _cache_paths(loader, backend, normalize=False)
        self.assertNotEqual(normalized, unnormalized)

    def test_extraction_rejects_batch_row_mismatch(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        loader = FakeDataLoader(
            "ham10000", paths, [(["a", "b"], paths)]
        )
        backend = FakeBackend([[[1.0]]])

        with self.assertRaisesRegex(ValueError, "batch image paths"):
            extract_embeddings(loader, backend, normalize=False, cache=False)

    def test_legacy_matrix_cache_is_not_reused(self):
        paths = ["/data/a.jpg"]
        loader = FakeDataLoader(
            "ham10000", paths, [(["a"], paths)]
        )
        backend = FakeBackend([[[1.0]]])
        _, legacy_filepath = _cache_paths(loader, backend, normalize=False)
        os.makedirs(os.path.dirname(legacy_filepath), exist_ok=True)
        np.save(legacy_filepath, np.asarray([[999.0]]))

        embeddings, _, source = extract_embeddings(
            loader, backend, normalize=False, cache=True
        )
        self.assertEqual(source, "computed")
        np.testing.assert_array_equal(embeddings, [[1.0]])


if __name__ == "__main__":
    unittest.main()
