import os
import tempfile
import unittest

import numpy as np

from scripts.extraction import (
    CACHE_SCHEMA_VERSION,
    _cache_paths,
    _precohort_cache_path,
    _write_embedding_cache,
    extract_embeddings,
    preprocessing_fingerprint,
)


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

    def __init__(self, batch_embeddings, preprocessing_tag="v1"):
        self.batch_embeddings = iter(batch_embeddings)
        self.calls = 0
        self.preprocessing_spec = {
            "schema_version": 1,
            "model_id": self.model_id,
            "framework": "fixture",
            "weights": "fixture-weights",
            "input": {"preprocessing_tag": preprocessing_tag},
            "embedding_output": {"method": "fixture"},
        }

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
            "ham10000", ["/data/a.jpg", "/data/c.jpg"],
            [(["a", "c"], ["/data/a.jpg", "/data/c.jpg"])],
        )
        changed_backend = FakeBackend([[[3.0], [4.0]]])
        embeddings, _, source = extract_embeddings(
            changed_loader, changed_backend, False, True
        )
        self.assertEqual(source, "computed")
        np.testing.assert_array_equal(embeddings[:, 0], [3.0, 4.0])

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
            preprocessing_sha256=np.asarray(preprocessing_fingerprint(backend)),
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

    def test_preprocessing_change_uses_a_distinct_cache(self):
        paths = ["/data/a.jpg"]
        loader = FakeDataLoader("ham10000", paths, [(["a"], paths)])
        first_backend = FakeBackend([[[1.0]]], preprocessing_tag="v1")
        first_path, _ = _cache_paths(loader, first_backend, normalize=False)
        extract_embeddings(loader, first_backend, normalize=False, cache=True)

        changed_loader = FakeDataLoader("ham10000", paths, [(["a"], paths)])
        changed_backend = FakeBackend([[[2.0]]], preprocessing_tag="v2")
        changed_path, _ = _cache_paths(
            changed_loader, changed_backend, normalize=False
        )
        embeddings, _, source = extract_embeddings(
            changed_loader, changed_backend, normalize=False, cache=True
        )

        self.assertNotEqual(first_path, changed_path)
        self.assertEqual(source, "computed")
        self.assertEqual(changed_backend.calls, 1)
        np.testing.assert_array_equal(embeddings, [[2.0]])

    def test_cache_rejects_tampered_preprocessing_fingerprint(self):
        paths = ["/data/a.jpg"]
        loader = FakeDataLoader("ham10000", paths, [])
        backend = FakeBackend([])
        filepath, _ = _cache_paths(loader, backend, normalize=False)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            schema_version=np.asarray(CACHE_SCHEMA_VERSION),
            dataset_name=np.asarray("ham10000"),
            model_id=np.asarray(backend.model_id),
            preprocessing_sha256=np.asarray("incorrect"),
            normalized=np.asarray(False),
            sample_ids=np.asarray(["a.jpg"]),
            ordered_ids_sha256=np.asarray("unused"),
            embeddings=np.asarray([[1.0]]),
        )

        with self.assertRaisesRegex(ValueError, "preprocessing mismatch"):
            extract_embeddings(loader, backend, normalize=False, cache=True)

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

    def test_runtime_separates_computation_and_cache_read(self):
        paths = ["/data/a.jpg"]
        loader = FakeDataLoader("ham10000", paths, [(["a"], paths)])
        computed = extract_embeddings(
            loader, FakeBackend([[[1.0]]]), False, True, return_runtime=True
        )
        self.assertEqual(computed[2], "computed")
        self.assertGreaterEqual(computed[3]["compute_seconds"], 0)
        self.assertEqual(computed[3]["cache_read_seconds"], 0)

        cached = extract_embeddings(
            FakeDataLoader("ham10000", paths, []), FakeBackend([]),
            False, True, return_runtime=True
        )
        self.assertEqual(cached[2], "cache")
        self.assertEqual(cached[3]["compute_seconds"], 0)
        self.assertGreaterEqual(cached[3]["cache_read_seconds"], 0)

    def test_validated_precohort_cache_is_migrated_without_inference(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        loader = FakeDataLoader("ham10000", paths, [])
        backend = FakeBackend([])
        old_path = _precohort_cache_path(loader, backend, normalize=False)
        current_path, _ = _cache_paths(loader, backend, normalize=False)
        _write_embedding_cache(
            old_path,
            np.asarray([[1.0], [2.0]], dtype=np.float32),
            ["a.jpg", "b.jpg"],
            "ham10000",
            backend.model_id,
            preprocessing_fingerprint(backend),
            False,
        )

        embeddings, aligned_paths, source, runtime = extract_embeddings(
            loader, backend, normalize=False, cache=True, return_runtime=True
        )
        self.assertEqual(source, "cache")
        self.assertEqual(backend.calls, 0)
        self.assertEqual(aligned_paths, paths)
        np.testing.assert_array_equal(embeddings[:, 0], [1.0, 2.0])
        self.assertTrue(os.path.isfile(current_path))
        self.assertEqual(runtime["compute_seconds"], 0)
        self.assertGreaterEqual(runtime["cache_write_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
