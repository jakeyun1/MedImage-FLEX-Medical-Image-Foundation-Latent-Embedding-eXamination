import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.runtime_metadata import (
    build_embedding_runtime,
    collect_backend_execution_metadata,
    collect_hardware_metadata,
)


class RuntimeMetadataTests(unittest.TestCase):
    @patch("scripts.runtime_metadata.subprocess.run")
    def test_hardware_collection_parses_rootless_gpu_query(self, run):
        run.return_value = SimpleNamespace(
            stdout="0, NVIDIA A100, GPU-abc, 40960, 550.54\n"
        )
        result = collect_hardware_metadata()

        self.assertEqual(result["schema_version"], 1)
        self.assertGreater(result["cpu"]["available_logical_cores"], 0)
        self.assertEqual(result["accelerators"]["status"], "available")
        self.assertEqual(result["accelerators"]["gpus"][0]["index"], 0)
        self.assertEqual(
            result["accelerators"]["gpus"][0]["memory_bytes"],
            40960 * 1024 * 1024,
        )

    @patch("scripts.runtime_metadata.subprocess.run", side_effect=FileNotFoundError)
    def test_hardware_collection_tolerates_missing_nvidia_smi(self, _run):
        result = collect_hardware_metadata()
        self.assertEqual(result["accelerators"]["status"], "unavailable")
        self.assertEqual(
            result["accelerators"]["reason"], "executable_not_found"
        )

    @patch(
        "scripts.runtime_metadata.subprocess.run",
        side_effect=subprocess.TimeoutExpired("nvidia-smi", 3),
    )
    def test_hardware_collection_tolerates_gpu_query_timeout(self, _run):
        result = collect_hardware_metadata()
        self.assertEqual(result["accelerators"]["reason"], "timeout")

    def test_embedding_runtime_distinguishes_cache_from_computation(self):
        computed = build_embedding_runtime(2.0, 20, 4, 2, "computed")
        cached = build_embedding_runtime(1.0, 20, 4, 2, "cache")
        self.assertTrue(computed["fresh_computation"])
        self.assertFalse(cached["fresh_computation"])
        self.assertEqual(computed["images_per_second"], 10.0)

    def test_embedding_runtime_rejects_invalid_measurements(self):
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            build_embedding_runtime(float("nan"), 20, 4, 2, "computed")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_embedding_runtime(1.0, 0, 4, 2, "computed")
        with self.assertRaisesRegex(ValueError, "source"):
            build_embedding_runtime(1.0, 20, 4, 2, "unknown")

    def test_embedding_runtime_separates_compute_and_cache_io(self):
        record = build_embedding_runtime(
            2.0, 20, 4, 2, "computed",
            timing_components={
                "total_seconds": 2.0,
                "compute_seconds": 1.5,
                "cache_read_seconds": 0.0,
                "cache_write_seconds": 0.4,
            },
        )
        timing = record["timing_components"]
        self.assertAlmostEqual(timing["overhead_seconds"], 0.1)
        self.assertAlmostEqual(timing["compute_images_per_second"], 20 / 1.5)
        with self.assertRaisesRegex(ValueError, "cannot report computation"):
            build_embedding_runtime(
                1.0, 20, 4, 2, "cache",
                timing_components={
                    "total_seconds": 1.0,
                    "compute_seconds": 0.1,
                    "cache_read_seconds": 0.9,
                    "cache_write_seconds": 0.0,
                },
            )

    def test_cpu_backend_execution_uses_model_parameter_evidence(self):
        backend = SimpleNamespace(
            device=torch.device("cpu"),
            model=torch.nn.Linear(2, 2),
            preprocessing_spec={"framework": "torchvision"},
            observed_output_device="cpu",
        )
        result = collect_backend_execution_metadata(backend)
        self.assertEqual(result["observed_device"], "cpu")
        self.assertEqual(result["device_evidence"], "model_parameter_device")
        self.assertEqual(result["observed_output_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
