import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from scripts.model_interface import CallableEmbeddingBackend
from scripts.models import (
    BIOMEDCLIP_REVISION,
    MED_FLAMINGO_REVISION,
    _build_biomedclip_backend,
    _build_cxr_foundation_backend,
    _build_med_flamingo_backend,
)


def _spec(model_id="fixture"):
    return {
        "schema_version": 1,
        "model_id": model_id,
        "framework": "fixture",
        "weights": "fixture",
        "input": {"color_mode": "RGB"},
        "embedding_output": {"method": "fixture"},
    }


class CallableEmbeddingBackendTests(unittest.TestCase):
    def test_enforces_declared_dimension(self):
        backend = CallableEmbeddingBackend(
            model_id="fixture",
            device=torch.device("cpu"),
            transform=None,
            encode_fn=lambda images: np.zeros((len(images), 3), dtype=np.float32),
            embedding_dim=4,
            preprocessing_spec=_spec(),
        )

        with self.assertRaisesRegex(ValueError, "declared embedding dimension"):
            backend.encode_batch([object(), object()])

    def test_converts_matrix_output_to_tensor(self):
        backend = CallableEmbeddingBackend(
            model_id="fixture",
            device=torch.device("cpu"),
            transform=None,
            encode_fn=lambda images: np.ones((len(images), 4), dtype=np.float32),
            embedding_dim=4,
            preprocessing_spec=_spec(),
        )

        output = backend.encode_batch([object(), object()])

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 4))


class SpecialModelBackendTests(unittest.TestCase):
    def test_biomedclip_uses_pinned_openclip_checkpoint_and_native_transform(self):
        class FakeBiomedCLIP(torch.nn.Module):
            def encode_image(self, images, normalize=False):
                self.last_normalize = normalize
                return torch.ones((len(images), 3), device=images.device)

        fake_model = FakeBiomedCLIP()
        transform = lambda image: torch.ones((3, 8, 8))
        create_model = Mock(return_value=(fake_model, object(), transform))
        model_configs = {}
        open_clip = types.ModuleType("open_clip")
        open_clip.create_model_and_transforms = create_model
        open_clip_factory = types.ModuleType("open_clip.factory")
        open_clip_factory._MODEL_CONFIGS = model_configs

        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = os.path.join(temporary_dir, "open_clip_config.json")
            checkpoint_path = os.path.join(
                temporary_dir, "open_clip_pytorch_model.bin"
            )
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump({
                    "model_cfg": {"embed_dim": 3},
                    "preprocess_cfg": {"size": 224},
                }, config_file)

            def download(repo_id, filename, revision):
                self.assertEqual(revision, BIOMEDCLIP_REVISION)
                if filename == "open_clip_config.json":
                    return config_path
                return checkpoint_path

            with (
                patch.dict(sys.modules, {
                    "open_clip": open_clip,
                    "open_clip.factory": open_clip_factory,
                }),
                patch("huggingface_hub.hf_hub_download", side_effect=download),
            ):
                backend = _build_biomedclip_backend(
                    "biomedclip", torch.device("cpu")
                )

        output = backend.encode_batch((torch.ones((3, 8, 8)),) * 2)

        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertFalse(fake_model.last_normalize)
        self.assertIs(backend.get_transform(), transform)
        self.assertEqual(
            backend.preprocessing_spec["weights"]["revision"],
            BIOMEDCLIP_REVISION,
        )

    def test_cxr_foundation_uses_general_embedding_and_mean_pools_tokens(self):
        seen_modes = []

        class FakeClient:
            def get_image_embeddings_from_images(self, images):
                seen_modes.extend(image.mode for image in images)
                tokens = np.stack([
                    np.ones(768, dtype=np.float32),
                    np.full(768, 3.0, dtype=np.float32),
                ])
                return [
                    SimpleNamespace(error=None, general_img_emb=tokens)
                    for _ in images
                ]

        client_constructor = Mock(return_value=FakeClient())
        clientside = types.ModuleType("clientside")
        clients = types.ModuleType("clientside.clients")
        clients.HuggingFaceCxrClient = client_constructor
        fake_tensorflow = types.ModuleType("tensorflow")
        fake_tensorflow.saved_model = SimpleNamespace(load=Mock(return_value=object()))

        with (
            patch.dict(os.environ, {
                "MEDIMAGE_FLEX_CXR_FOUNDATION_REVISION": "c" * 40,
            }),
            patch.dict(sys.modules, {
                "clientside": clientside,
                "clientside.clients": clients,
                "tensorflow": fake_tensorflow,
            }),
            patch(
                "huggingface_hub.snapshot_download",
                return_value="/tmp/fake-cxr-snapshot",
            ),
        ):
            backend = _build_cxr_foundation_backend("cxr-foundation")

        output = backend.encode_batch([
            Image.new("RGB", (8, 8)),
            Image.new("RGB", (8, 8)),
        ])

        self.assertEqual(tuple(output.shape), (2, 768))
        self.assertTrue(torch.allclose(output, torch.full_like(output, 2.0)))
        self.assertEqual(seen_modes, ["L", "L"])
        self.assertIsNone(backend.get_transform())

    def test_med_flamingo_uses_post_perceiver_mean_embedding(self):
        class FakeVisionEncoder(torch.nn.Module):
            def forward(self, images):
                tokens = torch.ones((len(images), 5, 1024), device=images.device)
                return torch.zeros((len(images), 1024)), tokens

        class FakePerceiver(torch.nn.Module):
            def forward(self, tokens):
                self.input_shape = tuple(tokens.shape)
                return torch.full(
                    (len(tokens), 1, 64, 1024),
                    2.0,
                    device=tokens.device,
                )

        class FakeModel:
            def __init__(self):
                self.vision_encoder = FakeVisionEncoder()
                self.perceiver = FakePerceiver()

            def load_state_dict(self, state_dict, strict):
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        fake_model = FakeModel()
        processor = lambda image: torch.ones((3, 8, 8))
        create_model = Mock(return_value=(fake_model, processor, object()))
        open_flamingo = types.ModuleType("open_flamingo")
        open_flamingo.create_model_and_transforms = create_model

        with tempfile.TemporaryDirectory() as llama_path:
            with (
                patch.dict(os.environ, {
                    "MEDIMAGE_FLEX_MEDFLAMINGO_LLAMA_PATH": llama_path,
                }),
                patch.dict(sys.modules, {"open_flamingo": open_flamingo}),
                patch(
                    "huggingface_hub.hf_hub_download",
                    return_value="/tmp/fake-med-flamingo.pt",
                ) as download,
                patch("scripts.models.torch.load", return_value={}),
            ):
                backend = _build_med_flamingo_backend(
                    "medflamingo", torch.device("cpu")
                )

        output = backend.encode_batch([
            Image.new("RGB", (8, 8)),
            Image.new("RGB", (8, 8)),
        ])

        self.assertEqual(tuple(output.shape), (2, 1024))
        self.assertTrue(torch.allclose(output, torch.full_like(output, 2.0)))
        self.assertEqual(fake_model.perceiver.input_shape, (2, 1, 1, 5, 1024))
        self.assertEqual(
            backend.preprocessing_spec["weights"]["revision"],
            MED_FLAMINGO_REVISION,
        )
        self.assertEqual(download.call_args.kwargs["revision"], MED_FLAMINGO_REVISION)


if __name__ == "__main__":
    unittest.main()
