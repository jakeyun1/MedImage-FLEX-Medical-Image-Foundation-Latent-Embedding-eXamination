import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet50_Weights

import scripts.model_interface as model_interface
from scripts.model_interface import HuggingFaceVisionBackend, TorchvisionBackend
from scripts.models import (
    HUGGINGFACE_REVISIONS,
    _torchvision_preprocessing_spec,
    build_backend,
)


class _TinyResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, images):
        return torch.zeros((len(images), 4), dtype=torch.float32)


class _SequenceOutputModel(torch.nn.Module):
    def forward(self, images):
        return torch.zeros((len(images), 5, 4), dtype=torch.float32)


def _fixture_spec(model_id="fixture"):
    return {
        "schema_version": 1,
        "model_id": model_id,
        "framework": "fixture",
        "weights": "fixture-weights",
        "input": {"color_mode": "RGB"},
        "embedding_output": {"method": "fixture"},
    }


class ModelPreprocessingTests(unittest.TestCase):
    def test_resnet_uses_the_exact_weight_transform(self):
        weights = ResNet50_Weights.DEFAULT
        transform = weights.transforms()
        spec = _torchvision_preprocessing_spec("resnet50", weights, transform)
        image = Image.new("RGB", (400, 200), color=(64, 128, 192))

        first = transform(image)
        second = transform(image)

        self.assertEqual(str(weights), "ResNet50_Weights.IMAGENET1K_V2")
        self.assertEqual(spec["input"]["resize_size"], [232])
        self.assertEqual(spec["input"]["crop_size"], [224])
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, second))

    @patch("scripts.models.models.resnet50")
    def test_resnet_builder_binds_weights_transform_and_output(self, resnet50):
        resnet50.return_value = _TinyResNet()

        backend = build_backend("resnet50")

        resnet50.assert_called_once_with(weights=ResNet50_Weights.DEFAULT)
        self.assertIsInstance(backend.model.fc, torch.nn.Identity)
        self.assertEqual(
            backend.preprocessing_spec["weights"],
            "ResNet50_Weights.IMAGENET1K_V2",
        )
        self.assertEqual(tuple(backend.get_transform().crop_size), (224,))

    def test_backend_rejects_non_matrix_embedding_output(self):
        backend = TorchvisionBackend(
            "fixture",
            _SequenceOutputModel(),
            torch.device("cpu"),
            transform=lambda image: image,
            preprocessing_spec=_fixture_spec(),
        )

        with self.assertRaisesRegex(ValueError, "\[batch, dimension\]"):
            backend.encode_batch(torch.zeros((2, 3, 8, 8)))

    def test_mobilenet_builder_applies_minus_one_to_one_preprocessing(self):
        fake_model = object()

        def preprocess_input(values):
            values /= 127.5
            values -= 1.0
            return values

        fake_tensorflow = SimpleNamespace(
            float32=np.float32,
            convert_to_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
            config=SimpleNamespace(set_visible_devices=lambda devices, kind: None),
            keras=SimpleNamespace(
                applications=SimpleNamespace(
                    MobileNetV2=Mock(return_value=fake_model),
                    mobilenet_v2=SimpleNamespace(preprocess_input=preprocess_input),
                )
            ),
        )

        with (
            patch.dict(sys.modules, {"tensorflow": fake_tensorflow}),
            patch.object(model_interface, "tf", fake_tensorflow),
        ):
            backend = build_backend("mobilenet_v2")
            transform = backend.get_transform()
            black = transform(Image.new("RGB", (32, 48), color=(0, 0, 0)))
            white = transform(Image.new("RGB", (48, 32), color=(255, 255, 255)))

        np.testing.assert_allclose(black, -1.0)
        np.testing.assert_allclose(white, 1.0)
        self.assertEqual(black.shape, (224, 224, 3))
        self.assertEqual(
            backend.preprocessing_spec["input"]["preprocess_function"],
            "tf.keras.applications.mobilenet_v2.preprocess_input",
        )
        fake_tensorflow.keras.applications.MobileNetV2.assert_called_once_with(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg",
            weights="imagenet",
        )

    @patch("scripts.model_interface.AutoImageProcessor.from_pretrained")
    @patch("scripts.model_interface.AutoModel.from_pretrained")
    def test_huggingface_output_key_is_explicit(self, auto_model, auto_image_processor):
        revision = "a" * 40

        class FakeModel:
            config = SimpleNamespace(_commit_hash=revision)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, pixel_values):
                return {
                    "last_hidden_state": torch.zeros((len(pixel_values), 5, 4)),
                    "pooler_output": torch.ones((len(pixel_values), 4)),
                }

        class FakeProcessor:
            do_resize = True
            size = {"height": 224, "width": 224}
            do_rescale = True
            do_normalize = True
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]

            def __call__(self, images, return_tensors):
                return {"pixel_values": torch.zeros((len(images), 3, 224, 224))}

        auto_model.return_value = FakeModel()
        auto_image_processor.return_value = FakeProcessor()
        backend = HuggingFaceVisionBackend(
            "fixture/hf-model",
            torch.device("cpu"),
            revision=revision,
            output_key="pooler_output",
        )

        output = backend.encode_batch([Image.new("RGB", (16, 16))] * 2)

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(
            backend.preprocessing_spec["embedding_output"],
            {"method": "model_forward", "key": "pooler_output"},
        )
        self.assertEqual(
            backend.preprocessing_spec["weights"]["revision"], revision
        )
        auto_model.assert_called_once_with(
            "fixture/hf-model",
            revision=revision,
            trust_remote_code=True,
        )
        auto_image_processor.assert_called_once_with(
            "fixture/hf-model",
            revision=revision,
            trust_remote_code=True,
            use_fast=True,
        )

    def test_huggingface_backend_rejects_floating_revision(self):
        with self.assertRaisesRegex(ValueError, "immutable 40-character"):
            HuggingFaceVisionBackend(
                "fixture/hf-model",
                torch.device("cpu"),
                revision="main",
                output_key="pooler_output",
            )

    def test_every_huggingface_model_has_an_immutable_revision(self):
        self.assertEqual(
            set(HUGGINGFACE_REVISIONS),
            {
                "google/medsiglip-448",
                "microsoft/rad-dino",
                "google/vit-base-patch16-224-in21k",
            },
        )
        for revision in HUGGINGFACE_REVISIONS.values():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
