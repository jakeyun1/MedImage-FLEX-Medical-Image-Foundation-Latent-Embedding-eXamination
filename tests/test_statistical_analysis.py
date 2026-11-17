import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.data_audit import ordered_ids_sha256
from scripts.oof import write_oof_artifact
from scripts.retrieval_artifacts import (
    summary_from_retrieval_artifact,
    write_retrieval_artifact,
)
from scripts.statistical_analysis import (
    build_statistical_input_report,
    load_analysis_config,
)
from scripts.statistical_analysis.config import ConfigurationError
from scripts.statistical_analysis.inference import holm_adjust
from scripts.statistical_analysis.loaders import (
    StatisticalInputError,
    load_chexpert_sensitivity_artifacts,
)


DATASET = "synthetic"


def _oof_artifact(y_true=None):
    sample_ids = np.asarray(["s0", "s1", "s2", "s3"])
    if y_true is None:
        y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "dataset_name": np.asarray(DATASET),
        "adapter": np.asarray("logistic_regression"),
        "label_type": np.asarray("multiclass"),
        "score_type": np.asarray("probability"),
        "fold_index_base": np.asarray(0, dtype=np.int64),
        "classes": np.asarray(["negative", "positive"]),
        "sample_ids": sample_ids,
        "group_ids": np.asarray(["g0", "g1", "g2", "g3"]),
        "outer_folds": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray([0, 1, 1, 1], dtype=np.int64),
        "y_score": np.asarray([
            [0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9],
        ]),
        "ordered_sample_ids_sha256": np.asarray(
            ordered_ids_sha256(sample_ids.tolist())
        ),
    }


def _retrieval_artifact(label_indices=None):
    sample_ids = np.asarray(["s0", "s1", "s2", "s3"])
    if label_indices is None:
        label_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    first_rank = np.asarray([1, 2, 1, 2], dtype=np.int64)
    return {
        "schema_version": np.asarray(2, dtype=np.int64),
        "dataset_name": np.asarray(DATASET),
        "label_type": np.asarray("multiclass"),
        "similarity": np.asarray("cosine"),
        "normalized": np.asarray(True),
        "candidate_exclusion": np.asarray("self"),
        "classes": np.asarray(["negative", "positive"]),
        "ks": np.asarray([1, 3], dtype=np.int64),
        "sample_ids": sample_ids,
        "group_ids": np.asarray(["g0", "g1", "g2", "g3"]),
        "unit_query_indices": np.arange(4, dtype=np.int64),
        "unit_sample_ids": sample_ids.copy(),
        "unit_group_ids": np.asarray(["g0", "g1", "g2", "g3"]),
        "unit_label_indices": np.asarray(label_indices, dtype=np.int64),
        "eligible": np.ones(4, dtype=bool),
        "exclusion_reasons": np.asarray(["", "", "", ""]),
        "n_candidates": np.full(4, 3, dtype=np.int64),
        "n_relevant": np.ones(4, dtype=np.int64),
        "first_relevant_rank": first_rank,
        "average_precision": np.asarray([1.0, 0.5, 1.0, 0.5]),
        "n_relevant_at_k": np.asarray([
            [1, 1], [0, 1], [1, 1], [0, 1],
        ], dtype=np.int64),
        "hit_at_k": np.asarray([
            [True, True], [False, True], [True, True], [False, True],
        ]),
        "ordered_sample_ids_sha256": np.asarray(
            ordered_ids_sha256(sample_ids.tolist())
        ),
    }


class StatisticalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_run(self, model_id, *, y_true=None, retrieval_labels=None):
        run_dir = self.results_dir / model_id
        run_dir.mkdir()
        with (run_dir / "config_used.json").open("w", encoding="utf-8") as output:
            json.dump({"model_id": model_id}, output)

        oof = _oof_artifact(y_true=y_true)
        oof_metadata = write_oof_artifact(
            oof,
            str(run_dir / "oof" / DATASET),
            "logistic_regression.npz",
            relative_path=f"oof/{DATASET}/logistic_regression.npz",
        )
        retrieval = _retrieval_artifact(label_indices=retrieval_labels)
        retrieval_summary = summary_from_retrieval_artifact(retrieval)
        retrieval_metadata = write_retrieval_artifact(
            retrieval,
            retrieval_summary,
            str(run_dir / "retrieval" / DATASET),
            relative_path=f"retrieval/{DATASET}/queries.npz",
        )
        result = {
            "result_schema_version": 5,
            "dataset_info": {
                "dataset_name": DATASET,
                "n_samples": 4,
                "n_classes": 2,
                "classes": ["negative", "positive"],
                "label_type": "multiclass",
                "evaluation_unit": "image",
            },
            "embedding_info": {"model_id": model_id},
            "reproducibility": {
                "sha256": "cohort-sha",
                "n_samples": 4,
                "n_groups": 4,
            },
            "data_audit": {"fingerprints": {
                "retained_sample_ids_sha256": "ids-sha",
                "retained_sample_group_labels_sha256": "labels-sha",
            }},
            "oof_predictions": {
                "schema_version": 1,
                "enabled": True,
                "artifacts": {"logistic_regression": oof_metadata},
            },
            "retrieval_queries": {
                "schema_version": 2,
                "enabled": True,
                "artifact": retrieval_metadata,
            },
            "logreg_cv": {},
            "retrieval": {},
        }
        with (run_dir / f"{DATASET}.json").open("w", encoding="utf-8") as output:
            json.dump(result, output)

    def _config_dict(self):
        return {
            "schema_version": 2,
            "results_dir": "results",
            "models": [
                {"model_id": "model-a", "display_name": "Model A"},
                {"model_id": "model-b", "display_name": "Model B"},
                {"model_id": "missing", "display_name": "Missing"},
            ],
            "datasets": [DATASET],
            "reference_model_id": "model-a",
            "classification": {
                "adapter": "logistic_regression",
                "endpoint": "pooled_oof_macro_f1",
            },
            "retrieval": {
                "endpoint": "label_macro_map",
                "require_all_labels_eligible": True,
            },
            "inference": {
                "bootstrap_repetitions": 50,
                "permutation_repetitions": 99,
                "confidence_level": 0.95,
                "seed": 42,
                "pairwise_alternative": "two_sided",
                "holm_family": "within_dataset_endpoint",
            },
            "compatibility": {
                "result_schema_versions": [5, 6],
                "oof_artifact_schema_versions": [1],
                "retrieval_artifact_schema_versions": [2],
            },
            "completeness": {
                "allow_missing_models": True,
                "allow_unexpected_models": False,
            },
            "integrity": {"verify_artifact_sha256": True},
        }

    def _write_config(self, value=None):
        path = self.root / "analysis.json"
        with path.open("w", encoding="utf-8") as output:
            json.dump(self._config_dict() if value is None else value, output)
        return path

    def test_reconstructs_endpoints_and_reports_preliminary_missing_model(self):
        self._write_run("model-a")
        self._write_run("model-b")
        report = build_statistical_input_report(
            load_analysis_config(self._write_config())
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["discovery"]["missing_model_ids"], ["missing"])
        self.assertEqual(report["alignment"][DATASET]["n_models"], 2)
        self.assertEqual(len(report["endpoints"]), 2)
        self.assertAlmostEqual(
            report["endpoints"][0]["pooled_oof_macro_f1"],
            (2 / 3 + 0.8) / 2,
        )
        self.assertAlmostEqual(report["endpoints"][0]["label_macro_map"], 0.75)
        for endpoint in ("pooled_oof_macro_f1", "label_macro_map"):
            inference = report["inference"][DATASET]["endpoints"][endpoint]
            self.assertEqual(inference["multiplicity"]["method"], "holm")
            self.assertEqual(inference["multiplicity"]["family_size"], 1)
            self.assertEqual(inference["pairwise"][0]["confidence_interval"], [0.0, 0.0])
            self.assertEqual(inference["pairwise"][0]["p_value_raw"], 1.0)
            self.assertEqual(inference["pairwise"][0]["p_value_holm"], 1.0)
            self.assertEqual(inference["omnibus_permutation"]["p_value"], 1.0)

    def test_rejects_cross_model_target_misalignment(self):
        self._write_run("model-a")
        self._write_run("model-b", y_true=np.asarray([0, 1, 1, 1]))
        with self.assertRaisesRegex(
            StatisticalInputError, "Retrieval query-label units disagree"
        ):
            build_statistical_input_report(load_analysis_config(self._write_config()))

    def test_rejects_artifact_hash_mismatch(self):
        self._write_run("model-a")
        self._write_run("model-b")
        result_path = self.results_dir / "model-b" / f"{DATASET}.json"
        with result_path.open(encoding="utf-8") as input_file:
            result = json.load(input_file)
        result["oof_predictions"]["artifacts"]["logistic_regression"]["sha256"] = "0" * 64
        with result_path.open("w", encoding="utf-8") as output:
            json.dump(result, output)
        with self.assertRaisesRegex(StatisticalInputError, "SHA-256 mismatch"):
            build_statistical_input_report(load_analysis_config(self._write_config()))

    def test_rejects_unknown_configuration_key(self):
        config = copy.deepcopy(self._config_dict())
        config["classification"]["typo"] = True
        with self.assertRaisesRegex(ConfigurationError, "unknown keys"):
            load_analysis_config(self._write_config(config))

    def test_holm_adjustment_is_step_down_and_order_preserving(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_inference_is_deterministic(self):
        self._write_run("model-a")
        self._write_run("model-b")
        config = load_analysis_config(self._write_config())
        first = build_statistical_input_report(config)["inference"]
        second = build_statistical_input_report(config)["inference"]
        self.assertEqual(first, second)

    def test_rejects_nonpositive_repetition_count(self):
        config = copy.deepcopy(self._config_dict())
        config["inference"]["bootstrap_repetitions"] = 0
        with self.assertRaisesRegex(ConfigurationError, "greater than or equal to 1"):
            load_analysis_config(self._write_config(config))

    def test_chexpert_sensitivities_require_complete_matched_artifacts(self):
        run_dir = self.root / "chex-run"
        sensitivity_dir = run_dir / "chexpert_sensitivity"
        sensitivity_dir.mkdir(parents=True)
        for policy in ("u_zeros", "u_ones"):
            (sensitivity_dir / f"{policy}.json").write_text("{}")
        run = SimpleNamespace(
            model=SimpleNamespace(model_id="model-a"), run_dir=run_dir
        )
        config = SimpleNamespace(datasets=("chexpert",))
        embedding_info = {
            "model_id": "model-a",
            "embedding_dim": 3,
            "normalized": True,
            "n_source_image_embeddings": 4,
            "n_evaluation_embeddings": 4,
            "n_evaluated_embeddings": 4,
            "ordered_source_image_ids_sha256": "source",
            "ordered_evaluation_ids_sha256": "evaluation",
            "evaluation_unit": "image",
            "preprocessing_sha256": "preprocessing",
        }
        reproducibility = {"sha256": "manifest", "n_samples": 4}
        paired = {
            "sample_ids": np.asarray(["s0", "s1"]),
            "group_ids": np.asarray(["g0", "g1"]),
            "outer_folds": np.asarray([0, 1]),
        }
        primary = SimpleNamespace(
            result={
                "embedding_info": embedding_info,
                "reproducibility": reproducibility,
            },
            oof=paired,
        )

        def fake_load(_config, _run, _dataset, result_path=None):
            policy = Path(result_path).stem
            fingerprint = {"sha256": f"labels-{policy}"}
            return SimpleNamespace(
                result={
                    "embedding_info": copy.deepcopy(embedding_info),
                    "reproducibility": copy.deepcopy(reproducibility),
                    "data_audit": {
                        "sensitivity_evaluation": {"fingerprint": fingerprint}
                    },
                    "sensitivity_analysis": {
                        "uncertainty_policy": policy,
                        "matched_cohort": True,
                        "matched_patient_groups": True,
                        "matched_outer_folds": True,
                        "reused_embeddings": True,
                        "label_fingerprint": fingerprint,
                    },
                },
                oof={key: value.copy() for key, value in paired.items()},
            )

        with patch(
            "scripts.statistical_analysis.loaders.load_dataset_artifacts",
            side_effect=fake_load,
        ):
            loaded = load_chexpert_sensitivity_artifacts(
                config, [run], {"model-a": primary}
            )
        self.assertEqual(set(loaded), {"u_zeros", "u_ones"})

        (sensitivity_dir / "u_ones.json").unlink()
        with self.assertRaisesRegex(StatisticalInputError, "incomplete"):
            load_chexpert_sensitivity_artifacts(
                config, [run], {"model-a": primary}
            )


if __name__ == "__main__":
    unittest.main()
