#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

try:
    from .adaptive_core import (
        answer_risk,
        build_group_features,
        normalized_entropy,
        validate_config,
    )
    from .evaluate_adaptive import simulate_sequential, verification_bundle
except ImportError:  # Direct script execution.
    from adaptive_core import (
        answer_risk,
        build_group_features,
        normalized_entropy,
        validate_config,
    )
    from evaluate_adaptive import simulate_sequential, verification_bundle


BASE = Path(__file__).resolve().parent


def extraction(answer, confidence="high", truncated=False):
    return {
        "answer_normalized": answer,
        "answer_raw": answer,
        "confidence": confidence,
        "truncated": truncated,
        "review_status": "auto_accepted",
    }


def generation(
    run_id,
    answer,
    *,
    run_kind,
    version_type,
    perturbation_type=None,
    repeat_index=0,
    tokens=10,
):
    return {
        "run_id": run_id,
        "original_id": "o1",
        "split": "dev",
        "run_kind": run_kind,
        "version_type": version_type,
        "perturbation_type": perturbation_type,
        "repeat_index": repeat_index,
        "status": "ok",
        "question": "A simple problem.",
        "raw_output": "The answer is stated without an equation.",
        "input_tokens": tokens // 2,
        "output_tokens": tokens - tokens // 2,
        "truncated": False,
        "answer_extraction_v2": extraction(answer),
    }


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((BASE / "config.json").read_text())

    def test_config_and_entropy(self):
        validate_config(self.config)
        self.assertEqual(normalized_entropy(["1", "1"]), 0.0)
        self.assertAlmostEqual(normalized_entropy(["1", "2"]), 1.0)
        risk = answer_risk("1", ["1", "2"])
        self.assertAlmostEqual(risk["original_support"], 0.5)
        self.assertAlmostEqual(risk["risk"], 0.75)

    def test_features_only_use_observed_variants(self):
        order = tuple(self.config["perturbation_order"])
        rows = [generation("o-r0", "10", run_kind="original_initial", version_type="original")]
        for name in order:
            rows.append(
                generation(
                    "v-" + name,
                    "10",
                    run_kind="variant_initial",
                    version_type="variant",
                    perturbation_type=name,
                )
            )
        features = build_group_features(rows, order, self.config)
        self.assertEqual(list(features["stages"][1]["answers"]), ["original", order[0]])
        self.assertNotIn(order[1], features["stages"][1]["answers"])
        self.assertFalse(features["stages"][1]["safe_stop_eligible"])
        self.assertTrue(features["stages"][2]["safe_stop_eligible"])
        self.assertEqual(features["stages"][2]["cumulative_monitor_tokens"], 20)

    def test_sequential_safe_stop_and_threshold(self):
        feature = {
            "original_id": "o1",
            "split": "dev",
            "original_answer": "10",
            "stages": [
                {
                    "stage": 0,
                    "risk_scores": {"full": 0.0},
                    "force_verify": False,
                    "safe_stop_eligible": False,
                    "cumulative_monitor_tokens": 0,
                },
                {
                    "stage": 1,
                    "risk_scores": {"full": 0.1},
                    "force_verify": False,
                    "safe_stop_eligible": False,
                    "cumulative_monitor_tokens": 10,
                },
                {
                    "stage": 2,
                    "risk_scores": {"full": 0.0},
                    "force_verify": False,
                    "safe_stop_eligible": True,
                    "cumulative_monitor_tokens": 20,
                },
            ],
        }
        bundle = {"final_answer": "11", "token_cost": 40}
        safe = simulate_sequential(feature, bundle, 0.5, "full")
        self.assertFalse(safe["triggered_verification"])
        self.assertEqual(safe["stop_stage"], 2)
        feature["stages"][1]["risk_scores"]["full"] = 0.5
        triggered = simulate_sequential(feature, bundle, 0.5, "full")
        self.assertTrue(triggered["triggered_verification"])
        self.assertEqual(triggered["final_answer"], "11")
        self.assertEqual(triggered["total_extra_tokens"], 50)

    def test_verification_tie_keeps_initial(self):
        answers = ["10", "11", "11", "12", "12"]
        rows = [
            generation(
                f"o-r{index}",
                answer,
                run_kind="original_initial" if index == 0 else "original_repeat",
                version_type="original",
                repeat_index=index,
                tokens=10,
            )
            for index, answer in enumerate(answers)
        ]
        bundle = verification_bundle(rows, 3)
        self.assertEqual(bundle["final_answer"], "10")
        self.assertEqual(bundle["update_reason"], "mode_tie_keep_initial")
        self.assertEqual(bundle["token_cost"], 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
