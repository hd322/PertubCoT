#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path

from answer_extraction_v2 import (
    attach_extractions,
    extract_answer_v2,
    normalize_numeric_token,
)
from apply_extraction_reviews_v2 import apply_reviews
from extract_answers_v2 import build_review_rows, extract_all, write_review_csv


class AnswerExtractionV2Tests(unittest.TestCase):
    def test_historical_corpus_review_queue(self):
        path = (
            Path(__file__).resolve().parent
            / "perturbcot_qwen25_math_15b_results"
            / "merged-results"
            / "generations.jsonl"
        )
        generations = [json.loads(line) for line in path.read_text().splitlines()]
        extractions, summary = extract_all(generations)
        self.assertEqual(len(extractions), 450)
        self.assertEqual(summary["review_required"], 162)
        self.assertEqual(summary["result_status"], "provisional")

    def test_numeric_normalization(self):
        self.assertEqual(normalize_numeric_token("$1,200.00"), "1200")
        self.assertEqual(normalize_numeric_token("40%"), "40")
        self.assertEqual(normalize_numeric_token("3/2"), "1.5")
        self.assertEqual(normalize_numeric_token("1/3"), "1/3")
        self.assertEqual(normalize_numeric_token("-2.50"), "-2.5")

    def test_regression_conclusion_uses_answer_not_last_number(self):
        cases = [
            (
                "96 * 6 = 576\nTherefore, you can make 576 hash browns out of 96 potatoes.",
                "576",
            ),
            (
                "10 / 5 = 2\nTherefore, the farmer needs 2 more cows to produce 25 liters.",
                "2",
            ),
            (
                "$60 - $24 = $36\nTherefore, Arun paid $36 after applying the 40% discount.",
                "36",
            ),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = extract_answer_v2(
                    text,
                    legacy_method="last_number_fallback",
                    force_legacy_review=True,
                )
                self.assertEqual(result["answer_normalized"], expected)
                self.assertEqual(result["review_status"], "pending")

    def test_explicit_single_number_is_high_confidence(self):
        result = extract_answer_v2("work\nFinal answer: $36")
        self.assertEqual(result["answer_normalized"], "36")
        self.assertEqual(result["confidence"], "high")
        self.assertFalse(result["review_required"])

    def test_explicit_multiple_numbers_requires_review(self):
        result = extract_answer_v2("Final answer: 6 pages per day for 4 days")
        self.assertEqual(result["answer_normalized"], "6")
        self.assertEqual(result["confidence"], "medium")
        self.assertTrue(result["review_required"])

    def test_conclusion_equation_prefers_rhs(self):
        result = extract_answer_v2("Therefore, 14 + 9 + 8 = 31 slices.")
        self.assertEqual(result["answer_normalized"], "31")
        self.assertEqual(result["method"], "conclusion_equation")

    def test_no_number_is_review_required(self):
        result = extract_answer_v2("No numeric result is given.")
        self.assertIsNone(result["answer_normalized"])
        self.assertEqual(result["method"], "not_found")
        self.assertTrue(result["review_required"])

    def test_official_gate_rejects_pending_reviews(self):
        generations = [{"run_id": "r1"}]
        extractions = [
            {
                "run_id": "r1",
                "answer_normalized": "1",
                "review_status": "pending",
            }
        ]
        with self.assertRaisesRegex(ValueError, "completed reviews"):
            attach_extractions(generations, extractions, require_reviewed=True)

    def test_blind_review_round_trip(self):
        generations = [
            {
                "run_id": "r1",
                "original_id": "o1",
                "run_kind": "original_initial",
                "question": "How many?",
                "raw_output": "Therefore, there are 2 items out of 5.",
                "truncated": False,
                "answer_extraction_method": "last_number_fallback",
            },
            {
                "run_id": "r2",
                "original_id": "o2",
                "run_kind": "original_initial",
                "question": "How many?",
                "raw_output": "Final answer: 7",
                "truncated": False,
                "answer_extraction_method": "final_answer",
            },
        ]
        extractions, summary = extract_all(generations)
        self.assertEqual(summary["review_required"], 1)
        review_rows = build_review_rows(generations, extractions)
        self.assertFalse(any("gold" in key for key in review_rows[0]))
        review_rows[0].update(
            {
                "decision": "correct",
                "reviewed_answer": "2",
                "reviewer": "blind-reviewer",
                "review_evidence": "there are 2 items",
            }
        )
        resolved, resolved_summary = apply_reviews(extractions, review_rows)
        self.assertEqual(resolved_summary["result_status"], "final")
        self.assertEqual(resolved[0]["answer_normalized"], "2")
        self.assertEqual(resolved[0]["review_status"], "human_corrected")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_review_csv(path, review_rows)
            with path.open(encoding="utf-8", newline="") as handle:
                fields = set(csv.DictReader(handle).fieldnames or ())
            self.assertFalse(fields & {"gold_answer", "is_correct"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
