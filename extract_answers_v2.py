#!/usr/bin/env python3
"""Extract versioned answers and create a gold-free blind-review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_extraction_v2 import extract_generation, read_jsonl, write_jsonl


REVIEW_FIELDS = (
    "run_id",
    "original_id",
    "run_kind",
    "question",
    "raw_output",
    "legacy_method",
    "proposed_answer",
    "v2_method",
    "confidence",
    "evidence",
    "decision",
    "reviewed_answer",
    "reviewer",
    "review_evidence",
    "notes",
)


def build_review_rows(
    generations: list[dict[str, Any]], extractions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_run = {row["run_id"]: row for row in generations}
    rows = []
    for extraction in extractions:
        if not extraction["review_required"]:
            continue
        generation = by_run[extraction["run_id"]]
        rows.append(
            {
                "run_id": extraction["run_id"],
                "original_id": generation["original_id"],
                "run_kind": generation["run_kind"],
                "question": generation["question"],
                "raw_output": generation.get("raw_output") or "",
                "legacy_method": generation.get("answer_extraction_method") or "",
                "proposed_answer": extraction.get("answer_normalized") or "",
                "v2_method": extraction["method"],
                "confidence": extraction["confidence"],
                "evidence": extraction["evidence"],
                "decision": "",
                "reviewed_answer": "",
                "reviewer": "",
                "review_evidence": "",
                "notes": "",
            }
        )
    return rows


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def extract_all(generations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_ids = [row["run_id"] for row in generations]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run_id values in generations")
    extractions = [
        extract_generation(row, force_legacy_review=True) for row in generations
    ]
    summary = {
        "records": len(extractions),
        "review_required": sum(row["review_required"] for row in extractions),
        "result_status": "provisional"
        if any(row["review_required"] for row in extractions)
        else "final",
        "method_counts": dict(sorted(Counter(row["method"] for row in extractions).items())),
        "confidence_counts": dict(
            sorted(Counter(row["confidence"] for row in extractions).items())
        ),
        "review_status_counts": dict(
            sorted(Counter(row["review_status"] for row in extractions).items())
        ),
    }
    return extractions, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generations = read_jsonl(args.generations)
    extractions, summary = extract_all(generations)
    review_rows = build_review_rows(generations, extractions)
    if len(review_rows) != summary["review_required"]:
        raise AssertionError("Review sheet count differs from extraction summary")
    write_jsonl(args.output, extractions)
    write_review_csv(args.review_csv, review_rows)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
