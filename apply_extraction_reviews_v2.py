#!/usr/bin/env python3
"""Validate and apply completed blind reviews to v2 answer extractions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_extraction_v2 import normalize_numeric_token, read_jsonl, write_jsonl


DECISIONS = {"accept", "correct", "unresolved"}
FORBIDDEN_REVIEW_FIELDS = {"gold_answer", "source_solution", "is_correct"}


def read_reviews(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        forbidden = fields & FORBIDDEN_REVIEW_FIELDS
        if forbidden:
            raise ValueError(f"Review sheet contains forbidden label fields: {sorted(forbidden)}")
        return list(reader)


def apply_reviews(
    extractions: list[dict[str, Any]], reviews: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extraction_ids = [row["run_id"] for row in extractions]
    if len(set(extraction_ids)) != len(extraction_ids):
        raise ValueError("Duplicate run_id values in extractions")
    required = {row["run_id"] for row in extractions if row.get("review_required")}
    review_ids = [row.get("run_id", "") for row in reviews]
    if len(set(review_ids)) != len(review_ids):
        raise ValueError("Duplicate run_id values in review sheet")
    if set(review_ids) != required:
        missing = sorted(required - set(review_ids))
        unexpected = sorted(set(review_ids) - required)
        raise ValueError(f"Review run_id mismatch: missing={missing}, unexpected={unexpected}")

    review_by_run = {row["run_id"]: row for row in reviews}
    resolved = []
    decisions = Counter()
    for extraction in extractions:
        result = dict(extraction)
        result["automatic_answer_raw"] = extraction.get("answer_raw")
        result["automatic_answer_normalized"] = extraction.get("answer_normalized")
        if not extraction.get("review_required"):
            resolved.append(result)
            continue
        review = review_by_run[extraction["run_id"]]
        decision = review.get("decision", "").strip().lower()
        reviewer = review.get("reviewer", "").strip()
        if decision not in DECISIONS:
            raise ValueError(f"Invalid review decision for {extraction['run_id']}: {decision!r}")
        if not reviewer:
            raise ValueError(f"Missing reviewer for {extraction['run_id']}")
        decisions[decision] += 1
        if decision == "accept":
            if extraction.get("answer_normalized") is None:
                raise ValueError(f"Cannot accept a missing candidate for {extraction['run_id']}")
            status = "human_accepted"
        elif decision == "correct":
            reviewed_raw = review.get("reviewed_answer", "").strip()
            normalized = normalize_numeric_token(reviewed_raw)
            if normalized is None:
                raise ValueError(f"Invalid corrected answer for {extraction['run_id']}")
            if not review.get("review_evidence", "").strip():
                raise ValueError(f"Corrected answer requires evidence for {extraction['run_id']}")
            result["answer_raw"] = reviewed_raw
            result["answer_normalized"] = normalized
            result["evidence"] = review["review_evidence"].strip()
            status = "human_corrected"
        else:
            result["answer_raw"] = None
            result["answer_normalized"] = None
            status = "human_unresolved"
        result.update(
            {
                "review_status": status,
                "reviewer": reviewer,
                "review_decision": decision,
                "review_notes": review.get("notes", "").strip(),
            }
        )
        resolved.append(result)

    summary = {
        "records": len(resolved),
        "reviewed": len(required),
        "result_status": "final",
        "decision_counts": dict(sorted(decisions.items())),
        "review_status_counts": dict(
            sorted(Counter(row["review_status"] for row in resolved).items())
        ),
    }
    return resolved, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved, summary = apply_reviews(
        read_jsonl(args.extractions), read_reviews(args.reviews)
    )
    write_jsonl(args.output, resolved)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
