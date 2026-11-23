#!/usr/bin/env python3
"""Deterministic, auditable answer extraction for PerturbCoT."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


EXTRACTOR_VERSION = "2.0.0"
NUMBER_PATTERN = (
    r"[-+]?(?:[$£€]\s*)?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?%?"
)
NUMBER_RE = re.compile(NUMBER_PATTERN)
EXPLICIT_PATTERNS = (
    (
        "final_answer",
        re.compile(
            r"(?im)^\s*(?:therefore,?\s*)?(?:the\s+)?final\s+answer\s*"
            r"(?:is|:|=)\s*(?P<body>.+?)\s*$"
        ),
    ),
    (
        "answer_statement",
        re.compile(r"(?im)^\s*answer\s*(?:is|:|=)\s*(?P<body>.+?)\s*$"),
    ),
    ("boxed", re.compile(r"\\boxed\s*\{(?P<body>[^{}]+)\}")),
    ("hash_answer", re.compile(r"####\s*(?P<body>[^\n]+)")),
)
CONCLUSION_RE = re.compile(
    r"(?im)^.*\b(?:therefore|thus|hence|consequently|so)\b[^\n]*$"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _finite_decimal(value: Fraction) -> str | None:
    denominator = value.denominator
    reduced = denominator
    for factor in (2, 5):
        while reduced % factor == 0:
            reduced //= factor
    if reduced != 1:
        return None
    number = Decimal(value.numerator) / Decimal(denominator)
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def normalize_numeric_token(value: Any) -> str | None:
    """Return a canonical GSM8K numeric value without presentation symbols."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[$£€\s]", "", text)
    if text.endswith("%"):
        text = text[:-1]
    if not text:
        return None
    try:
        if "/" in text:
            number = Fraction(text)
        else:
            number = Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if number.denominator == 1:
        return str(number.numerator)
    finite = _finite_decimal(number)
    return finite if finite is not None else f"{number.numerator}/{number.denominator}"


def _candidate(text: str, match: re.Match[str], method: str) -> dict[str, Any] | None:
    raw = match.group(0).strip()
    normalized = normalize_numeric_token(raw)
    if normalized is None:
        return None
    return {
        "raw": raw,
        "normalized": normalized,
        "start": match.start(),
        "end": match.end(),
        "method": method,
    }


def _number_candidates(text: str, offset: int = 0, method: str = "number") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in NUMBER_RE.finditer(text):
        candidate = _candidate(text, match, method)
        if candidate is not None:
            candidate["start"] += offset
            candidate["end"] += offset
            candidates.append(candidate)
    return candidates


def _line_evidence(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def _equality_candidates(text: str, offset: int = 0) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pattern = re.compile(r"=\s*(?P<answer>" + NUMBER_PATTERN + r")")
    for match in pattern.finditer(text):
        answer = match.group("answer")
        answer_start = match.start("answer")
        number_match = NUMBER_RE.search(answer)
        if number_match is None:
            continue
        candidate = _candidate(answer, number_match, "equation_rhs")
        if candidate is None:
            continue
        candidate["start"] = offset + answer_start + number_match.start()
        candidate["end"] = offset + answer_start + number_match.end()
        results.append(candidate)
    return results


def _selection(
    text: str,
    candidate: dict[str, Any] | None,
    method: str,
    confidence: str,
    all_candidates: list[dict[str, Any]],
    truncated: bool,
    review_required: bool,
) -> dict[str, Any]:
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "answer_raw": candidate["raw"] if candidate else None,
        "answer_normalized": candidate["normalized"] if candidate else None,
        "method": method,
        "confidence": confidence,
        "evidence": _line_evidence(text, candidate["start"], candidate["end"])
        if candidate
        else "",
        "evidence_start": candidate["start"] if candidate else None,
        "evidence_end": candidate["end"] if candidate else None,
        "candidates": all_candidates,
        "candidate_count": len(all_candidates),
        "truncated": bool(truncated),
        "review_required": bool(review_required),
        "review_status": "pending" if review_required else "auto_accepted",
    }


def extract_answer_v2(
    text: str | None,
    *,
    truncated: bool = False,
    legacy_method: str | None = None,
    force_legacy_review: bool = False,
) -> dict[str, Any]:
    """Extract one forced candidate and retain enough evidence to audit it."""
    text = text or ""
    all_candidates = _number_candidates(text)
    historical_review = force_legacy_review and legacy_method in {
        "last_number_fallback",
        "not_found",
    }

    explicit_selection: tuple[str, dict[str, Any], bool] | None = None
    for method, pattern in EXPLICIT_PATTERNS:
        matches: list[tuple[int, dict[str, Any], bool]] = []
        for marker in pattern.finditer(text):
            body = marker.group("body")
            body_offset = marker.start("body")
            body_equations = _equality_candidates(body, body_offset)
            body_numbers = _number_candidates(body, body_offset, method)
            candidate = body_equations[-1] if body_equations else (body_numbers[0] if body_numbers else None)
            if candidate is not None:
                ambiguous = not body_equations and len(body_numbers) > 1
                matches.append((marker.start(), candidate, ambiguous))
        if matches:
            _, candidate, ambiguous = max(matches, key=lambda item: item[0])
            explicit_selection = (method, candidate, ambiguous)
            break
    if explicit_selection:
        method, candidate, ambiguous = explicit_selection
        return _selection(
            text,
            candidate,
            method,
            "medium" if ambiguous else "high",
            all_candidates,
            truncated,
            historical_review or ambiguous,
        )

    conclusions = list(CONCLUSION_RE.finditer(text))
    for conclusion in reversed(conclusions):
        body = conclusion.group(0)
        offset = conclusion.start()
        equations = _equality_candidates(body, offset)
        if equations:
            return _selection(
                text,
                equations[-1],
                "conclusion_equation",
                "medium",
                all_candidates,
                truncated,
                historical_review,
            )
        cue = re.search(r"(?i)\b(?:therefore|thus|hence|consequently|so)\b", body)
        cue_end = cue.end() if cue else 0
        numbers = _number_candidates(body[cue_end:], offset + cue_end, "conclusion_number")
        if numbers:
            return _selection(
                text,
                numbers[0],
                "conclusion_number",
                "medium",
                all_candidates,
                truncated,
                historical_review,
            )

    equations = _equality_candidates(text)
    if equations:
        return _selection(
            text,
            equations[-1],
            "last_equation_rhs",
            "low",
            all_candidates,
            truncated,
            True,
        )
    if all_candidates:
        return _selection(
            text,
            all_candidates[-1],
            "last_number_fallback",
            "low",
            all_candidates,
            truncated,
            True,
        )
    return _selection(
        text,
        None,
        "not_found",
        "none",
        all_candidates,
        truncated,
        True,
    )


def extract_generation(row: dict[str, Any], *, force_legacy_review: bool = False) -> dict[str, Any]:
    result = extract_answer_v2(
        row.get("raw_output"),
        truncated=bool(row.get("truncated")),
        legacy_method=row.get("answer_extraction_method"),
        force_legacy_review=force_legacy_review,
    )
    return {"run_id": row["run_id"], **result}


def validate_extractions(
    generations: list[dict[str, Any]],
    extractions: list[dict[str, Any]],
    *,
    require_reviewed: bool = False,
) -> dict[str, Any]:
    generation_ids = [row["run_id"] for row in generations]
    extraction_ids = [row["run_id"] for row in extractions]
    if len(set(generation_ids)) != len(generation_ids):
        raise ValueError("Duplicate run_id values in generations")
    if len(set(extraction_ids)) != len(extraction_ids):
        raise ValueError("Duplicate run_id values in extractions")
    missing = sorted(set(generation_ids) - set(extraction_ids))
    unexpected = sorted(set(extraction_ids) - set(generation_ids))
    if missing or unexpected:
        raise ValueError(f"Extraction run_id mismatch: missing={missing}, unexpected={unexpected}")
    pending = [row["run_id"] for row in extractions if row.get("review_status") == "pending"]
    if require_reviewed and pending:
        raise ValueError(f"Official metrics require completed reviews; pending={len(pending)}")
    statuses = Counter(row.get("review_status", "missing") for row in extractions)
    return {
        "records": len(extractions),
        "pending_reviews": len(pending),
        "review_status_counts": dict(sorted(statuses.items())),
        "result_status": "provisional" if pending else "final",
    }


def attach_extractions(
    generations: list[dict[str, Any]],
    extractions: list[dict[str, Any]],
    *,
    require_reviewed: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status = validate_extractions(
        generations, extractions, require_reviewed=require_reviewed
    )
    by_run = {row["run_id"]: row for row in extractions}
    attached = [
        {**row, "answer_extraction_v2": by_run[row["run_id"]]}
        for row in generations
    ]
    return attached, status


def effective_answer(row: dict[str, Any]) -> str | None:
    extraction = row.get("answer_extraction_v2")
    if extraction is not None:
        return extraction.get("answer_normalized")
    return normalize_numeric_token(row.get("extracted_answer"))
