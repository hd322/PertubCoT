#!/usr/bin/env python3
"""Gold-free feature construction for adaptive PerturbCoT verification."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MONITORS = ROOT / "perturbcot_monitors"
for import_path in (ROOT, MONITORS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from answer_extraction_v2 import effective_answer  # noqa: E402
from monitor_cot import (  # noqa: E402
    compare_relations,
    extract_version,
    irrelevant_number_use,
)


EXPECTED_PERTURBATIONS = (
    "entity_rename",
    "format_change",
    "irrelevant_information",
    "paraphrase",
)
SCORE_KEYS = ("answer_only", "answer_plus_cot", "full")


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


def token_cost(row: dict[str, Any]) -> int:
    return int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)


def group_generations(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["original_id"]].append(row)
    return dict(grouped)


def normalized_entropy(answers: list[str]) -> float:
    if len(answers) <= 1:
        return 0.0
    counts = Counter(answers)
    entropy = -sum(
        (count / len(answers)) * math.log(count / len(answers))
        for count in counts.values()
    )
    return entropy / math.log(len(answers))


def answer_risk(original_answer: str | None, answers: list[str | None]) -> dict[str, float]:
    valid = [answer for answer in answers if answer is not None]
    if original_answer is None:
        return {"original_support": 0.0, "normalized_entropy": 1.0, "risk": 1.0}
    if not valid:
        return {"original_support": 0.0, "normalized_entropy": 0.0, "risk": 0.5}
    support = sum(answer == original_answer for answer in valid) / len(valid)
    entropy = normalized_entropy(valid)
    return {
        "original_support": support,
        "normalized_entropy": entropy,
        "risk": ((1.0 - support) + entropy) / 2.0,
    }


def extraction_quality(row: dict[str, Any], config: dict[str, Any]) -> float:
    extraction = row.get("answer_extraction_v2") or {}
    if effective_answer(row) is None or row.get("truncated") or extraction.get("truncated"):
        return 1.0
    confidence = extraction.get("confidence", "none")
    return float(config["extraction_quality_risk"].get(confidence, 1.0))


def cot_features(
    original_row: dict[str, Any],
    variant_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    original = extract_version(original_row)
    variants = [extract_version(row) for row in variant_rows]
    conflicts: list[dict[str, Any]] = []
    comparison_count = 0
    for variant in variants:
        comparison_count += len(
            set(original["relation_map"]) & set(variant["relation_map"])
        )
        conflicts.extend(compare_relations(original, variant))

    irrelevant_uses: list[dict[str, Any]] = []
    for row, variant in zip(variant_rows, variants):
        if row.get("perturbation_type") == "irrelevant_information":
            irrelevant_uses.extend(irrelevant_number_use(original_row, row, variant))

    original_errors = original["arithmetic_errors"]
    variant_errors = [
        {"version": variant["version"], **error}
        for variant in variants
        for error in variant["arithmetic_errors"]
    ]
    if original_errors or conflicts or irrelevant_uses:
        severity = 1.0
    elif variant_errors:
        severity = 0.5
    else:
        severity = 0.0
    return {
        "risk": severity,
        "comparison_count": comparison_count,
        "original_arithmetic_error_count": len(original_errors),
        "relation_conflict_count": len(conflicts),
        "irrelevant_number_use_count": len(irrelevant_uses),
        "variant_arithmetic_error_count": len(variant_errors),
    }


def risk_scores(
    answer_value: float,
    cot_value: float,
    extraction_value: float,
    config: dict[str, Any],
) -> dict[str, float]:
    weights = config["risk_weights"]
    answer_cot_weight = weights["answer"] + weights["cot"]
    return {
        "answer_only": answer_value,
        "answer_plus_cot": (
            weights["answer"] * answer_value + weights["cot"] * cot_value
        ) / answer_cot_weight,
        "full": (
            weights["answer"] * answer_value
            + weights["cot"] * cot_value
            + weights["extraction"] * extraction_value
        ),
    }


def _select_group_rows(
    rows: list[dict[str, Any]], order: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    originals = [row for row in rows if row.get("run_kind") == "original_initial"]
    variants = {
        row.get("perturbation_type"): row
        for row in rows
        if row.get("run_kind") == "variant_initial"
    }
    if len(originals) != 1:
        raise ValueError("Each group must contain exactly one original_initial row")
    missing = sorted(set(order) - set(variants))
    if missing:
        raise ValueError(f"Missing perturbations for {originals[0]['original_id']}: {missing}")
    return originals[0], variants


def build_group_features(
    rows: list[dict[str, Any]],
    order: tuple[str, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    original, variants = _select_group_rows(rows, order)
    stages: list[dict[str, Any]] = []
    original_answer = effective_answer(original)
    observed_rows: list[dict[str, Any]] = []

    for stage_index in range(len(order) + 1):
        if stage_index:
            observed_rows.append(variants[order[stage_index - 1]])
        all_rows = [original, *observed_rows]
        answers = [effective_answer(row) for row in all_rows]
        answer_component = answer_risk(original_answer, answers)
        cot_component = cot_features(original, observed_rows)
        quality_values = [extraction_quality(row, config) for row in all_rows]
        extraction_component = sum(quality_values) / len(quality_values)
        scores = risk_scores(
            answer_component["risk"],
            cot_component["risk"],
            extraction_component,
            config,
        )
        safe = (
            stage_index >= config["minimum_safe_variants"]
            and all(answer is not None for answer in answers)
            and len(set(answers)) == 1
            and cot_component["risk"] == 0
            and all(value < 0.5 for value in quality_values)
        )
        force_verify = original_answer is None or (
            stage_index == len(order)
            and any(value >= 1.0 for value in quality_values)
        )
        stages.append(
            {
                "stage": stage_index,
                "observed_perturbations": list(order[:stage_index]),
                "answers": {
                    "original": original_answer,
                    **{
                        row["perturbation_type"]: effective_answer(row)
                        for row in observed_rows
                    },
                },
                "answer_component": answer_component,
                "cot_component": cot_component,
                "extraction_component": extraction_component,
                "risk_scores": scores,
                "safe_stop_eligible": safe,
                "force_verify": force_verify,
                "cumulative_monitor_tokens": sum(token_cost(row) for row in observed_rows),
            }
        )

    return {
        "original_id": original["original_id"],
        "split": original.get("split"),
        "order": list(order),
        "original_run_id": original["run_id"],
        "original_answer": original_answer,
        "stages": stages,
    }


def build_feature_rows(
    generations: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    all_orders: bool = False,
) -> list[dict[str, Any]]:
    grouped = group_generations(generations)
    primary = tuple(config["perturbation_order"])
    orders = list(permutations(primary)) if all_orders else [primary]
    return [
        build_group_features(grouped[original_id], order, config)
        for order in orders
        for original_id in sorted(grouped)
    ]


def validate_config(config: dict[str, Any]) -> None:
    order = config["perturbation_order"]
    if sorted(order) != sorted(EXPECTED_PERTURBATIONS):
        raise ValueError("perturbation_order must contain each supported type once")
    weights = config["risk_weights"]
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("risk_weights must sum to 1")
    thresholds = config["threshold_grid"]
    if thresholds != sorted(set(thresholds)) or not thresholds:
        raise ValueError("threshold_grid must be non-empty, sorted, and unique")
