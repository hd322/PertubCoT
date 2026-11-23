#!/usr/bin/env python3
"""Select adaptive policies on dev and evaluate them offline on test."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_extraction_v2 import (  # noqa: E402
    attach_extractions,
    effective_answer,
    normalize_numeric_token,
)
try:  # noqa: E402
    from .adaptive_core import (
        SCORE_KEYS,
        group_generations,
        read_jsonl,
        token_cost,
        validate_config,
        write_jsonl,
    )
except ImportError:  # Direct script execution.
    from adaptive_core import (
        SCORE_KEYS,
        group_generations,
        read_jsonl,
        token_cost,
        validate_config,
        write_jsonl,
    )


def verification_bundle(
    rows: list[dict[str, Any]], minimum_votes: int
) -> dict[str, Any]:
    originals = sorted(
        (row for row in rows if row.get("version_type") == "original"),
        key=lambda row: row["repeat_index"],
    )
    if len(originals) != 5:
        raise ValueError("Verification requires five original generations per group")
    initial = effective_answer(originals[0])
    votes = [effective_answer(row) for row in originals]
    valid = [answer for answer in votes if answer is not None]
    selected = None
    reason = "insufficient_votes"
    if len(valid) >= minimum_votes:
        counts = Counter(valid)
        top_count = max(counts.values())
        winners = sorted(answer for answer, count in counts.items() if count == top_count)
        if len(winners) == 1:
            selected = winners[0]
            reason = "unique_mode"
        else:
            reason = "mode_tie"
    return {
        "votes": votes,
        "selected_answer": selected,
        "final_answer": selected if selected is not None else initial,
        "update_reason": reason if selected is not None else reason + "_keep_initial",
        "token_cost": sum(token_cost(row) for row in originals[1:]),
    }


def _base_outcome(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_id": feature["original_id"],
        "split": feature["split"],
        "initial_answer": feature["original_answer"],
        "final_answer": feature["original_answer"],
        "triggered_verification": False,
        "verification": None,
        "stop_stage": 0,
        "stop_reason": "keep_initial",
        "monitor_tokens": 0,
        "verification_tokens": 0,
        "total_extra_tokens": 0,
    }


def _apply_verification(
    outcome: dict[str, Any], bundle: dict[str, Any]
) -> dict[str, Any]:
    outcome["triggered_verification"] = True
    outcome["verification"] = bundle
    outcome["final_answer"] = bundle["final_answer"]
    outcome["verification_tokens"] = bundle["token_cost"]
    outcome["total_extra_tokens"] = outcome["monitor_tokens"] + bundle["token_cost"]
    return outcome


def simulate_sequential(
    feature: dict[str, Any],
    bundle: dict[str, Any],
    threshold: float,
    score_key: str,
) -> dict[str, Any]:
    outcome = _base_outcome(feature)
    for stage in feature["stages"]:
        outcome["stop_stage"] = stage["stage"]
        outcome["monitor_tokens"] = stage["cumulative_monitor_tokens"]
        outcome["total_extra_tokens"] = stage["cumulative_monitor_tokens"]
        score = stage["risk_scores"][score_key]
        outcome["risk_score_at_stop"] = score
        if stage["force_verify"]:
            outcome["stop_reason"] = "forced_verification"
            return _apply_verification(outcome, bundle)
        if score >= threshold:
            outcome["stop_reason"] = "risk_threshold"
            return _apply_verification(outcome, bundle)
        if stage["safe_stop_eligible"]:
            outcome["stop_reason"] = "safe_early_stop"
            return outcome
    outcome["stop_reason"] = "max_perturbations_below_threshold"
    return outcome


def simulate_full_risk(
    feature: dict[str, Any],
    bundle: dict[str, Any],
    threshold: float,
    score_key: str = "full",
) -> dict[str, Any]:
    stage = feature["stages"][-1]
    outcome = _base_outcome(feature)
    outcome.update(
        {
            "stop_stage": stage["stage"],
            "monitor_tokens": stage["cumulative_monitor_tokens"],
            "total_extra_tokens": stage["cumulative_monitor_tokens"],
            "risk_score_at_stop": stage["risk_scores"][score_key],
            "stop_reason": "full_risk_below_threshold",
        }
    )
    if stage["force_verify"] or stage["risk_scores"][score_key] >= threshold:
        outcome["stop_reason"] = "full_risk_verification"
        return _apply_verification(outcome, bundle)
    return outcome


def simulate_any_disagreement(
    feature: dict[str, Any], bundle: dict[str, Any]
) -> dict[str, Any]:
    stage = feature["stages"][-1]
    answers = list(stage["answers"].values())
    valid = [answer for answer in answers if answer is not None]
    trigger = len(valid) != len(answers) or len(set(valid)) > 1
    outcome = _base_outcome(feature)
    outcome.update(
        {
            "stop_stage": stage["stage"],
            "monitor_tokens": stage["cumulative_monitor_tokens"],
            "total_extra_tokens": stage["cumulative_monitor_tokens"],
            "stop_reason": "any_disagreement" if trigger else "all_answers_agree",
        }
    )
    return _apply_verification(outcome, bundle) if trigger else outcome


def simulate_always(feature: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    outcome = _base_outcome(feature)
    outcome["stop_reason"] = "always_verify"
    return _apply_verification(outcome, bundle)


def attach_correctness(
    outcome: dict[str, Any], gold_answer: str | None
) -> dict[str, Any]:
    result = dict(outcome)
    result["initial_correct"] = result["initial_answer"] == gold_answer
    result["final_correct"] = result["final_answer"] == gold_answer
    return result


def policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    initial_correct = sum(row["initial_correct"] for row in rows)
    final_correct = sum(row["final_correct"] for row in rows)
    corrected = sum(
        not row["initial_correct"] and row["final_correct"] for row in rows
    )
    harmed = sum(row["initial_correct"] and not row["final_correct"] for row in rows)
    triggered = [row for row in rows if row["triggered_verification"]]
    actual_errors = sum(not row["initial_correct"] for row in rows)
    trigger_errors = sum(not row["initial_correct"] for row in triggered)
    false_triggers = sum(row["initial_correct"] for row in triggered)
    monitor_tokens = sum(row["monitor_tokens"] for row in rows)
    verification_tokens = sum(row["verification_tokens"] for row in rows)
    total_tokens = monitor_tokens + verification_tokens
    net = corrected - harmed
    return {
        "groups": total,
        "initial_correct": initial_correct,
        "initial_accuracy": initial_correct / total if total else None,
        "final_correct": final_correct,
        "final_accuracy": final_correct / total if total else None,
        "errors_corrected": corrected,
        "correct_answers_harmed": harmed,
        "net_corrections": net,
        "verification_triggers": len(triggered),
        "verification_trigger_rate": len(triggered) / total if total else None,
        "error_recall": trigger_errors / actual_errors if actual_errors else None,
        "trigger_precision": trigger_errors / len(triggered) if triggered else None,
        "correct_answer_trigger_rate": (
            false_triggers / initial_correct if initial_correct else None
        ),
        "mean_variants_used": (
            sum(row["stop_stage"] for row in rows) / total if total else None
        ),
        "stop_stage_counts": dict(
            sorted(Counter(str(row["stop_stage"]) for row in rows).items())
        ),
        "monitor_tokens": monitor_tokens,
        "verification_tokens": verification_tokens,
        "total_extra_tokens": total_tokens,
        "extra_tokens_per_group": total_tokens / total if total else None,
        "tokens_per_net_correction": total_tokens / net if net > 0 else None,
    }


def evaluate_policy(
    features: list[dict[str, Any]],
    generation_groups: dict[str, list[dict[str, Any]]],
    gold: dict[str, str | None],
    simulator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    minimum_votes: int,
) -> list[dict[str, Any]]:
    outcomes = []
    for feature in features:
        bundle = verification_bundle(
            generation_groups[feature["original_id"]], minimum_votes
        )
        outcome = simulator(feature, bundle)
        outcomes.append(attach_correctness(outcome, gold[feature["original_id"]]))
    return outcomes


def select_threshold(
    dev_features: list[dict[str, Any]],
    generation_groups: dict[str, list[dict[str, Any]]],
    gold: dict[str, str | None],
    config: dict[str, Any],
    score_key: str,
    *,
    allow_infeasible: bool = False,
) -> tuple[float, list[dict[str, Any]], bool]:
    minimum_votes = config["minimum_verification_votes"]
    always = evaluate_policy(
        dev_features,
        generation_groups,
        gold,
        simulate_always,
        minimum_votes,
    )
    reference_budget = policy_metrics(always)["total_extra_tokens"] * config[
        "primary_budget_fraction"
    ]
    budget = reference_budget * (1 + config["budget_tolerance_fraction"])
    candidates = []
    for threshold in config["threshold_grid"]:
        outcomes = evaluate_policy(
            dev_features,
            generation_groups,
            gold,
            lambda feature, bundle, value=threshold: simulate_sequential(
                feature, bundle, value, score_key
            ),
            minimum_votes,
        )
        metrics = policy_metrics(outcomes)
        candidates.append(
            {
                "threshold": threshold,
                "within_budget": metrics["total_extra_tokens"] <= budget,
                "budget_tokens": budget,
                "reference_budget_tokens": reference_budget,
                "metrics": metrics,
            }
        )
    feasible = [row for row in candidates if row["within_budget"]]
    if not feasible:
        if not allow_infeasible:
            raise ValueError(f"No {score_key} threshold satisfies the dev token budget")
        selected = min(
            candidates,
            key=lambda row: (
                row["metrics"]["total_extra_tokens"],
                -row["metrics"]["final_accuracy"],
                row["metrics"]["correct_answer_trigger_rate"] or 0,
                -row["threshold"],
            ),
        )
        return selected["threshold"], candidates, False
    selected = min(
        feasible,
        key=lambda row: (
            -row["metrics"]["final_accuracy"],
            row["metrics"]["total_extra_tokens"],
            row["metrics"]["correct_answer_trigger_rate"] or 0,
            -row["threshold"],
        ),
    )
    return selected["threshold"], candidates, True


def rank_metrics(scores: list[float], labels: list[bool]) -> dict[str, float | None]:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        auc = None
    else:
        wins = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        auc = wins / (len(positives) * len(negatives))
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / len(positives) if positives else None
    return {"auroc": auc, "average_precision": average_precision}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left_by_id = {row["original_id"]: row for row in left}
    right_by_id = {row["original_id"]: row for row in right}
    ids = sorted(left_by_id)
    if ids != sorted(right_by_id):
        raise ValueError("Bootstrap policies cover different test groups")
    rng = random.Random(seed)
    accuracy_differences = []
    token_differences = []
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        accuracy_differences.append(
            sum(
                int(left_by_id[item]["final_correct"])
                - int(right_by_id[item]["final_correct"])
                for item in sample
            )
            / len(sample)
        )
        token_differences.append(
            sum(
                left_by_id[item]["total_extra_tokens"]
                - right_by_id[item]["total_extra_tokens"]
                for item in sample
            )
            / len(sample)
        )
    return {
        "accuracy_difference": {
            "estimate": policy_metrics(left)["final_accuracy"]
            - policy_metrics(right)["final_accuracy"],
            "ci_95": [
                percentile(accuracy_differences, 0.025),
                percentile(accuracy_differences, 0.975),
            ],
        },
        "extra_tokens_per_group_difference": {
            "estimate": policy_metrics(left)["extra_tokens_per_group"]
            - policy_metrics(right)["extra_tokens_per_group"],
            "ci_95": [
                percentile(token_differences, 0.025),
                percentile(token_differences, 0.975),
            ],
        },
    }


def random_budget_baseline(
    features: list[dict[str, Any]],
    generation_groups: dict[str, list[dict[str, Any]]],
    gold: dict[str, str | None],
    budget: int,
    config: dict[str, Any],
    target_accuracy: float,
) -> dict[str, Any]:
    feature_by_id = {row["original_id"]: row for row in features}
    ids = sorted(feature_by_id)
    bundles = {
        item: verification_bundle(
            generation_groups[item], config["minimum_verification_votes"]
        )
        for item in ids
    }
    repetitions = []
    for repeat in range(config["random_repetitions"]):
        rng = random.Random(config["random_seed"] + repeat)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        selected: set[str] = set()
        spent = 0
        for item in shuffled:
            cost = bundles[item]["token_cost"]
            if spent + cost <= budget:
                selected.add(item)
                spent += cost
        outcomes = []
        for item in ids:
            feature = feature_by_id[item]
            outcome = _base_outcome(feature)
            outcome["stop_reason"] = "random_budget"
            if item in selected:
                outcome = _apply_verification(outcome, bundles[item])
            outcomes.append(attach_correctness(outcome, gold[item]))
        repetitions.append(policy_metrics(outcomes))
    accuracies = [row["final_accuracy"] for row in repetitions]
    costs = [row["total_extra_tokens"] for row in repetitions]
    triggers = [row["verification_triggers"] for row in repetitions]
    return {
        "repetitions": len(repetitions),
        "budget_tokens": budget,
        "final_accuracy": {
            "mean": sum(accuracies) / len(accuracies),
            "interval_95": [percentile(accuracies, 0.025), percentile(accuracies, 0.975)],
        },
        "total_extra_tokens": {
            "mean": sum(costs) / len(costs),
            "interval_95": [percentile(costs, 0.025), percentile(costs, 0.975)],
        },
        "verification_triggers": {
            "mean": sum(triggers) / len(triggers),
            "interval_95": [percentile(triggers, 0.025), percentile(triggers, 0.975)],
        },
        "target_accuracy_percentile": (
            (
                sum(value < target_accuracy for value in accuracies)
                + 0.5 * sum(value == target_accuracy for value in accuracies)
            )
            / len(accuracies)
        ),
    }


def write_curve_csv(path: Path, curves: list[dict[str, Any]]) -> None:
    fields = [
        "score_key",
        "threshold",
        "selected",
        "split",
        "final_accuracy",
        "extra_tokens_per_group",
        "verification_trigger_rate",
        "mean_variants_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curves)


def render_curve_svg(curves: list[dict[str, Any]]) -> str:
    test_rows = [row for row in curves if row["split"] == "test" and row["score_key"] == "full"]
    width, height = 760, 460
    left, right, top, bottom = 75, 30, 35, 65
    x_values = [row["extra_tokens_per_group"] for row in test_rows]
    y_values = [row["final_accuracy"] for row in test_rows]
    x_max = max(x_values) * 1.05 if x_values else 1
    y_min = max(0.0, min(y_values) - 0.05) if y_values else 0
    y_max = min(1.0, max(y_values) + 0.05) if y_values else 1
    if y_max == y_min:
        y_max = min(1.0, y_min + 0.1)

    def point(row: dict[str, Any]) -> tuple[float, float]:
        x = left + row["extra_tokens_per_group"] / x_max * (width - left - right)
        y = top + (y_max - row["final_accuracy"]) / (y_max - y_min) * (height - top - bottom)
        return x, y

    ordered = sorted(test_rows, key=lambda row: row["extra_tokens_per_group"])
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(point, ordered))
    circles = []
    for row in ordered:
        x, y = point(row)
        color = "#b42318" if row["selected"] else "#175cd3"
        radius = 6 if row["selected"] else 4
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"><title>'
            f'threshold={row["threshold"]}, accuracy={row["final_accuracy"]:.3f}, '
            f'tokens/problem={row["extra_tokens_per_group"]:.1f}</title></circle>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">Sequential risk: accuracy vs extra tokens</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#475467"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#475467"/>
<text x="{width / 2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">Extra tokens per problem</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="sans-serif" font-size="13">Final accuracy</text>
<polyline points="{points}" fill="none" stroke="#175cd3" stroke-width="2"/>
{''.join(circles)}
<text x="{left}" y="{height-bottom+20}" font-family="sans-serif" font-size="11">0</text>
<text x="{width-right}" y="{height-bottom+20}" text-anchor="end" font-family="sans-serif" font-size="11">{x_max:.0f}</text>
<text x="{left-8}" y="{height-bottom}" text-anchor="end" font-family="sans-serif" font-size="11">{y_min:.2f}</text>
<text x="{left-8}" y="{top+4}" text-anchor="end" font-family="sans-serif" font-size="11">{y_max:.2f}</text>
</svg>
'''


def render_summary(metrics: dict[str, Any]) -> str:
    policies = metrics["test_policies"]
    selected = metrics["frozen_policies"]["full"]["threshold"]
    sequential = policies["sequential_full"]
    random_result = metrics["random_budget_baseline"]
    status = metrics["result_status"]
    return f"""# Adaptive verification pilot results

- Result status: `{status}`
- Frozen full-risk threshold: `{selected}`
- Test groups: {sequential['groups']}
- Initial accuracy: {sequential['initial_correct']}/{sequential['groups']} ({sequential['initial_accuracy']:.1%})
- Sequential final accuracy: {sequential['final_correct']}/{sequential['groups']} ({sequential['final_accuracy']:.1%})
- Sequential verification triggers: {sequential['verification_triggers']}/{sequential['groups']}
- Sequential mean variants used: {sequential['mean_variants_used']:.2f}
- Sequential extra tokens per problem: {sequential['extra_tokens_per_group']:.1f}
- Errors corrected / correct answers harmed: {sequential['errors_corrected']} / {sequential['correct_answers_harmed']}
- Cost-matched random verification mean accuracy: {random_result['final_accuracy']['mean']:.1%}

The policy was selected on the 10-problem dev split and evaluated on the 40-problem test split. All model outputs were generated before this replay. Costs are counterfactual deployment costs: only perturbations observed before stopping and the four verification samples used after a trigger are charged. Current results remain provisional until all answer-extraction reviews are complete.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--order-features", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=BASE / "config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--require-reviewed", action="store_true")
    gate.add_argument("--allow-provisional", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    require_reviewed = args.require_reviewed or not args.allow_provisional
    generations, extraction_status = attach_extractions(
        read_jsonl(args.generations),
        read_jsonl(args.extractions),
        require_reviewed=require_reviewed,
    )
    features = read_jsonl(args.features)
    order_features = read_jsonl(args.order_features)
    generation_groups = group_generations(generations)
    gold_rows = read_jsonl(args.gold)
    gold = {
        row["original_id"]: normalize_numeric_token(row["gold_answer"])
        for row in gold_rows
    }
    if len(features) != 50 or set(gold) != {row["original_id"] for row in features}:
        raise ValueError("Expected matching features and gold for 50 groups")
    dev = [row for row in features if row["split"] == "dev"]
    test = [row for row in features if row["split"] == "test"]
    if len(dev) != 10 or len(test) != 40:
        raise ValueError(f"Expected dev=10 and test=40, found {len(dev)} and {len(test)}")

    selected_thresholds: dict[str, float] = {}
    selection_rows: dict[str, list[dict[str, Any]]] = {}
    curves: list[dict[str, Any]] = []
    for score_key in SCORE_KEYS:
        threshold, candidates, budget_feasible = select_threshold(
            dev, generation_groups, gold, config, score_key
        )
        selected_thresholds[score_key] = threshold
        selection_rows[score_key] = candidates
        if not budget_feasible:
            raise AssertionError("Primary perturbation order must satisfy the dev budget")
        for candidate in candidates:
            for split_name, split_features in (("dev", dev), ("test", test)):
                outcomes = evaluate_policy(
                    split_features,
                    generation_groups,
                    gold,
                    lambda feature, bundle, value=candidate["threshold"], key=score_key: simulate_sequential(
                        feature, bundle, value, key
                    ),
                    config["minimum_verification_votes"],
                )
                summary = policy_metrics(outcomes)
                curves.append(
                    {
                        "score_key": score_key,
                        "threshold": candidate["threshold"],
                        "selected": candidate["threshold"] == threshold,
                        "split": split_name,
                        "final_accuracy": summary["final_accuracy"],
                        "extra_tokens_per_group": summary["extra_tokens_per_group"],
                        "verification_trigger_rate": summary["verification_trigger_rate"],
                        "mean_variants_used": summary["mean_variants_used"],
                    }
                )

    full_threshold = selected_thresholds["full"]
    simulators = {
        "no_verification": lambda feature, bundle: _base_outcome(feature),
        "always_verify": simulate_always,
        "all4_any_disagreement": simulate_any_disagreement,
        "all4_full_risk": lambda feature, bundle: simulate_full_risk(
            feature, bundle, full_threshold
        ),
        "sequential_answer_only": lambda feature, bundle: simulate_sequential(
            feature, bundle, selected_thresholds["answer_only"], "answer_only"
        ),
        "sequential_answer_plus_cot": lambda feature, bundle: simulate_sequential(
            feature, bundle, selected_thresholds["answer_plus_cot"], "answer_plus_cot"
        ),
        "sequential_full": lambda feature, bundle: simulate_sequential(
            feature, bundle, full_threshold, "full"
        ),
    }
    outcomes_by_policy: dict[str, list[dict[str, Any]]] = {}
    test_metrics: dict[str, Any] = {}
    for name, simulator in simulators.items():
        outcomes = evaluate_policy(
            test,
            generation_groups,
            gold,
            simulator,
            config["minimum_verification_votes"],
        )
        for outcome in outcomes:
            outcome["policy"] = name
        outcomes_by_policy[name] = outcomes
        test_metrics[name] = policy_metrics(outcomes)

    main_outcomes = outcomes_by_policy["sequential_full"]
    random_result = random_budget_baseline(
        test,
        generation_groups,
        gold,
        test_metrics["sequential_full"]["total_extra_tokens"],
        config,
        test_metrics["sequential_full"]["final_accuracy"],
    )
    random_result["sequential_minus_random_mean_accuracy"] = (
        test_metrics["sequential_full"]["final_accuracy"]
        - random_result["final_accuracy"]["mean"]
    )

    final_stage_scores = [row["stages"][-1]["risk_scores"]["full"] for row in test]
    initial_error_labels = [
        row["original_answer"] != gold[row["original_id"]] for row in test
    ]
    ranking = rank_metrics(final_stage_scores, initial_error_labels)

    bootstrap = {
        other: bootstrap_difference(
            main_outcomes,
            outcomes_by_policy[other],
            config["bootstrap_iterations"],
            config["random_seed"],
        )
        for other in ("no_verification", "always_verify", "all4_full_risk")
    }

    sensitivity_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in order_features:
        sensitivity_groups[tuple(row["order"])].append(row)
    order_results = []
    for order, rows in sorted(sensitivity_groups.items()):
        order_dev = [row for row in rows if row["split"] == "dev"]
        order_test = [row for row in rows if row["split"] == "test"]
        threshold, _, budget_feasible = select_threshold(
            order_dev,
            generation_groups,
            gold,
            config,
            "full",
            allow_infeasible=True,
        )
        outcomes = evaluate_policy(
            order_test,
            generation_groups,
            gold,
            lambda feature, bundle, value=threshold: simulate_sequential(
                feature, bundle, value, "full"
            ),
            config["minimum_verification_votes"],
        )
        order_results.append(
            {
                "order": list(order),
                "selected_dev_threshold": threshold,
                "dev_budget_feasible": budget_feasible,
                "is_primary_order": list(order) == config["perturbation_order"],
                "test_metrics": policy_metrics(outcomes),
            }
        )
    ranked_orders = sorted(
        order_results,
        key=lambda row: (
            -row["test_metrics"]["final_accuracy"],
            row["test_metrics"]["total_extra_tokens"],
            row["order"],
        ),
    )
    for rank, row in enumerate(ranked_orders, start=1):
        row["descriptive_test_rank"] = rank

    frozen = {
        key: {
            "threshold": selected_thresholds[key],
            "selection_split": "dev",
            "budget_fraction_of_always_verify": config["primary_budget_fraction"],
            "budget_tolerance_fraction": config["budget_tolerance_fraction"],
        }
        for key in SCORE_KEYS
    }
    metrics = {
        "experiment_version": config["experiment_version"],
        "result_status": extraction_status["result_status"],
        "extraction_status": extraction_status,
        "evidence_scope": "exploratory_pilot_existing_50",
        "split_counts": {"dev": len(dev), "test": len(test)},
        "frozen_policies": frozen,
        "test_policies": test_metrics,
        "risk_ranking_test": ranking,
        "random_budget_baseline": random_result,
        "paired_bootstrap": bootstrap,
        "order_sensitivity": ranked_orders,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "evaluated_policy_traces.jsonl",
        (
            row
            for name in simulators
            for row in outcomes_by_policy[name]
        ),
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "frozen_policy.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "threshold_selection.json").write_text(
        json.dumps(selection_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "order_sensitivity.json").write_text(
        json.dumps(ranked_orders, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_curve_csv(args.output_dir / "accuracy_cost_curve.csv", curves)
    (args.output_dir / "accuracy_cost_curve.svg").write_text(
        render_curve_svg(curves), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_summary(metrics), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
