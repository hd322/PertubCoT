# Adaptive verification experiment

This module replays the saved 450 generations to evaluate an interpretable
risk score and sequential perturbation early stopping. It does not run a
model. The four perturbation outputs provide monitoring signals; original
repeats `r1` through `r4` are reserved for the five-sample verification
bundle.

## Protocol

The primary perturbation order is `irrelevant_information`, `entity_rename`,
`paraphrase`, then `format_change`. At every stage the risk score combines
answer instability, deterministic CoT equation anomalies, and extraction
uncertainty. A high-risk item immediately uses the verification bundle. An
item with two or more completely agreeing perturbations can stop early.

Thresholds are selected on the 10-item dev split within 105% of the
always-verify token budget. The 5% tolerance handles discrete, variable-length
generations; every result reports the exact realized token cost. The 40-item
test split is evaluation-only. Because these saved test results have already
been inspected during the project, the experiment is exploratory rather than
a new blinded confirmation.

## Reproduce provisional results

From the repository root:

```bash
python3 perturbcot_adaptive/build_risk_traces.py \
  --generations perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl \
  --extractions perturbcot_answer_extraction_v2/extractions_v2.jsonl \
  --output-dir perturbcot_adaptive/provisional_features \
  --allow-provisional

python3 perturbcot_adaptive/evaluate_adaptive.py \
  --features perturbcot_adaptive/provisional_features/risk_features.jsonl \
  --order-features perturbcot_adaptive/provisional_features/order_sensitivity_features.jsonl \
  --generations perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl \
  --extractions perturbcot_answer_extraction_v2/extractions_v2.jsonl \
  --gold perturbcot_data_prep/data/offline_eval/gold.jsonl \
  --output-dir perturbcot_adaptive/provisional_evaluation \
  --allow-provisional
```

Omit `--allow-provisional`, or pass `--require-reviewed`, for an official run.
Both commands then reject pending extraction reviews. Gold is accepted only by
the offline evaluator; feature files contain no correctness labels.

## Current status

The checked-in outputs are provisional because 162 extraction records still
require blind review. On the 40-item test split, both always-verify and the
primary sequential full-risk policy reach 30/40 from an initial 23/40. The
full-risk policy uses 58,119 extra tokens versus 57,014 for always-verify. The
answer-only ablation also reaches 30/40 and uses 54,645 tokens, so the current
pilot supports early stopping but does not show incremental value from the CoT
or extraction-risk components.
