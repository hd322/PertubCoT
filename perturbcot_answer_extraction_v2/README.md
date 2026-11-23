# PerturbCoT answer extraction v2

This directory contains the deterministic re-extraction of the 450 saved model
generations. It is versioned separately so the original generations and legacy
evaluation outputs remain unchanged.

## Current status

- 450 generation records were re-extracted without reading gold answers.
- 288 records are automatically accepted.
- 162 records require blind review: 149 legacy last-number fallbacks, one
  no-number response, and 12 explicit answer statements containing multiple
  competing numbers.
- Metrics produced before those reviews are complete are provisional.

`blind_review_v2.csv` intentionally contains no gold answer or correctness
field. A reviewer should inspect only the question and raw response, then set:

- `decision=accept` to accept the proposed answer;
- `decision=correct` with `reviewed_answer` and `review_evidence`;
- `decision=unresolved` when the response does not provide a reliable answer.

Every reviewed row must also name a `reviewer`.

## Reproduce the extraction

From the repository root:

```bash
python3 extract_answers_v2.py \
  --generations perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl \
  --output perturbcot_answer_extraction_v2/extractions_v2.jsonl \
  --review-csv perturbcot_answer_extraction_v2/blind_review_v2.csv \
  --summary perturbcot_answer_extraction_v2/extraction_summary_v2.json
```

After blind review:

```bash
python3 apply_extraction_reviews_v2.py \
  --extractions perturbcot_answer_extraction_v2/extractions_v2.jsonl \
  --reviews perturbcot_answer_extraction_v2/blind_review_v2.csv \
  --output perturbcot_answer_extraction_v2/resolved_extractions_v2.jsonl \
  --summary perturbcot_answer_extraction_v2/review_summary_v2.json
```

Official evaluation must use the resolved file and the review gate:

```bash
python3 perturbcot_model_run/evaluate_outputs.py \
  --generations perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl \
  --gold perturbcot_data_prep/data/offline_eval/gold.jsonl \
  --extractions perturbcot_answer_extraction_v2/resolved_extractions_v2.jsonl \
  --require-reviewed \
  --output-dir perturbcot_answer_extraction_v2/offline_evaluation
```

The same `--extractions` and `--require-reviewed` options are supported by the
answer monitor, CoT monitor, and monitor evaluator.

