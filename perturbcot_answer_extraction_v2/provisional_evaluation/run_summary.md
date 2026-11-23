# Qwen2.5-Math-1.5B pilot run summary

- Records: 450 (50 original problems)
- Result status: provisional
- Successful answer extraction: 449/450
- Truncated generations: 20
- Original initial accuracy: 27/50 (54.0%)
- Variant accuracy: 119/200 (59.5%)
- Five-sample majority accuracy: 35/50 (70.0%)
- Original repeat groups with one stable answer: 17/50 (34.0%)
- Variant answers matching the original initial answer: 92/200 (46.0%)
- Answer-only monitor alerts: 37/50 (74.0%)
- Answer-only monitor unknown groups: 0/50
- Total tokens: 157,747 (46,022 input + 111,725 output)
- Summed generation latency: 4805.6 seconds

The answer-only alert treats a group as unstable when the original initial answer and its four perturbation answers are not all present and numerically identical. Standard answers are joined only in this offline evaluation step.
