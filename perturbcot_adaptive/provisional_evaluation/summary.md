# Adaptive verification pilot results

- Result status: `provisional`
- Frozen full-risk threshold: `0.3`
- Test groups: 40
- Initial accuracy: 23/40 (57.5%)
- Sequential final accuracy: 30/40 (75.0%)
- Sequential verification triggers: 24/40
- Sequential mean variants used: 1.73
- Sequential extra tokens per problem: 1453.0
- Errors corrected / correct answers harmed: 7 / 0
- Cost-matched random verification mean accuracy: 75.0%

The policy was selected on the 10-problem dev split and evaluated on the 40-problem test split. All model outputs were generated before this replay. Costs are counterfactual deployment costs: only perturbations observed before stopping and the four verification samples used after a trigger are charged. Current results remain provisional until all answer-extraction reviews are complete.
