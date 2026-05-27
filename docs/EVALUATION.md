# Prototype Evaluation Summary

## Anti-Spoofing Selection

Multiple passive alternatives were tested during development. Generic RGB models were rejected when high-quality photos were accepted or live samples were blocked inconsistently. rPPG was rejected because it required approximately 12 seconds and 120 to 200 frames for practical stability while still failing on at least one photo presentation. Active challenges were not selected because they make normal office entry awkward.

The final implementation uses a locally trained CDCN-style model and passive single-transaction access policy.

## Final Observed Validation

Threshold: `0.42`

| Trial type | Scores | Classification result |
| --- | --- | --- |
| Real face | `0.732`, `0.754`, `0.513`, `0.542`, `0.526` | 5/5 classified as live |
| Photo/screen spoof | `0.045`, `0.040`, `0.099`, `0.025` | 4/4 classified as spoof |

| Metric | Observed value |
| --- | --- |
| FAR / APCER | `0 / 4 = 0.00%` |
| FRR / BPCER | `0 / 5 = 0.00%` |
| Accuracy | `9 / 9 = 100.00%` |
| Observed EER on tested sample | `0.00%` |
| Separation gap | `0.513 - 0.099 = 0.414` |

These results must be read as internal prototype validation. A production claim would require broader subjects, multiple environmental conditions, replay-video attacks, mask/print variations, and standardized external testing.

