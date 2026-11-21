# P16 historical replay gate evidence

P16's outcome is that selecting an earlier cycle produces a prediction using only readings available through that cycle. All nine child issues are closed, and their task changes are on `dev`. This note records evidence for shared gate #18 before phase promotion.

## Shared acceptance

| Gate condition | Verified evidence |
| --- | --- |
| Future-row mutation does not change earlier predictions | P16-S1 #76 and P16-S2 #153 report 20/20 development-validation predictions unchanged under future-row mutation, with maximum raw difference 0.0 cycles. P16-A2 #152 also verifies separate cutoff copies: a predictor mutation at one cutoff does not alter the next cutoff or caller-owned rows. |
| Slider, sensor chart and prediction use the same cutoff | P16-T1 #77 AppTest selected engine 2/cycle 2 and confirmed inference received only cycles 1-2. On the packaged FD001/train subset, engine 4/cycle 100 displayed 135.72 cycles (raw 135.722467); the sensor chart ended at cycle 100. |
| Earlier predictions are visible and bounded | P16-T2 #154 plots the saved replay outputs at no more than 10 observed cutoffs, through the selected cycle. P16-A1 #75 replays engine 4 at cycles 50/100/164 (154.478528/135.722467/34.196078 cycles). P16-A3 #212 uses a 128-entry LRU keyed by engine/dataset, cutoff, history prefix, model version, feature config, and manifest/model bytes; tests cover reuse, invalidation, and eviction. |
| Model and fit provenance are explicit | P16-T3 #257 shows `decision-tree-rul-v1` and fit membership next to the result. Packaged engines 4, 5, and 12 are listed in the saved development-validation split, not fitting. AppTest also verifies fitting, validation, and unknown statuses. P16-S3 #213 verifies the page describes the saved P06 development model and labels training examples conservatively. |

## Observed timeline example

The saved `decision-tree-rul-v1` model was applied to the packaged FD001/train history for engine 4 through cycle 100. The selected prediction was 135.722467 raw cycles (135.72 displayed), using a 30-row feature window. The ten observed cutoff predictions were:

| Observed cycle | Estimated remaining cycles |
| ---: | ---: |
| 1 | 117.445205 |
| 12 | 117.445205 |
| 23 | 195.400000 |
| 34 | 117.112500 |
| 45 | 202.729682 |
| 56 | 202.729682 |
| 67 | 202.729682 |
| 78 | 169.924324 |
| 89 | 117.112500 |
| 100 | 135.722467 |

The estimates vary and are not monotonic. They are software outputs on a small development training-history subset; they do not establish official-test accuracy, maintenance readiness, or a guaranteed failure time.

## Scope and verification

- Full test suite on the P16-T3 revision: 468 passed.
- Ruff, `git diff --check`, and Streamlit health smoke: passed; health endpoint returned HTTP 200 `ok`.
- The available subset contains validation-engine histories only. No fitting-engine prediction, final refitted artifact, or official-test output is claimed.
- Task evidence and merge records remain linked from issues #75, #76, #77, #152, #153, #154, #212, #213, and #257. PR #412's implementation tree is on `dev` at `d204eb329895e19e3abc0d42a93e02c6b3f7727c`; the GitHub API did not mark PR #412 formally merged, so it was closed as already landed and that discrepancy is recorded on the PR and issue.
