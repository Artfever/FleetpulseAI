# P08-S2 Random Forest high-error examples

**Issue:** [P08-S2 #129](https://github.com/Artfever/FleetpulseAI/issues/129)

**Source:** local, Git-ignored `artifacts/development/random_forest_validation.json`

**Partition:** FD001/train development validation; one saved cutoff per engine

**Model:** `random-forest-rul@random-forest-rul-v1`

**Observed:** 2026-11-16

This inspection ranks the 20 saved validation rows by their recorded absolute
error. Signed error is prediction minus target, so a positive value means the
model overpredicted RUL and a negative value means it underpredicted it. The
figures below are copied from the saved prediction rows; the P08-S1 audit
independently checked each row's identity, target, absolute error, and squared
error against the validation table.

| Rank | Engine | Cutoff cycle | Target RUL | Predicted RUL | Signed error | Absolute error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 | 27 | 136 | 191.139 | +55.139 | 55.139 |
| 2 | 5 | 58 | 211 | 159.877 | -51.123 | 51.123 |
| 3 | 28 | 36 | 129 | 172.425 | +43.425 | 43.425 |
| 4 | 15 | 63 | 144 | 185.025 | +41.025 | 41.025 |
| 5 | 82 | 56 | 158 | 184.721 | +26.721 | 26.721 |

Four of 20 rows have absolute error greater than 40 cycles. Across all 20,
13 predictions are above target and 7 are below. The largest overprediction is
engine 29 at cycle 27 (+55.139 cycles); the largest underprediction is engine
5 at cycle 58 (-51.123 cycles). Thus the prominent errors occur in both
directions; this small set does not support a claim of a consistent high or
low bias by itself.

## Interpretation and limits

The largest absolute error is about 55 cycles, and several other engines have
errors above 40 cycles at their single saved cutoffs. These are examples of
where this candidate can be far from the known development target. The table
does not explain why those errors occurred: it contains one prediction per
engine and does not establish a sensor-level cause or a cycle-by-cycle error
pattern. It is descriptive error inspection, not a reliability or maintenance
assessment.

No model tuning or refitting was performed. No official FD001/test data or RUL
answers were read. These results apply only to the saved FD001/train
development-validation rows and must not be represented as official-test
performance. The model, raw data, and prediction JSON remain local and
Git-ignored; this tracked report records only the selected validation examples.
