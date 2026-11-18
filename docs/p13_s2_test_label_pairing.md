# P13-S2: Pair FD001/test predictions with the separate RUL vector

`python -m fleetpulse.evaluation.official_test_pairing` verifies the saved P13-S1 prediction artifact and the original answer-file checksum, reads `RUL_FD001.txt` only in this held-out scoring stage, then creates an explicit line-to-engine map. It does not calculate metrics; P13-S3 owns the full result/error presentation and P13-A2 independently recalculates final scores.

## Alignment contract

- Require the P13-S1 artifact schema, valid payload checksum, frozen model versions, and the pinned official `test_FD001.txt` checksum.
- Require exactly the official FD001 test IDs 1 through 100 and exactly 100 integer RUL answers.
- Sort the distinct test `unit_id` values and map RUL line `i` to sorted engine ID at position `i` (1-based). Store both `answer_line` and `unit_id` on every paired record.
- For each model, require one and only one prediction per engine and require each prediction cutoff to match the saved engine cutoff. Reject missing, duplicate, extra, stale, non-finite, or misaligned rows.
- Preserve raw predictions and labels in cycles. Do not clip, compute scores, or use results to retune.

The original archive readme describes a vector of final test RUL values; it contains no engine-ID column. The implementation applies the issue-specified sorted-ID mapping and records the exact mapping in the output. The independent P13-A1 audit (#66) remains responsible for its separate shuffled-ID and actual-data audit.

## Observed local run

Command: `.venv/Scripts/python.exe -m fleetpulse.evaluation.official_test_pairing`

- Test-history SHA-256: `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`.
- Original RUL file SHA-256: `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`.
- The reader returned 100 answers; engine IDs were exactly 1–100. The pairing includes 5 models × 100 engines = 500 rows, ordered by engine ID, with answer line and observed-cycle identity preserved. No metric was calculated (`scores_computed: false`).
- First mappings from the saved vector: line 1 → engine 1, RUL 112 cycles; line 2 → engine 2, RUL 98 cycles; line 3 → engine 3, RUL 69 cycles; …; line 100 → engine 100, RUL 20 cycles.
- Engine 1 example (observed cycle 31; true RUL 112 cycles): Decision Tree `164.96629213`, Random Forest `153.98205177`, XGBoost `145.28688049`, constant mean `107.80786196`, final MLP `158.88244629` cycles.
- Ignored local result: `artifacts/final/p13/official_test_pairings.json`. Payload SHA-256: `2f5a05e3015602405857950824d71399f8b9352d1ab7012129208304b4f41730`; whole-file SHA-256: `a07811927f729e6a92037288fac43aded185d58124204214a528bcf8a45f8519`.

## Verification and boundary

- Shuffled synthetic engine rows verify that answer lines attach by sorted ID, not incoming row order. Tests also reject wrong answer counts, duplicate predictions, mismatched cutoffs, and invalid prediction checksums.
- `.venv/Scripts/python.exe -m pytest tests/test_official_test_pairing.py -q`: 4 passed.
- The answer file was used only for pairing here. It must not be used to tune, select, or refit the frozen models. This artifact is paired data, not metric output or proof of generalization.
