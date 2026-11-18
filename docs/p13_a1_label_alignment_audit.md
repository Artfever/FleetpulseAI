# P13-A1: Independent FD001 test-label alignment audit

This audit uses a separate implementation in `scripts/audit_p13_label_alignment.py`; it does not import or call Shaheer's P13-S1/S2 evaluator or pairing functions. It independently inspects the original test file, reads the one-column RUL vector, and compares its explicit line mapping against every saved model row in the P13-S2 pairing artifact.

## Alignment rule and evidence

- The RUL file is a vector without engine IDs. Apply the task's explicit convention: answer line `i` maps to the `i`th distinct test engine in sorted `unit_id` order.
- The actual `test_FD001.txt` contains 100 complete engine blocks, and first-occurrence/block order is exactly IDs 1–100. The separate RUL file contains 100 integer answers. Therefore the mapping checked is line 1 → engine 1, line 2 → engine 2, …, line 100 → engine 100.
- The original archive readme describes a vector of test RUL answers but does not embed unit IDs or spell out that mapping. The mapping is recorded explicitly because the source format has no identifiers; this audit verifies consistency with the task convention and actual file order. It cannot infer an alternative permutation from the label vector alone.
- For each of five frozen models, independently check exactly one finite prediction for each ID, saved answer line and true label match the independently loaded vector, and saved observed cycle matches that engine's last test cycle.

## Actual run

Command: `.venv/Scripts/python.exe scripts/audit_p13_label_alignment.py`

- Test file SHA-256: `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`.
- RUL file SHA-256: `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`.
- Paired input payload SHA-256: `2f5a05e3015602405857950824d71399f8b9352d1ab7012129208304b4f41730`.
- Observed 100 test engines, 100 answer rows, and 500 audited model/engine pair rows. The test history's first-occurrence order matched sorted IDs. First and last mappings were 1→1 and 100→100.
- Ignored local audit: `artifacts/final/p13/a1_alignment_audit.json`, payload SHA-256 `3f9a10ac187263d6113ed99cfc70b5700f84ebaacd41b7a54b8354ee6fcfa784`; whole-file SHA-256 `e7179fffaf7d9a189402ad9d4c49437cb79b98c582414eb37109488340c4c283`.
- Four independent audit tests pass, including shuffled toy IDs `(30, 10, 20)` mapped to answer lines by sorted ID, duplicate/count rejection, and joining shuffled model rows by engine ID while checking cutoffs.

The audit calculates no scores and performs no tuning. It verifies the mapping convention and consistency of saved rows; P13-A2 independently recalculates MAE/RMSE. No teammate audit or approval is claimed.
