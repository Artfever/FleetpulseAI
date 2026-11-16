# P16-S2 standalone replay parity

The check compares every saved FD001/train development-validation prediction
with a new service inference supplied only that engine's raw history through
the saved cutoff. It verifies the source checksum, exact unique identity match
between the split and saved predictions, prefix row count, terminal cycle, and
exact raw-prediction equality. It fails on any missing, duplicate, unexpected,
or mismatched replay point.

The check reuses the saved P06 `decision-tree-rul-v1` development bundle; it
does not fit a model or read the separate FD001 test answers. These replay
points are development-validation examples, not final-model test results.

Run with the reviewed ignored FD001/train source and P06 development bundle:

```powershell
.venv\Scripts\python.exe scripts\check_replay_prefix_parity.py
.venv\Scripts\python.exe -m pytest -q tests\test_replay_prefix_parity.py
```

## Observed local result (2026-11-16)

The script compared all 20 saved development-validation engine/cycle points
with fresh `predict_rul` calls supplied each engine's standalone prefix. All
20 raw predictions matched exactly; maximum absolute difference was 0.0
cycles. The run used `decision-tree-rul-v1`, scikit-learn 1.9.1, source
SHA-256 `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`,
and model SHA-256
`2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`.

This confirms replay parity for the saved development-validation points only;
it is not a final-model evaluation or official-test result.
