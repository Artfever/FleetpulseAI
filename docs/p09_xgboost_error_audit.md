# P09-S1: independently recomputed XGBoost errors

## Method and provenance

`scripts/audit_p09_xgboost_errors.py` independently recalculates absolute error,
squared error, MAE and RMSE from the saved prediction rows in the Decision
Tree, Random Forest and XGBoost validation manifests. Before comparing scores,
it checks each manifest content hash, requires 20 unique engine-level rows,
and verifies the same ordered partition/engine/cutoff/target identities,
feature schema, source digest, split digest and validation-table digest across
all three models. It also verifies every saved row error and aggregate against
fresh calculations within `1e-12` absolute tolerance.

The common set contains 20 unique held-out engine prefixes from `FD001/train`.
All use `past-only-v1`, the 30-cycle window and the same 96-feature schema.
Source SHA-256 is
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`, split
manifest digest is
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`, and
validation-table payload digest is
`f843a342a3f8799a22a1d386936e9f4c2ab57b6f7cb30cc21885a2b989fcdff7`.

## Independently observed results

| Candidate | Version | MAE (cycles) | RMSE (cycles) |
| --- | --- | ---: | ---: |
| Decision Tree | `decision-tree-rul-v1` | 22.494270 | 29.449574 |
| Random Forest | `random-forest-rul-v1` | 18.833479 | 25.243023 |
| XGBoost | `xgboost-rul-v1` | 15.108697 | 21.092113 |

On these 20 validation engines, XGBoost's MAE was 7.385573 cycles lower than
the Decision Tree's and 3.724783 cycles lower than the Random Forest's. Its
RMSE was 8.357460 and 4.150910 cycles lower, respectively. The per-engine
absolute-error comparison also shows that aggregate scores do not mean XGBoost
won for each engine: its error was higher on 6/20 engines than the tree and
7/20 engines than the forest, lower on the rest, and tied on none.

For engine 4 observed through cycle 164, all three targets are 25 cycles:
Decision Tree predicted 34.196078 cycles (absolute error 9.196078), Random
Forest predicted 34.043205 (error 9.043205), and XGBoost predicted 28.693811
(error 3.693811). This is one example from the shared validation set, not an
independent test case.

## Limits and reproduction

These are development-validation metrics on one saved cutoff per engine in the
FD001 training partition. They are not official-test results or evidence of
future accuracy, maintenance readiness, or model superiority outside this
small split. The three validation manifests and model bundles remain ignored
local artifacts; no model binary or official test data is committed.

With the verified local artifact bundles available, run:

```powershell
.venv/Scripts/python.exe scripts/audit_p09_xgboost_errors.py
```

The audit does not fit or deserialize any estimator; it checks the three saved
validation manifests and prints the exact independent recomputation, model
differences, source digests, and engine 4 example. The model bundle hashes are
`2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018` (tree),
`b9a7bcf7b65ae82cf9c499170325de7b24ac5433d440f45746a07e1a4bb8c70b` (forest),
and `20801e897816f879c0f93f56dc199c49735c3afc3a544b595247f5c62d419612`
(XGBoost). The corresponding manifest payload hashes are
`fd0382e90385109ca8b67ffd6aa536383c63e8f363838c5a5a5553120d8c12f6`,
`92385dacb77b333280f63a97474c2abcbae84e50c9c75d1c64a725492a99ce96`, and
`33f4243b111c75881051a3d5d9b2972dd16aacea6960c10fa3034df19b5557d5`.
