# P12-S2: final MLP fixed-epoch refit

The final MLP uses the P11-frozen `(128, 64)` architecture, seed `42`, batch
size `128`, Adam learning rate `0.001`, and exactly eight epochs. The eight
epochs were selected during P11 development. This refit does not reuse the
validation cohort to select epochs and has no validation or early-stopping
input.

The script rebuilds one past-only 96-feature row for every observed cycle in
all 100 original FD001/train engine histories. Each target is the uncapped
engine-specific remaining-cycle count derived from that complete training
history. A fresh scaler from P12-S1 is loaded and checked against the source,
P11 selection, and feature configuration. Official test histories and labels
are not read.

## Reproduce

With the original `data/raw/train_FD001.txt`, P11 selection manifest, and
P12-S1 scaler artifact available locally:

```powershell
.venv/Scripts/python.exe scripts/refit_p12_final_mlp.py
.venv/Scripts/python.exe -m pytest tests/test_final_mlp.py -q
```

The script writes `artifacts/final/mlp/refit_state.pt` and
`artifacts/final/mlp/refit_manifest.json`. Both are local ignored artifacts;
P12-S3 packages the final model separately from development models. The
manifest records hashes, source lineage, feature schema, the fixed training
rule, per-epoch training losses, and runtime. Training losses are not
generalization metrics.
