# P12-S1: final MLP scaler fitted on original training engines

## Result

Created a fresh `StandardScaler` using one 96-feature `past-only-v1` vector at
every observed cycle across all 100 FD001/train engines. The selected 30-cycle
window and 24 channels match the frozen P11 configuration. The 20,631 rows
include each engine's terminal cycle; each feature row receives only that
engine's prefix through the row's cycle. RUL labels and official test data are
not used by the scaler fit.

Observed source SHA-256:
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
P11 selection manifest payload SHA-256:
`4cbda49f05eb928ebacc0a673de175dfae24b0017f7798db62a37cda8462fa2d`.
The ignored final artifact is
`artifacts/final/mlp/scaler_manifest.json`; its payload SHA-256 is
`b56c8d1d3a8b946f48bbfd231ff9f6596c44a3ccbf2ab2235c69f4acde4f69cd`.

The artifact records the ordered feature schema, source and selection hashes,
100 engine identities and row counts, 20,631 fit rows, and the scaler means,
scales and variances. Its loader checks the canonical payload hash, source,
feature order, sample counts, and expected P11 feature configuration. The
script reloaded the saved manifest, matched all scaler arrays exactly, and
transformed all 20,631 rows to finite values.

## Reproduce

With the local original `data/raw/train_FD001.txt` and the tracked P11 selection
manifest present, run:

```powershell
.venv/Scripts/python.exe scripts/fit_p12_mlp_scaler.py
.venv/Scripts/python.exe -m pytest tests/test_final_scaler.py -q
```

The scaler is stored under `artifacts/final/mlp/`, separate from the development
MLP bundle. This is a preprocessing artifact for the subsequent frozen-epoch
MLP refit; it is not a trained final MLP or a deployment claim.
