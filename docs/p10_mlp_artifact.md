# P10 MLP checkpoint artifact

`fleetpulse.models.mlp_artifact` writes the P10-S3 validation-selected MLP bundle.
The development files live under ignored `artifacts/development/mlp/` and are not
committed because they are derived from local FD001 training data.

## Files and verification

- `mlp_best_state.pt`: CPU-only Torch `state_dict`; load with `map_location="cpu"`
  and `weights_only=True`.
- `mlp_training_manifest.json`: hashed JSON manifest containing the architecture,
  feature order/configuration, StandardScaler statistics fitted on fitting rows,
  run parameters, best epoch, training and validation MSE curves, input hashes,
  and the validation prediction rows.

Run `python scripts/train_mlp.py` to perform one bounded CPU fit (at most 20
epochs, default seed 42) and write the bundle. The command starts
`scripts/verify_mlp_artifact.py` in a fresh process. The verifier checks hashes
and data provenance, restores the architecture and scaler, loads weights on CPU
in evaluation mode, and requires exact replay of all saved validation
predictions. The JSON and weights are local development outputs; this code and
document define their reproducible format.

## Observed run

On 2026-11-16, Torch 2.15.0+cpu trained against 16,738 FD001/train fitting rows
and 20 engine-prefix validation rows with 96 `past-only-v1` features. Seed 42,
batch size 128, hidden layers 64/32, Adam learning rate 0.001, MSE loss, maximum
20 epochs, and patience 3 were used. Early stopping ended after 15 epochs and
selected epoch 12. The selected validation MSE was 526.1515625 cycles²; the
training run took 13.870148 seconds. The fresh-process verifier reloaded the
CPU checkpoint in evaluation mode and reproduced 20/20 validation predictions
with maximum difference 0.0 cycles. The model weights SHA-256 was
`a5dcf637a1178f44c837688646295250971dca5862cd934e3935293be53e6273`.

Validation labels selected the checkpoint, so this score is selection evidence,
not an untouched performance estimate. No official test data or answers were
read. The bundle does not establish deployment, maintenance readiness, or
performance on unseen engines.
