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

When the verified bundle is present, the RUL Prediction page offers **Dense
MLP (CPU)** as a selectable saved model. The Model Laboratory independently
verifies the checkpoint against the same fitting/validation tables before
showing its per-engine MAE/RMSE and prediction points. Validation MSE and the
selected epoch are identified as checkpoint-selection evidence because that
split selected the checkpoint; the displayed results are not an untouched
performance estimate. No MLP selection or score is shown when its bundle is
missing or fails verification.

## Model page inspection

Using the local saved bundle with Python 3.12.10, PyTorch 2.15.0+cpu and
scikit-learn 1.9.1, Model Laboratory verified the artifact against the matching
FD001/train tables and showed 20 validation engines: MAE 15.511258 cycles and
RMSE 22.937993 cycles. The prediction plot includes the same saved per-engine
predictions. The RUL Prediction page selected **Dense MLP (CPU)** for engine 4
at cycle 164 and displayed 15.85 cycles (raw service output 15.8526315689,
model `dense-rul-mlp-v1`). This single-row service output differs from the saved
batch oracle by 0.0000019073 cycles because CPU floating-point execution varies
slightly with batch shape. These remain development-validation results; the
checkpoint was selected using these validation targets, and no official test
partition or answers were used.

## Saved learning curves

The Model Laboratory plots the training and validation MSE arrays stored in the
verified manifest; it does not recompute or smooth them. Both curves report mean
squared error in cycles² per epoch. The training curve aggregates 16,738
cycle-level fitting rows, while the validation curve aggregates 20 held-out
engine prefixes (one saved cutoff per engine). The shared loss units do not make
the sample units interchangeable, so their absolute levels should be read with
that distinction in mind. The lowest saved validation MSE is 526.1515625
cycles² at epoch 12; that curve selected the checkpoint, so it is selection
evidence rather than an untouched final estimate. The plot labels the split,
sample unit/count, epoch, and loss units and repeats this caveat below the chart.

## App inference does not train

The app selects an available saved bundle and the prediction service loads its
CPU weights for inference. Training remains an explicit offline operation via
`python scripts/train_mlp.py`; app startup and prediction do not invoke
`train_mlp`. A regression test creates a small synthetic saved checkpoint,
replaces the trainer with a function that fails if called, opens the prediction
page, and verifies a finite Dense MLP result without any trainer calls.

The local saved development bundle was also exercised through the prediction
page with training disabled. On the packaged FD001/train prefix for engine 4 at
cycle 164, the page displayed 15.85 cycles and model `dense-rul-mlp-v1`; the
service raw output is 15.8526315689 cycles using the saved 30-cycle feature
window. The local verification observed zero training calls and no Streamlit
page exceptions. This uses the validation-selected development bundle and is
not official-test performance or a maintenance guarantee.

## P10-A2 inference contract verification

The saved bundle declares `dense-rul-mlp-v1`, schema
`mlp-validation-checkpoint-v1`, and ordered `past-only-v1` 30-cycle features
(96 columns). Its CPU `DenseRULRegressor` has 64/32 ReLU hidden layers and one
output; the saved `StandardScaler` was fitted on 16,738 fitting rows. Runtime
versions for this bundle are Python 3.12.10, PyTorch 2.15.0+cpu, and
scikit-learn 1.9.1. Automated tests check one output per input row, reject
wrong input width and altered feature order, confirm CPU/eval loading, and
require finite service predictions.

`python scripts/verify_mlp_artifact.py` was run as a separate process against
the local bundle. It reported `prediction_count: 20`, `device: cpu`,
`eval_mode: true`, best epoch 12, and maximum replay difference 0.0 cycles.
The saved predictions are aligned to 20 FD001/train engine prefixes. As a
separate service check, engine 4 at observed cycle 164 returned
15.8526315689 cycles (displayed 15.85). No official test histories or answers
were used; the checkpoint was selected using the validation split, so these
results are development evidence rather than an untouched final estimate.

## P10-A3 scaler provenance check

The scaler is fitted in `train_mlp` on fitting features only. Tests independently
fit `StandardScaler` on fitting rows and compare the saved mean, scale, variance
and sample count; the same test confirms validation evaluation leaves the
statistics unchanged and that combining validation rows would change them.
The artifact round-trip test also compares all serialized scaler numbers to the
trained scaler.

For the actual local FD001/train bundle, the saved scaler reports 16,738 fitted
rows and 96 features. Compared with a newly fitted StandardScaler over the
matching fitting table, maximum absolute differences in mean, scale and
variance were each 0.0. Fitting over the combined fitting and 20 validation
rows instead would change maximum absolute mean by 0.0024867583, scale by
0.0029577058, and variance by 0.0970157967. This confirms the recorded values
come from fitting rows only; validation rows were not used to fit the scaler.
