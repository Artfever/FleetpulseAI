# P08-S3 forest feature-version audit

**Issue:** [P08-S3 #246](https://github.com/Artfever/FleetpulseAI/issues/246)

**Inputs:** local, Git-ignored `artifacts/development/random_forest_validation.json`,
`random_forest_metrics.json`, and feature tables

**Audit date:** 2026-11-16

## Result

The saved forest manifest and metrics record agree on both model identity and
the exact feature configuration. No version or feature-order mismatch was
found.

| Check | Observed value | Result |
| --- | --- | --- |
| Validation-manifest schema | `random-forest-validation-v1` | Pass |
| Metrics-record schema | `metrics-v1` | Pass |
| Model identity in both records | `random-forest-rul@random-forest-rul-v1` | Match |
| Feature contract version | `past-only-v1` | Match |
| Window | 30 cycles | Match |
| Ordered raw channels | 24: `setting_1` through `setting_3`, then `sensor_1` through `sensor_21` | Match |
| Feature schema | 96 columns: latest, mean, population standard deviation, and slope for each ordered channel | Match |
| Metrics feature config vs. manifest training config | Exact JSON equality | Match |
| Manifest feature names vs. current `FeatureConfig.output_schema` | Exact ordered equality | Match |
| Saved model file | SHA-256 `b9a7bcf7b65ae82cf9c499170325de7b24ac5433d440f45746a07e1a4bb8c70b` | Matches manifest; checked without deserialization |

The common metrics record also agrees with the manifest's split, fitting and
validation table hashes, random seed, estimator parameters, runtime, validation
partition and per-engine metrics. The existing read-only validation audit
loads both feature tables against the manifest's `FeatureConfig`; the dataset
loader validates the saved config version and exact ordered feature names.

## Example and provenance

For the saved validation row `FD001/train`, engine 4, cutoff cycle 164, the
feature table records 164 observed rows through the cutoff, 30 rows in the
configured window and 96 output values. The first four values are named
`setting_1_latest`, `setting_1_mean`, `setting_1_std` and `setting_1_slope`.
This confirms the example stays within the observed engine prefix and follows
the declared channel-major feature order. No physical sensor units are inferred
from these values.

The independently audited validation set has 20 engines, MAE **18.833479230754712**
cycles and RMSE **25.24302308237894** cycles. Its saved source is FD001/train
development validation. These values are not official-test results.

## Verification and limits

Command: `.venv/Scripts/python.exe scripts/audit_random_forest_validation.py`

Result: **PASS**. All 20 validation identities, targets and row errors were
checked; manifest and metrics-v1 agree; the saved model's SHA-256 matches the
manifest without loading or deserializing the model.

No refit or tuning was performed. No official FD001/test file or RUL answers
were read. This task checks version and metadata consistency; independent
model reload parity and real forest inference in the app remain Phase 8 shared
gate evidence, not claims established by this audit.
