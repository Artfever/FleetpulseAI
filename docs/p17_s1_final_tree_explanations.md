# P17-S1: frozen Decision Tree explanations

## Service contract

`fleetpulse.services.explain_final_tree_prediction` explains one validated
observed FD001 engine prefix with the frozen P12 Decision Tree. It reads the
Decision Tree manifest and weights through the checksum-verifying P12 loader,
then constructs the exact ordered feature row from that bundle's saved
configuration. It accepts observed raw history only; targets and future rows
are not prediction inputs. The explanation contract is proposed for downstream
P17 consumers and includes:

- model name/version, dataset-qualified engine and observed cutoff;
- raw, unclipped RUL in cycles and SHAP baseline value in cycles;
- every signed feature contribution, named and ordered exactly as the saved
  feature schema, alongside the feature value used;
- a top-ten mean absolute SHAP summary over the fixed reference rows;
- hashes for the model manifest/payload, weights, feature configuration, input
  history, training source, and reference feature matrix.

SHAP uses a frozen sklearn `DecisionTreeRegressor`, `TreeExplainer` with
`feature_perturbation="interventional"` and `model_output="raw"`. The result
checks that baseline plus all 96 local contributions reconstructs the same
raw prediction within `2e-5 + 1e-7 * abs(prediction)` cycles. This tolerance
covers floating-point attribution summation while remaining far below a
hundredth of one operating cycle. The result's
bounded `top_contributions(10)` helper sorts by absolute signed contribution
with schema order as the stable tie break.

## Reference data and limitations

The background is a fixed, feature-only sample from the original
`data/raw/train_FD001.txt`: one prefix per each of the 100 training engines,
ending at observed cycle 90. The existing saved P05 feature configuration
engineers each prefix. The full source-file SHA-256 must match the frozen P12
model's training-source hash; reference feature names/order must match the
model; and the calculated matrix has its own content hash. The builder reads
the raw 26-column history file only. It does not load RUL answers or official
test data.

SHAP distributes this one model's raw output relative to this selected
reference distribution. Feature dependence is handled by the interventional
assumption; correlations among sensor statistics can produce masked feature
combinations that do not occur in real histories. Contributions are not
causal effects, uncertainty, confidence, or model accuracy. This output is
one engine at one cutoff, not fleet-wide behavior.

The base application pins `shap==0.54.1`, so the RUL page can show explanations
after an explicit user action. The TreeExplainer options and
background semantics follow the [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

## Verification

Synthetic automated coverage is in `tests/test_explanations.py`. The local
frozen-bundle audit command is:

```powershell
.venv/Scripts/python.exe scripts/check_p17_s1_explanation.py
```

Its ignored JSON report is written to
`reports/P17_S1_final_tree_explanation.json`; it records the full ordered
contributions and bounded global summary. Model and raw data remain local and
are not committed.

Actual local frozen-bundle execution (18 November 2026) passed for packaged
FD001/train engine 4 at cycle 164. The model returned raw RUL **29.7941176471
cycles**. The reference baseline was **126.9360964740 cycles** and the sum of
the 96 signed feature contributions reconstructed the prediction with an
absolute discrepancy of **0.0000045323 cycles**, below the configured
`2e-5 + 1e-7 * abs(prediction)` tolerance. The ten largest local contributions
by magnitude were led by `sensor_4_mean` at −29.4441 cycles,
`sensor_9_mean` at −13.3104, and `sensor_21_mean` at −11.7710. A negative
contribution lowers this model output relative to this background baseline;
it does not show that changing that sensor would cause the same change.

Across the 100 fixed reference rows, the largest mean absolute SHAP values
were `sensor_21_mean` 12.6610, `sensor_13_mean` 11.2342, `sensor_4_mean`
7.2478, `sensor_9_mean` 5.0074 and `sensor_4_slope` 4.9970 cycles. This is a
summary of this model's allocations on this selected reference sample, not
feature importance for the physical engine or a general population claim.

## Prediction/explanation consistency boundary (P17-A3)

`check_final_explanation_consistency` binds a `FinalTreeExplanation` to the
matching successful `FinalModelPrediction` in one `FinalHistoryComparison`.
It checks engine/partition/cutoff and canonical input-history hash, model
version, manifest and weight hashes, feature-config and training-source hashes,
all ordered feature names and values, and raw RUL equality within `1e-12`
cycles. It returns stable mismatch codes if any check fails. It requires the
raw comparison object; a presentation-only/clamped prediction object is
rejected, and raw explanations remain separate from display values.

The actual local integration audit is:

```powershell
.venv/Scripts/python.exe scripts/check_p17_a3_explanation_consistency.py
```

It runs the final-model comparison and explanation on the same packaged
engine-4/cycle-164 prefix, then records both outputs and the consistency
result in ignored `reports/P17_A3_explanation_consistency.json`.

Actual local run on 18 November 2026 returned `is_consistent=true` with no
mismatch codes. Prediction and explanation both reported raw RUL
29.794117647058822 cycles, model version `decision-tree-rul-v1`, identical
manifest/weight/feature-config and input-history hashes, and matching 96
ordered feature names and values. The explanation's SHAP additivity difference
remained 0.00000453222900986816 cycles.
