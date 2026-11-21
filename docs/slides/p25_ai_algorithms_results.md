---
marp: true
theme: default
paginate: true
---

# FleetPulse: models, explanations, and planning

### Regression, measured results, and limitations

Shaheer AI and evidence | P25-S3

NASA C-MAPSS FD001 is simulated data. RUL estimates are measured in operating cycles; they do not diagnose faults or establish maintenance needs.

<!--
Source: docs/report/p25_s1_model_evaluation_scheduling.md
Source: docs/report/p25_s2_evidence_captions.md
Audience: project teammates. Suggested duration: 6 minutes.
-->

---

## The model learns a number: remaining cycles

For a complete training engine whose last cycle is `L`, the target at cycle `t` is:

```text
RUL(engine, t) = L - t
```

The target is a separate numeric label in cycles. The input is an ordered vector of 96 `past-only-v1` features from an observed window of up to 30 cycles. Engine identity, full lifetime, future rows, and true RUL are not predictor features.

**Regression** means learning a function from those numeric sensor summaries to a continuous cycle estimate. It does not classify a fault type.

<!--
Ask: What is known for a training label that must remain hidden from the predictor?
Answer: The full training lifetime, including future cycle rows.
Source: docs/training_targets.md; docs/feature_contract.md; docs/report/p25_a1_data_features_architecture.md
-->

---

## Fit on some engines; check on different engines

The development manifest holds out entire engines: 80 fit engines and 20 validation engines. The same engine's rows do not appear on both sides.

The wider MLP uses 96 standardized inputs, hidden widths 128 and 64 with ReLU, and one RUL output. Its training minimizes mean squared error. Backpropagation applies the chain rule from output error through the layers; Adam uses those gradients to update weights in mini-batches of 128.

For the final MLP, the scaler used 20,631 past-only rows across 100 `FD001/train` engines. Architecture, seed 42, CPU and an eight-epoch training rule were frozen from development. The final refit did not use validation for early stopping.

<!--
Training configuration is taken from P11-S2 and P12-S1/S2. Training rows are simulated data. This is an explanation of the implemented training procedure, not a new fit.
-->

---

## Metrics answer different questions

| Frozen FD001/test model | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| Decision Tree | 24.5133 | 33.9522 |
| Random Forest | 20.3497 | 27.5200 |
| XGBoost | 18.4489 | **24.1664** |
| Constant mean | 40.8793 | 52.6249 |
| Dense MLP | **18.1605** | 25.5735 |

**MAE** is the average absolute prediction error. **RMSE** is the square root of average squared error, so larger misses have more influence. Here the MLP has lower MAE, while XGBoost has lower RMSE. XGBoost was selected earlier using development validation RMSE; these final test results are reporting only and did not change that choice.

All rows summarize the same 100 held-out simulated test engines. They are not accuracy percentages or guarantees for a future engine.

<!--
Exact full-precision values and pairing source: docs/p25_s2_evidence_captions.md and docs/p13_a2_independent_metric_audit.md
-->

---

## A tree explanation is an attribution

For packaged `FD001/train` engine 4 through cycle 164, the frozen final Decision Tree output was **29.7941 raw RUL cycles**. Its reference baseline was **126.9361 cycles**.

- `sensor_13_mean`: **+6.5856 cycles** relative to the baseline
- `sensor_4_mean`: **−29.4441 cycles** relative to the baseline

These signed SHAP contributions describe how this model allocated its output relative to a fixed 100-prefix training reference. They do not show that a sensor caused wear, identify a fault, or establish what a physical sensor change would do.

<!--
This is one simulated training-history explanation, not an accuracy result. Source: docs/p17_s1_final_tree_explanations.md
-->

---

## The scheduler compares orders under assumptions

EDD is a deterministic earliest-due-date baseline. The seeded Genetic Algorithm searches job orders for lower weighted tardiness. Across 18 fixed simulated scenarios and five seeds each:

**90 paired comparisons: 60 ties · 30 GA improvements · 0 regressions**

Ties count as valid results. For example, a five-engine scenario with a two-cycle buffer tied at zero for all seeds. A ten-engine, one-team stress case with a 120-cycle buffer had EDD 85.102 weighted cycles; GA scores ranged from 61.600 to 67.000. Durations, priorities, teams, buffers and horizon were scenario assumptions.

This bounded result does not prove GA is always better or globally optimal. Schedule scores are not maintenance evidence.

<!--
Source: docs/p22_a1_fixed_fleet_scenarios.md and docs/p22_a2_seed_comparison.md
-->

---

## What we can and cannot claim

- **Can say:** on these 100 simulated FD001/test engines, the MLP had the lowest MAE and XGBoost the lowest RMSE.
- **Can say:** SHAP assigned signed contributions for this frozen tree and selected reference set.
- **Can say:** GA tied or improved EDD on the reported fixed scenarios and seeds.
- **Cannot say:** these results predict a real aircraft fleet, diagnose a failure, quantify individual-engine uncertainty, or prove maintenance benefit.
- **Cannot say:** duration, priority, capacity, buffer, or horizon values came from NASA; they are simulation inputs.

Read the source-linked [P25-S1 model and evaluation chapter](../report/p25_s1_model_evaluation_scheduling.md) and [P25-S2 result captions](../report/p25_s2_evidence_captions.md) for methods, exact values, hashes, and reproduction notes.
