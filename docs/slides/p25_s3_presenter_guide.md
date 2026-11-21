# P25-S3 presenter guide: AI results and technical questions

Companion to [FleetPulse model/results slides](p25_ai_algorithms_results.md). The audience is the project team. The slides summarize actual saved evidence; use the linked reports for full precision and provenance.

## Six-minute talk track

### 1. Regression target

FleetPulse maps a vector of numeric summaries from the observed sensor history to an RUL estimate. For a complete training engine, a row at cycle `t` gets a separate target `L - t`, where `L` is that engine's final training cycle. The predictor sees only the observed prefix and never receives the target, future rows, or full engine lifetime. RUL is a cycle count, not a fault class.

### 2. MLP and backpropagation

The final dense model has 96 inputs, hidden widths 128 and 64, ReLU activations, and one numeric output. A saved `StandardScaler` converts each feature using training statistics. Training minimizes mean squared error. Backpropagation applies the chain rule to calculate how each weight contributed to output error; Adam updates the weights in batches. The final refit used seed 42 and the eight epochs frozen during development, without validation-based stopping. Do not conflate training loss with the held-out metrics.

### 3. Metrics and model comparison

One prediction per engine was made on 100 simulated FD001/test histories before the separate answer file was paired. MAE averages absolute cycle errors. RMSE squares errors before averaging and taking the square root, so big misses weigh more. MLP has the lowest MAE (18.1605 cycles); XGBoost has the lowest RMSE (24.1664). Development validation selected the XGBoost candidate before this final test evaluation. The test table is final reporting, not another selection round.

### 4. Explanation

For FD001/train engine 4 through cycle 164, the final tree estimated 29.7941 cycles against a SHAP reference baseline of 126.9361. The listed +6.5856 sensor_13_mean and -29.4441 sensor_4_mean contributions explain movement relative to that baseline. A contribution is neither a physical cause nor a predicted effect of changing the sensor.

### 5. Scheduling

The scheduler takes the same scenario and compares EDD with a seeded heuristic search. Across the reported scenarios/seeds, GA tied 60 times, improved 30 and regressed zero times. Explain that equality is a valid outcome: EDD already works well for some scenarios. All maintenance durations, priorities, team counts, buffers and horizons were chosen for simulation. The 120-cycle buffer was a stress setting. The study is not proof of global optimality or operational superiority.

### 6. End with evidence limits

All underlying engine histories are NASA C-MAPSS simulation data. The test set contains 100 simulated engines; development validation contains 20 different training engines. The SHAP example is one training prefix. Scheduler jobs use explicit assumptions. None establishes real fleet performance, a physical fault, individual prediction uncertainty or maintenance benefit.

## Questions and concise answers

1. **What does regression do here?** It maps the 96 numeric features to a continuous RUL value in cycles; it does not classify fault type.
2. **What is the target for an engine at cycle `t`?** Its complete training lifetime's last cycle `L` minus `t`; it stays separate from the feature vector.
3. **Why split by engine?** To evaluate on engines whose rows were not used for fitting, rather than letting related rows from one engine appear on both sides.
4. **What is backpropagation?** The chain rule calculates gradients from output error back through network layers; the optimizer uses them to update weights.
5. **Why do MAE and RMSE rank candidates differently?** MAE weights each absolute cycle error linearly; RMSE squares residuals and gives larger errors more influence. On final FD001/test, MLP leads MAE and XGBoost leads RMSE.
6. **Did the test result select XGBoost?** No. XGBoost was selected earlier on development validation RMSE. The final test scores were calculated after models were frozen for reporting.
7. **Does a positive SHAP contribution mean a sensor causes wear?** No. It is an attribution relative to a chosen reference distribution for one model output.
8. **Does a GA win every schedule?** No. In 90 paired runs it improved 30 and tied 60; it had no regressions in those tested scenarios. These counts do not generalize to every possible schedule.
9. **Are the service times or weights measured NASA values?** No. The scheduler uses them as explicit simulation assumptions because FD001 contains no maintenance-duration or priority labels.
10. **Can we use the output as a maintenance recommendation?** No. These are simulated-data model estimates and schedule comparisons, not validated maintenance evidence.

## Evidence to open

- [P25-S1 model/evaluation/explanation/scheduling chapter](../report/p25_s1_model_evaluation_scheduling.md)
- [P25-S2 captioned evidence appendix](../report/p25_s2_evidence_captions.md)
- [P13 final scores](../p13_s3_final_test_results.md) and [independent metric audit](../p13_a2_independent_metric_audit.md)
- [P11 MLP settings](../p11_s2_mlp_settings.md), [P12 final MLP refit](../p12_s2_final_mlp_refit.md), and [P12 bundle](../p12_s3_final_mlp_bundle.md)
- [P17 tree explanation](../p17_s1_final_tree_explanations.md)
- [P22 scheduling scenarios](../p22_a1_fixed_fleet_scenarios.md) and [seed comparisons](../p22_a2_seed_comparison.md)
