# P25-T3: Repeatable FleetPulse live demo

Issue: [P25-T3 #235](https://github.com/Artfever/FleetpulseAI/issues/235)  
Inputs: [assembled report](report/p25_final_report.md), [checked slide PDF](../output/pdf/p25_final_deck.pdf), and [local browser evidence](p24_t1_browser_demo_evidence.md).

## Demo goal and limits

Show one full user route: prepare an observed history, validate the upload, request a saved-model RUL estimate, pass that result into the planner, and compare EDD with the Genetic Algorithm on the same scenario. Then point out where Model Laboratory separates development evidence from final FD001/test reporting.

This is a local software demo over simulated NASA C-MAPSS FD001 data. The RUL estimate is in operating cycles. It is not an aircraft fault diagnosis or a maintenance recommendation. Team capacity, durations, priority weights, buffer, and horizon are explicitly illustrative schedule assumptions. Do not claim the deployed Streamlit revision is current based on these local steps.

## Presenter roles and timing (8 minutes)

| Time | Presenter | Segment |
| --- | --- | --- |
| 0:00–0:45 | Abdullah | Problem, intended course-project user, Path A, and simulation/maintenance boundary. |
| 0:45–2:00 | Taha | Home/navigation, exact-schema upload, input validation, prediction and visible model/cycle context. |
| 2:00–3:30 | Taha | Save the successful result to Planner; explain the shared scenario, assumptions, EDD/GA metrics, and soft horizon. |
| 3:30–5:15 | Abdullah | Explain 26 raw fields, partition-qualified engine IDs, 96 ordered past-only features, and train/validation/test separation. |
| 5:15–6:45 | Shaheer | Explain MAE versus RMSE and the five frozen FD001/test results; distinguish final reporting from model selection. |
| 6:45–7:30 | Shaheer | Explain one SHAP attribution and scheduler comparison without causal or operational claims. |
| 7:30–8:00 | All | Cross-module questions; close with data and human-validation limits. |

The table is a proposed rehearsal plan, not evidence that the group has rehearsed. The group must agree presenter assignments and complete a timed rehearsal separately; no teammate coordination is claimed in this runbook.

## Preflight (about 5 minutes before presenting)

1. Use the repository's supported Python 3.12 environment. In PowerShell from the repository root, run:

   ```powershell
   .\.venv\Scripts\python.exe --version
   .\.venv\Scripts\python.exe -m ruff check .
   .\.venv\Scripts\python.exe -m pytest -q
   ```

2. Confirm the tracked demo subset exists at `data/demo/fd001_training_prefixes.txt`. Do not download or substitute unlabeled/raw files during the demo.
3. Start the app from the repository root:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app/Home.py
   ```

4. Open the local URL printed by Streamlit. Confirm Home and all six navigation items appear. If the app reports that a model bundle is unavailable, stop the prediction route and use the read-only report/PDF instead; do not invent a result or retrain during the presentation.
5. Create a temporary, headered upload CSV from the tracked sample. This writes outside the repository by default and includes only the 26 raw-history fields:

   ```powershell
   .\.venv\Scripts\python.exe scripts/export_demo_upload.py --unit-id 4
   ```

   The command prints the generated path and row count. For the recorded prefix it should say 164 rows through cycle 164. To choose another output location, pass `--output <path>`.

## Click-by-click flow (3½ minutes)

1. **Home:** point out the navigation and the simulated-data notice. Say that the app estimates cycles; it does not diagnose a physical fault.
2. **RUL Prediction:** choose **Upload CSV** as the history source.
3. In **Engine history CSV**, select the path printed by `export_demo_upload.py`.
4. Confirm the app reports successful validation and shows engine 4 with 164 of 164 rows. If it reports a schema or unavailable-model error, show the error honestly and switch to the report/screenshots; do not edit column meanings or assert a prediction.
5. Keep the selected `decision-tree-rul-v1` bundle. Click **Predict remaining cycles**. Point out raw/display RUL around **34.20 cycles**, observed cycle **164**, and **30** feature-window rows. The stored browser record has the exact raw output `34.19607843137255` cycles.
6. Open **Maintenance Planner**. Add/save the prediction if the UI asks, then click **Build selected-fleet planning inputs**. Confirm the row carries `user/upload`, engine 4, and planner RUL around 34.1961 cycles.
7. Set one team, duration 1, priority 1, buffer 2, and soft horizon 6 only if those are not already visible. State that these are demo assumptions, not NASA measurements.
8. Compare EDD and GA. For this one-job case, both should show zero weighted tardiness, makespan 1 cycle, and overflow 0. The purpose is to show how one prediction flows into a shared scheduling scenario, not to claim a GA advantage.
9. Open **Model Laboratory**. Point to the dedicated **Official held-out test results (FD001/test)** table, separate from the P22 study table. Explain that the five scores summarize 100 simulated test engines and were not used for model selection.
10. End with the limitations: simulated FD001 only; no per-engine uncertainty guarantee or real-fleet evidence; no maintenance-duration labels; a local run does not verify the current hosted SHA.

Recorded reference values, versions, screenshots, and the prior six-screen browser result are in [P24-T1 browser evidence](p24_t1_browser_demo_evidence.md). The checked Model Laboratory screenshot and its limitations are in [P24-T2 evidence](p24_t2_screenshots_known_issues.md). Those are prior local captures; this guide does not claim a new browser rehearsal.

## Cross-module questions for the group

1. **Why can the predictor not see the complete training lifetime or test answers?** They would leak future/answer information into the feature vector. Validation labels are derived separately after choosing the observed cutoff.
2. **Why split by engine instead of randomly splitting rows?** Rows from one engine share a trajectory; splitting rows could let the model see the same engine during both fitting and validation.
3. **What does 34.20 mean, and what does it not mean?** It is a saved-model estimate of remaining cycles for one observed simulated history, not a fault label, accuracy score, safety guarantee, or maintenance deadline.
4. **Why do MAE and RMSE name different leaders?** MAE weights absolute errors linearly; RMSE gives larger errors more influence. On this fixed test set MLP has the lower MAE and XGBoost the lower RMSE.
5. **Does the SHAP value say sensor 4 caused degradation?** No; it is a signed model attribution relative to a reference dataset for one prefix.
6. **Are planner duration, priority, buffer, team, or horizon values measured?** No. They are visible simulation assumptions because FD001 has no maintenance-resource labels.
7. **Does GA always beat EDD?** No. In the bounded saved study there were ties and improvements, not proof of universal or globally optimal behavior.
8. **Can we present these outputs as aircraft maintenance advice?** No. The benchmark is simulated and the tool is an educational prototype.

## Recovery and optional backup

- If upload fails, show the validation message and switch to the packaged Fleet Overview subset (engines 4, 5, and 12); explain that it is a small parity demo, not a complete fleet.
- If model artifacts are unavailable, do not retrain or make up a number. Present the checked PDF/report and the saved local screenshots, labeling them as prior local evidence.
- If Streamlit cannot start, use the exported [10-slide PDF](../output/pdf/p25_final_deck.pdf) and the screenshots in `images/p24_t1/` and `images/p24_t2/`. The optional backup is a slide/screenshot walkthrough; no recording file is included.
- Do not present a Streamlit Cloud deployment SHA as verified by this local runbook.

## Verification record

The underlying uploaded-history and planner interaction has a prior local-browser record: [P24-T1 evidence](p24_t1_browser_demo_evidence.md), which records Chrome 158.0.8127.36, Streamlit 1.67.0, 6 screens, successful 164-row upload, raw RUL `34.19607843137255`, and paired one-job planner output. The [P25-T2 audit](p25_t2_layout_reference_audit.md) records the current assembled deck export/layout check. These provide the source evidence for the click path; they are not a new group rehearsal or a new remote deployment test.
