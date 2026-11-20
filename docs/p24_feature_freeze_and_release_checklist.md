# P24 feature freeze and release checklist

Issue: [P24-T3 #232](https://github.com/Artfever/FleetpulseAI/issues/232)

## Feature freeze

As of the P24 evidence merged to `dev` on 2026-11-21, the principal FleetPulse interface is frozen for release review. The app keeps its six current screens and their existing workflows:

1. **Home** — project scope, limitations, and navigation.
2. **Fleet Overview** — saved-model fleet estimates for the packaged FD001/train parity subset, with engine identity and threshold context.
3. **Engine Explorer** — inspect an engine's observed sensor history and past-only feature preview.
4. **RUL Prediction** — select a model and history source, or upload a headered CSV with the exact 26 raw-history fields; show raw and display estimates with model/history context.
5. **Maintenance Planner** — pass selected predictions to paired EDD and genetic-algorithm schedules under explicit illustrative duration, weight, team, buffer, and horizon assumptions.
6. **Model Laboratory** — show saved development studies separately from the official FD001/test results and their stated provenance.

The freeze covers these screens, navigation, data/model selection, prediction, upload validation, planner handoff, saved-result views, and their current labels and limitations. No new product features or cosmetic redesigns enter the release candidate. A narrowly scoped correction may proceed only to fix a demonstrated correctness, security, reproducibility, or blocking installation defect; record its evidence and rerun affected checks. Anything else is deferred until after the release review.

This is a source-level feature freeze on `dev`, not proof that Streamlit Cloud is serving this commit. The remote revision remains **NOT VERIFIED** in this workspace: an anonymous request redirected to Streamlit sign-in, and the response exposed no deployed SHA. See [P24-T2 evidence](p24_t2_screenshots_known_issues.md).

## Remaining release checklist

| ID | Item and current evidence | Owner | Status / next action |
| --- | --- | --- | --- |
| R1 | Identify whether the current P24 `dev` head is deployed. P23-A3 recorded a signed-in remote parity check and Abdullah's report that deployment matched `ffe8a4555ba278ce4dd061bc14abdcbc40239873`; the host did not expose its SHA. P24-T2's anonymous request redirected to sign-in, and `dev` has advanced since that earlier check. | Taha (deployment) with Abdullah (parity) | **Open for the current P24 revision; earlier P23 parity is verified/reported.** Before final live demo, confirm the current hosted revision and capture the remote flow if access permits. Do not represent P24 local screenshots as remote verification. |
| R2 | Make the intended raw FD001 source histories available to a clean reproduction environment. P24-T2 found `data/raw/*.txt` is not tracked; the packaged `data/demo/` subset is separate and tracked. | Abdullah (data/reproduction) | **Open, documented limitation.** Confirm the required source files/checksums and provisioning route for the final reproduction; never commit raw data without the data-policy approval. |
| R3 | Decide whether to address the nonfatal observed-cycle slider session-state warning. The P24-T1 browser flow rendered and completed; no prediction error was observed. | Taha (UI) | **Deferred cosmetic/runtime warning.** Keep frozen unless it blocks the demo or causes incorrect state; if changed, rerun navigation and prediction checks. |
| R4 | Preserve and, if a supported explanation becomes available, investigate the tiny final-MLP numerical drift. P24-S2 found up to `0.0000152587890625` cycles across 43/100 rows on the first rerun; the second current-environment run was exact and bundle hashes matched. Cause remains unproven. | Shaheer (model reproducibility) | **Open, documented; no retuning.** Keep the measured difference in the audit trail. Only change model/evaluation code with new reproducible evidence and review. |
| R5 | Keep scientific and planner limitations visible in the final demonstration and report: FD001 is simulated; raw model estimates are not fault diagnosis; schedule durations/weights/capacity/buffer/horizon are assumptions. | All three, coordinated by Taha (demo) | **Required release check.** Confirm wording remains visible in final report and demo; do not turn estimates into maintenance claims. |

An item marked open is not a claim that a software failure has been observed. R1/R2 describe verification/provisioning gaps; R3 is explicitly deferred; R4 is an unresolved reproducibility explanation; R5 is a required communication check. No new issue or due date is invented here. Revisit owners/status in the P25 release materials and P26 package audit.

## Evidence supporting the freeze

- [P24-T1 local browser flow](p24_t1_browser_demo_evidence.md): all six screens were navigated in a fresh local Chrome context; upload prediction and planner handoff rendered; local Streamlit health returned HTTP 200/`ok`.
- [P24-T2 screenshots and known issues](p24_t2_screenshots_known_issues.md): five additional local screenshots cover Engine Explorer, Model Laboratory, invalid upload, P22 studies, and held-out results.
- [P24-S1 result trace](p24_s1_result_trace.md): traces reported metrics to source reports, datasets/splits, and frozen model identities.
- [P24-S2 frozen evaluation](p24_s2_frozen_evaluation.md): records the no-retuning rerun and its unexplained, very small MLP numerical difference.
- [P24-S3 claims and assumptions audit](p24_s3_claims_and_assumptions_audit.md): audits published quantitative claims and scheduler assumptions.

The interface examples use simulated FD001 data and saved model artifacts. They are not real-fleet performance, safety, or maintenance evidence. The official FD001/test result view is a saved evaluation display, not a live retraining feature.
