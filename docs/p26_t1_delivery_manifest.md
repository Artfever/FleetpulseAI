# P26-T1 final delivery map

Issue: [P26-T1 #107](https://github.com/Artfever/AI_PROJECT/issues/107)

## Submission-facing materials

| Item | Repository location | Evidence or status |
| --- | --- | --- |
| Final report | [`docs/report/p25_final_report.md`](report/p25_final_report.md) | Assembles the problem, user journey, system design, results, schedule simulation, and limitations; detailed claims link to source chapters and artifacts |
| Editable slide source | [`docs/slides/p25_final_deck.md`](slides/p25_final_deck.md) | Version columns were added in P26-S2; exact model values are traced in [`p26_s2_claim_consistency.md`](p26_s2_claim_consistency.md) |
| Exported slide PDF | [`output/pdf/p25_final_deck.pdf`](../output/pdf/p25_final_deck.pdf) | Existing 10-page export; P26-T2 must regenerate and inspect it after the P26-S2 version-column changes |
| Project README | [`README.md`](../README.md) | Setup, local app launch, training/evaluation reproduction, source layout, report/slides, and this delivery map |
| Demo procedure | [`docs/p25_live_demo_runbook.md`](p25_live_demo_runbook.md) | Timed six-screen click path, upload helper, recovery steps, and technical Q&A |
| Public application | [https://fleetpulseai.streamlit.app/](https://fleetpulseai.streamlit.app/) | Abdullah reported checking that the public app loaded on 2026-11-21. This workspace did not independently identify the deployed commit SHA; this is owner-reported live URL evidence, not a verified revision identifier. |

## Claim trace

- Final test scores and the five frozen model versions: [`p26_s1_ai_evidence_inventory.md`](p26_s1_ai_evidence_inventory.md), [`p26_s2_claim_consistency.md`](p26_s2_claim_consistency.md), and `artifacts/final/p13/final_test_results.json`.
- Data source, expected dataset checksums, and raw-data access instructions: `docs/fd001_provenance.md` and `README.md`.
- Figures and scenario comparisons: `docs/p22_report_results.md`, tracked SVG/JSON evidence in `docs/figures/` and `docs/p22_*.json`.
- Product click-path screenshots are prior local browser evidence in `docs/images/p24_t1/` and `docs/images/p24_t2/`; they are not represented as screenshots from a currently identified cloud SHA.
- Scheduling duration, priority, team count, buffer, horizon, and genetic-search settings are declared simulation assumptions. FD001 is simulated data; RUL and errors are cycles, not accuracy percentages or maintenance guarantees.

## Review boundary

This map prepares the materials for final review. It does not record a new
human scientific/layout review, all-three-member sign-off, course-platform
submission, or a current cloud revision SHA. Those checks remain open until the
actual final package and refreshed PDF are ready.
