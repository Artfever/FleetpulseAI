# P25-T2: Heading, reference, diagram, and export audit

Issue: [P25-T2 #181](https://github.com/Artfever/FleetpulseAI/issues/181)  
Upstream assembly: [P25-T1 #104](https://github.com/Artfever/FleetpulseAI/issues/104), merged in PR [#540](https://github.com/Artfever/FleetpulseAI/pull/540) at `fcad261f50680b94dec8823273f9d95cd731f588`.

## Scope and inspection method

Audited the assembled [final report](report/p25_final_report.md), [editable Marp deck](slides/p25_final_deck.md), and linked A1/A2/S1/S2 source chapters. Checked heading hierarchy, tables, relative references and image targets. Rendered the Marp deck to the committed [PDF export](../output/pdf/p25_final_deck.pdf), extracted text and rendered every page to PNG for visual inspection. Rendered both Mermaid diagrams from the P25-A2 source with Mermaid CLI and inspected their readable layout.

The T1 report and deck have consistent section names, units, model rankings, data boundaries, and limitations. The deck uses rounded scores for legibility while the report keeps the full precision from the result caption/source table. Each display is traceable to the following source records.

## Claim and figure trace

| Displayed claim or figure | Report/deck location | Source evidence and check |
| --- | --- | --- |
| FD001/train and FD001/test cohort roles; separate test answers; 80/20 engine split; `RUL = L - t` | Final report sections 4 and 7; deck slides 4–5 | [P25-A1 data/features chapter](report/p25_a1_data_features_architecture.md), [development split](development_split.md), [evaluation protocol](evaluation_protocol.md), [P13 label-pairing report](p13_s2_test_label_pairing.md) |
| `past-only-v1`, 24 channels, four statistics, 96 features, 30 observed-cycle window | Final report section 4; deck slide 5 | [Feature contract](feature_contract.md), [feature example](feature_example.md), and [P25-A1](report/p25_a1_data_features_architecture.md) |
| Local upload output: engine 4/cycle 164, raw 34.196078 cycles, displayed 34.20, 30 feature rows; paired planner output and assumptions | Final report section 3; deck slides 6–7 | [P24-T1 browser evidence](p24_t1_browser_demo_evidence.md); image files under [`images/p24_t1/`](images/p24_t1/) were visually inspected. Evidence is local and simulated; hosted SHA remains unverified. |
| Five frozen FD001/test model metrics; MLP lowest MAE and XGBoost lowest RMSE | Final report section 5; deck slide 8 | [P25-S2 exact score table](report/p25_s2_evidence_captions.md), [P13 final report](p13_s3_final_test_results.md), [P13 independent audit](p13_a2_independent_metric_audit.md) |
| Final-tree SHAP attribution example and non-causal interpretation | Final report section 5; deck slide 9 | [P17-S1 output and method](p17_s1_final_tree_explanations.md), [P25-S1](report/p25_s1_model_evaluation_scheduling.md), and [P25-S2 captions](report/p25_s2_evidence_captions.md) |
| 18 scenario comparisons and 90 seeded schedule outcomes | Final report section 6; deck slide 9 | [P22-A1 fixed scenarios](p22_a1_fixed_fleet_scenarios.md), [P22-A2 seed comparisons](p22_a2_seed_comparison.md), and [P25-S2 captions](report/p25_s2_evidence_captions.md). Schedule inputs are explicitly labelled assumptions. |
| App screenshots and Model Laboratory test section | Final report section 3; deck slides 6–8 | [P24-T1](p24_t1_browser_demo_evidence.md) and [P24-T2](p24_t2_screenshots_known_issues.md). Screenshots show local app evidence, not a newly verified cloud revision. |

The [P25-A2 pipeline/reproduction chapter](report/p25_a2_pipeline_and_reproduction.md) retains Mermaid source for both process diagrams. The diagrams cover development versus held-out evaluation and the online prediction/planning path; their nodes and edges correspond to named modules and commands in that chapter.

## Export and layout results

- Marp CLI `4.5.1` exported the editable source to `output/pdf/p25_final_deck.pdf`.
- Export SHA-256: `ECE668893FB659C12B32B770A4567E15BC2B48D5A3B1D828F88AA93925B45075` (266,024 bytes).
- PDF has 10 landscape 16:9 pages. PyMuPDF `2.1` extracted text from every page and rendered all 10 pages to PNG; every page contained substantive text. I visually inspected all rendered pages. Headings, tables, text blocks, page numbers, and screenshot placements fit on-page; the final score table is fully visible.
- The first render exposed two screenshot placements that crowded content. The deck was adjusted and exported again; the final render was inspected after the adjustment. The home screenshot was removed from a text-heavy slide and the redundant Model Laboratory screenshot removed from the score-table slide. The final deck still includes the upload and planner flow screenshots, while its own score table presents all five scores in readable text.
- Mermaid CLI `12.2.0` rendered the two P25-A2 diagrams. The original left-to-right layouts measured 784×161 and 784×47 pixels, too wide/short for readable inline presentation. Changing both diagrams to top-to-bottom produced 784×669 and 517×1565 previews. Both revised diagrams were visually inspected; labels and arrows are readable. The report source now uses the revised top-to-bottom layout.
- Relative Markdown links/images in the final report and deck were checked; every referenced local path exists. `git diff --check` passes.

## Human checks still required

This audit checks source traceability and a locally rendered export. It is not a teammate approval, a group rehearsal, or proof that the exported slides suit the final presentation room. The issue owner will inspect the PDF and presentation content after this PR; issue #181 remains open until that human acceptance is recorded. No teammate review or rehearsal is claimed here.
