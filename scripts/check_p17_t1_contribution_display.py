"""Exercise the RUL page's final-tree contribution display on the local bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, services, ui
from fleetpulse.data.loader import load_engine_history

PREFIX_PATH = PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
PAGE_PATH = PROJECT_ROOT / "app" / "modules" / "3_RUL_Prediction.py"


def build_report() -> dict[str, object]:
    """Run the real Streamlit page and capture its displayed positive/negative lists."""
    records = load_engine_history(PREFIX_PATH)
    original_loader = ui.render_training_data
    original_explainer = services.explain_final_tree_history
    captured_explanation = {}
    ui.render_training_data = lambda: records

    def capture_explanation(*args, **kwargs):
        result = original_explainer(*args, **kwargs)
        captured_explanation["result"] = result
        return result

    services.explain_final_tree_history = capture_explanation
    try:
        page = AppTest.from_file(str(PAGE_PATH), default_timeout=60).run()
        if page.exception:
            raise RuntimeError(f"RUL page failed before explanation: {page.exception}")
        explain_button = next(
            (button for button in page.button if button.label == "Show feature contributions"),
            None,
        )
        if explain_button is None:
            raise RuntimeError("RUL page did not render the feature-contribution action.")
        explain_button.click().run()
        if page.exception:
            raise RuntimeError(f"RUL page explanation failed: {page.exception}")
        if page.selectbox[2].value != 4 or page.select_slider[0].value != 164:
            raise RuntimeError("The app did not retain the intended packaged engine/cutoff.")
        if any("Final Decision Tree explanation is unavailable" in item.value for item in page.info):
            raise RuntimeError("Local final-model explanation unavailable; check ignored bundles/data.")
        labels = [metric.label for metric in page.metric]
        try:
            raw_metric = page.metric[labels.index("Final raw RUL estimate")].value
            baseline_metric = page.metric[labels.index("SHAP reference baseline")].value
        except ValueError as error:
            raise RuntimeError("Final raw RUL or SHAP baseline was not displayed.") from error
        positive = page.dataframe[-2].value.to_dict("records")
        negative = page.dataframe[-1].value.to_dict("records")
        result = captured_explanation.get("result")
        if result is None:
            raise RuntimeError("The UI action did not call the frozen final-tree explanation service.")
        if not positive or not negative:
            raise RuntimeError("The local audit expected both positive and negative contributions.")
        if any(float(row["Contribution (cycles)"]) <= 0 for row in positive):
            raise RuntimeError("Positive contribution table contains a nonpositive value.")
        if any(float(row["Contribution (cycles)"]) >= 0 for row in negative):
            raise RuntimeError("Negative contribution table contains a nonnegative value.")
        return {
            "schema_version": "p17-t1-contribution-display-v1",
            "source_file": PREFIX_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "dataset_partition": "FD001/train",
            "unit_id": 4,
            "observed_cycle": 164,
            "development_prediction_visible_separately": True,
            "final_model_name": result.model_name,
            "final_model_version": result.model_version,
            "final_raw_rul_cycles": result.raw_rul_cycles,
            "raw_rul_metric_shown": raw_metric,
            "shap_baseline_cycles": result.base_value_cycles,
            "shap_baseline_metric_shown": baseline_metric,
            "additivity_error_cycles": result.additivity_error_cycles,
            "positive_contributions": positive,
            "negative_contributions": negative,
            "contribution_units": "cycles",
            "limitations": [
                "The explanation uses the frozen final Decision Tree, separate from the development prediction above.",
                "Signed contributions are relative to the documented FD001/train reference sample, not causal effects.",
                "This one packaged training prefix is not accuracy or fleet-wide evidence.",
            ],
        }
    finally:
        ui.render_training_data = original_loader
        services.explain_final_tree_history = original_explainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "P17_T1_contribution_display.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS: RUL page showed signed final-tree contributions for "
        f"engine {report['unit_id']} cycle {report['observed_cycle']} in {report['contribution_units']}"
    )
    print(f"UI evidence: {args.output.resolve()}")


if __name__ == "__main__":
    main()
