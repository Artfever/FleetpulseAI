"""Exercise the RUL page's raw frozen-model comparison on local bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, services, ui
from fleetpulse.data.loader import load_engine_history

PREFIX_PATH = PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
PAGE_PATH = PROJECT_ROOT / "app" / "pages" / "3_RUL_Prediction.py"


def build_report() -> dict[str, object]:
    """Run the real RUL page and capture its displayed frozen-model comparison."""
    records = load_engine_history(PREFIX_PATH)
    original_loader = ui.render_training_data
    original_comparator = services.compare_final_models
    captured = {}
    ui.render_training_data = lambda: records

    def capture_comparison(*args, **kwargs):
        result = original_comparator(*args, **kwargs)
        captured["result"] = result
        return result

    services.compare_final_models = capture_comparison
    try:
        page = AppTest.from_file(str(PAGE_PATH), default_timeout=90).run()
        if page.exception:
            raise RuntimeError(f"RUL page failed before comparison: {page.exception}")
        compare_button = next(
            (button for button in page.button if button.label == "Compare frozen final models"),
            None,
        )
        if compare_button is None:
            raise RuntimeError("The RUL page did not render the final-model comparison action.")
        compare_button.click().run()
        if page.exception:
            raise RuntimeError(f"RUL page comparison failed: {page.exception}")
        if page.selectbox[2].value != 4 or page.select_slider[0].value != 164:
            raise RuntimeError("The app did not retain the intended packaged engine/cutoff.")
        result = captured.get("result")
        if result is None:
            raise RuntimeError("The UI action did not call the frozen-model comparison service.")
        if result.dataset_partition != "FD001/train" or result.unit_id != 4:
            raise RuntimeError("Comparison used a different dataset-qualified engine.")
        if result.observed_cycle != 164 or result.failures:
            raise RuntimeError(
                f"Expected all local bundles at cycle 164; failures={result.failures!r}."
            )
        if len(result.predictions) != 5:
            raise RuntimeError(f"Expected five frozen model outputs, got {len(result.predictions)}.")
        labels = [metric.label for metric in page.metric]
        try:
            spread_metric = page.metric[labels.index("Raw model disagreement (max − min)")].value
        except ValueError as error:
            raise RuntimeError("The raw model disagreement metric was not displayed.") from error
        rows = page.dataframe[-1].value.to_dict("records")
        expected_rows = [
            {
                "Model": item.model_name,
                "Version": item.model_version,
                "Raw estimate (cycles)": item.raw_rul_cycles,
            }
            for item in result.predictions
        ]
        if rows != expected_rows:
            raise RuntimeError("The displayed comparison rows differ from the raw service outputs.")
        if abs(result.raw_spread_cycles - (result.raw_max_rul_cycles - result.raw_min_rul_cycles)) > 1e-12:
            raise RuntimeError("Displayed disagreement does not equal max(raw) − min(raw).")
        if any("not a confidence interval" in item.value for item in page.caption) is False:
            raise RuntimeError("The page does not disclose that spread is not a confidence interval.")
        return {
            "schema_version": "p17-t2-model-spread-display-v1",
            "input_file": PREFIX_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "input_file_sha256": hashlib.sha256(PREFIX_PATH.read_bytes()).hexdigest(),
            "dataset_partition": result.dataset_partition,
            "unit_id": result.unit_id,
            "observed_cycle": result.observed_cycle,
            "model_rows": rows,
            "raw_min_rul_cycles": result.raw_min_rul_cycles,
            "raw_max_rul_cycles": result.raw_max_rul_cycles,
            "raw_spread_cycles": result.raw_spread_cycles,
            "spread_metric_shown": spread_metric,
            "failure_count": len(result.failures),
            "spread_label": "model disagreement, not a confidence interval",
            "limitations": [
                "The comparison uses one packaged FD001/train prefix, not a complete fleet.",
                "Raw outputs and their range are not accuracy, probability, or confidence evidence.",
                "Final bundle files are local ignored artifacts and are not committed.",
            ],
        }
    finally:
        ui.render_training_data = original_loader
        services.compare_final_models = original_comparator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "P17_T2_model_spread_display.json",
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
        "PASS: RUL page displayed five frozen model estimates and raw spread "
        f"{report['raw_spread_cycles']:.6f} cycles for engine 4 cycle 164"
    )
    print(f"UI evidence: {args.output.resolve()}")


if __name__ == "__main__":
    main()
