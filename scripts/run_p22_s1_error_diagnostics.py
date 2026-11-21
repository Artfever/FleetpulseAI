"""Reproduce P22-S1 diagnostics from verified P13 FD001/test pairings."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAIRINGS_PATH = ROOT / "artifacts/final/p13/official_test_pairings.json"
PREDICTIONS_PATH = ROOT / "artifacts/final/p13/official_test_predictions.json"
RESULTS_PATH = ROOT / "artifacts/final/p13/final_test_results.json"
METRIC_AUDIT_PATH = ROOT / "artifacts/final/p13/a2_independent_metric_audit.json"
BUNDLE_AUDIT_PATH = ROOT / "artifacts/final/p13/a3_frozen_bundle_audit.json"
TEST_PATH = ROOT / "data/raw/test_FD001.txt"
RUL_PATH = ROOT / "data/raw/RUL_FD001.txt"
EXPECTED_TEST_SHA256 = "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
EXPECTED_RUL_SHA256 = "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
EXPECTED_MODELS = {
    "decision-tree-rul": "decision-tree-rul-v1",
    "random-forest-rul": "random-forest-rul-v1",
    "xgboost-rul": "xgboost-rul-v1",
    "constant-rul": "constant-rul-v1",
    "dense-rul-mlp-final": "dense-rul-mlp-final-v1",
}
RUL_BINS = (
    ("low", 0, 30),
    ("medium", 31, 100),
    ("high", 101, None),
)
MODEL_COLORS = ("#2563eb", "#0f766e", "#ea580c", "#9333ea", "#475569")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_verified_payload(path: Path) -> tuple[dict[str, Any], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("sha256") != _canonical_sha256(payload):
        raise ValueError(f"Invalid payload checksum or shape: {path}")
    return payload, document["sha256"]


def rul_bin(value: int) -> str:
    """Assign integer true RUL to predeclared inclusive analytical bins."""
    if type(value) is not int or value < 0:
        raise ValueError("true RUL must be a nonnegative integer number of cycles")
    if value <= 30:
        return "low"
    if value <= 100:
        return "medium"
    return "high"


def summarize_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute global and binned residual metrics for one model's paired rows."""
    if not rows:
        raise ValueError("model rows cannot be empty")
    residuals_by_bin: dict[str, list[float]] = {name: [] for name, _, _ in RUL_BINS}
    all_residuals = []
    for row in rows:
        prediction = row.get("raw_prediction_cycles")
        truth = row.get("true_rul_cycles")
        if (
            type(prediction) not in (int, float)
            or not math.isfinite(prediction)
            or type(truth) is not int
            or truth < 0
        ):
            raise ValueError("model row contains an invalid prediction or true RUL")
        residual = float(prediction) - truth
        all_residuals.append(residual)
        residuals_by_bin[rul_bin(truth)].append(residual)

    def metrics(residuals: Sequence[float]) -> dict[str, float | int]:
        return {
            "count": len(residuals),
            "mae_cycles": sum(abs(value) for value in residuals) / len(residuals),
            "rmse_cycles": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
            "mean_residual_cycles": sum(residuals) / len(residuals),
        }

    return {
        "sample_count": len(rows),
        "overall": metrics(all_residuals),
        "by_true_rul_bin": {
            name: metrics(residuals) if residuals else {"count": 0, "mae_cycles": None,
                                                          "rmse_cycles": None, "mean_residual_cycles": None}
            for name, residuals in residuals_by_bin.items()
        },
    }


def _check_sources() -> dict[str, str]:
    actual = {
        "test_history_sha256": _file_sha256(TEST_PATH),
        "answer_file_sha256": _file_sha256(RUL_PATH),
    }
    if actual["test_history_sha256"] != EXPECTED_TEST_SHA256:
        raise ValueError("FD001/test history hash differs from the frozen P13 source.")
    if actual["answer_file_sha256"] != EXPECTED_RUL_SHA256:
        raise ValueError("FD001/test answer hash differs from the frozen P13 source.")
    return actual


def load_verified_diagnostics() -> dict[str, Any]:
    """Verify P13 pairing, score, and bundle records before calculating bins."""
    source_hashes = _check_sources()
    pairings, pairings_sha = _read_verified_payload(PAIRINGS_PATH)
    predictions, predictions_sha = _read_verified_payload(PREDICTIONS_PATH)
    results, results_sha = _read_verified_payload(RESULTS_PATH)
    metric_audit, metric_audit_sha = _read_verified_payload(METRIC_AUDIT_PATH)
    bundle_audit, bundle_audit_sha = _read_verified_payload(BUNDLE_AUDIT_PATH)

    if pairings.get("schema_version") != "p13-official-test-pairings-v1":
        raise ValueError("Unexpected P13 pairing schema.")
    if pairings.get("dataset_partition") != "FD001/test" or pairings.get("engine_count") != 100:
        raise ValueError("Pairing artifact is not the frozen 100-engine FD001/test set.")
    if pairings.get("scores_computed") is not False:
        raise ValueError("P13 pairing file must remain a pairing record, not a new score run.")
    if pairings.get("test_history_sha256") != source_hashes["test_history_sha256"]:
        raise ValueError("Pairing artifact test source hash does not match the local source.")
    if pairings.get("answer_file_sha256") != source_hashes["answer_file_sha256"]:
        raise ValueError("Pairing artifact label hash does not match the local answer file.")
    if pairings.get("prediction_payload_sha256") != predictions_sha:
        raise ValueError("Pairing artifact does not match the saved P13 prediction payload.")
    if results.get("results_use") != "final reporting only; not used to select, tune, or refit models":
        raise ValueError("P13 results are not marked as reporting-only.")
    if metric_audit.get("metrics_recomputed") is not True:
        raise ValueError("P13 independent metric audit is not complete.")
    if bundle_audit.get("evaluated_models_match_frozen_inventory") is not True:
        raise ValueError("P13 models do not match the frozen bundle inventory.")
    if bundle_audit.get("test_artifact_used_for_training_or_tuning") is not False:
        raise ValueError("P13 audit does not certify test separation.")
    if (
        metric_audit.get("schema_version") != "p13-a2-independent-metric-audit-v1"
        or bundle_audit.get("schema_version") != "p13-a3-frozen-bundle-audit-v1"
    ):
        raise ValueError("P13 audit schema version is unexpected.")

    answers = [int(line.strip()) for line in RUL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(answers) != 100 or predictions.get("engine_count") != 100:
        raise ValueError("Expected exactly 100 FD001/test labels and prediction histories.")
    expected_ids = set(range(1, 101))
    model_rows = {}
    for model in pairings.get("models", []):
        name, version, rows = model.get("name"), model.get("version"), model.get("rows")
        if name not in EXPECTED_MODELS or version != EXPECTED_MODELS[name]:
            raise ValueError(f"Unexpected frozen model identity: {name} {version}")
        if name in model_rows or not isinstance(rows, list) or len(rows) != 100:
            raise ValueError(f"Duplicate or incomplete P13 pairing rows: {name}")
        by_id = {row.get("unit_id"): row for row in rows}
        if set(by_id) != expected_ids:
            raise ValueError(f"Missing or duplicate engine IDs for {name}")
        paired_output = next(item for item in results["models"] if item["name"] == name)
        if paired_output.get("version") != version or paired_output.get("sample_count") != 100:
            raise ValueError(f"P13 score table identity mismatch for {name}")
        score_rows = {row.get("unit_id"): row for row in paired_output.get("rows", [])}
        if set(score_rows) != expected_ids:
            raise ValueError(f"P13 scored rows are incomplete for {name}")
        for unit_id in sorted(expected_ids):
            row = by_id[unit_id]
            result_row = score_rows[unit_id]
            if row.get("answer_line") != unit_id or row.get("true_rul_cycles") != answers[unit_id - 1]:
                raise ValueError(f"Answer-vector alignment mismatch for {name} engine {unit_id}")
            if (
                row.get("raw_prediction_cycles") != result_row.get("raw_prediction_cycles")
                or row.get("true_rul_cycles") != result_row.get("true_rul_cycles")
                or result_row.get("residual_prediction_minus_true_cycles")
                != result_row.get("raw_prediction_cycles") - result_row.get("true_rul_cycles")
            ):
                raise ValueError(f"Pairing/result mismatch for {name} engine {unit_id}")
        computed = summarize_model([by_id[unit_id] for unit_id in sorted(expected_ids)])
        reported_metrics = paired_output.get("metrics", {})
        for metric_name in ("mae_cycles", "rmse_cycles"):
            if not math.isclose(
                float(computed["overall"][metric_name]),
                float(reported_metrics.get(metric_name, float("nan"))),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Recomputed {metric_name} differs from P13 for {name}")
        model_rows[name] = {
            "display_name": paired_output.get("display_name", name),
            "version": version,
            "rows": [by_id[unit_id] for unit_id in sorted(expected_ids)],
            "summary": computed,
        }

    if set(model_rows) != set(EXPECTED_MODELS):
        raise ValueError("P13 paired model inventory is incomplete.")
    return {
        "schema_version": "p22-s1-error-diagnostics-v1",
        "dataset_partition": "FD001/test",
        "evaluation_unit": "engine at greatest observed test cycle",
        "residual_definition": "raw_prediction_cycles - true_rul_cycles; positive means over-prediction",
        "rul_bins": [
            {"name": name, "minimum_inclusive_cycles": lower,
             "maximum_inclusive_cycles": upper}
            for name, lower, upper in RUL_BINS
        ],
        "source_hashes": source_hashes,
        "artifact_hashes": {
            "pairings_file_sha256": _file_sha256(PAIRINGS_PATH),
            "pairings_payload_sha256": pairings_sha,
            "predictions_file_sha256": _file_sha256(PREDICTIONS_PATH),
            "predictions_payload_sha256": predictions_sha,
            "final_results_file_sha256": _file_sha256(RESULTS_PATH),
            "final_results_payload_sha256": results_sha,
            "metric_audit_file_sha256": _file_sha256(METRIC_AUDIT_PATH),
            "metric_audit_payload_sha256": metric_audit_sha,
            "bundle_audit_file_sha256": _file_sha256(BUNDLE_AUDIT_PATH),
            "bundle_audit_payload_sha256": bundle_audit_sha,
        },
        "model_count": len(model_rows),
        "engine_count_per_model": 100,
        "models": {
            name: {
                "display_name": item["display_name"],
                "version": item["version"],
                "summary": item["summary"],
            }
            for name, item in model_rows.items()
        },
        "limits": [
            "Frozen P13 FD001/test results are descriptive only; these diagnostics did not select, tune, or refit models.",
            "NASA C-MAPSS is simulated data. Engine counts and cycle errors are not fleet or maintenance evidence.",
            "RUL-bin summaries describe these 100 engine-level test rows and are not individual prediction intervals.",
        ],
        "_plot_rows": model_rows,
    }


def _text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "middle") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial,sans-serif" font-size="{size}" fill="#172033">'
            f'{html.escape(text)}</text>')


def _scatter_svg(models: Mapping[str, Any], *, residual: bool) -> str:
    width, height, columns, rows = 1560, 900, 3, 2
    panel_w, panel_h = width / columns, height / rows
    x_min = 0.0
    true_rul_max = max(
        float(point["true_rul_cycles"])
        for model in models.values() for point in model["rows"]
    )
    x_max = max(25.0, math.ceil(true_rul_max / 25) * 25)
    if residual:
        all_residuals = [
            float(row["raw_prediction_cycles"]) - row["true_rul_cycles"]
            for model in models.values() for row in model["rows"]
        ]
        bound = math.ceil(max(abs(value) for value in all_residuals) / 25) * 25
        y_min, y_max = -bound, bound
        title = "FD001/test residuals by true RUL"
        y_label = "Residual: prediction − true RUL (cycles)"
    else:
        y_min, y_max = 0.0, 250.0
        title = "FD001/test actual versus predicted RUL"
        y_label = "Raw predicted RUL (cycles)"
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(width / 2, 34, title, size=23),
    ]
    for index, (name, model) in enumerate(models.items()):
        col, row_index = index % columns, index // columns
        left, top = col * panel_w, row_index * panel_h
        plot_left, plot_top = left + 72, top + 70
        plot_w, plot_h = panel_w - 110, panel_h - 140
        def sx(value, plot_left=plot_left, plot_w=plot_w):
            return plot_left + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value, plot_top=plot_top, plot_h=plot_h):
            return plot_top + (y_max - value) / (y_max - y_min) * plot_h
        pieces.append(_text(left + panel_w / 2, top + 58,
                            f"{model['display_name']} — {model['version']}", size=16))
        x_tick_step = 25 if x_max <= 150 else 50
        for tick in range(0, int(x_max) + 1, x_tick_step):
            pieces.append(f'<line x1="{sx(tick):.1f}" y1="{plot_top}" x2="{sx(tick):.1f}" y2="{plot_top + plot_h}" stroke="#e2e8f0"/>')
            pieces.append(_text(sx(tick), plot_top + plot_h + 20, str(tick), size=11))
        y_tick_step = 50
        y_tick_start = math.floor(y_min / y_tick_step) * y_tick_step
        y_tick_end = math.ceil(y_max / y_tick_step) * y_tick_step
        for tick in range(y_tick_start, y_tick_end + 1, y_tick_step):
            if y_min <= tick <= y_max:
                pieces.append(f'<line x1="{plot_left}" y1="{sy(tick):.1f}" x2="{plot_left + plot_w}" y2="{sy(tick):.1f}" stroke="#e2e8f0"/>')
                pieces.append(_text(plot_left - 12, sy(tick) + 4, str(tick), size=11, anchor="end"))
        if residual:
            pieces.append(f'<line x1="{plot_left}" y1="{sy(0):.1f}" x2="{plot_left + plot_w}" y2="{sy(0):.1f}" stroke="#0f172a" stroke-width="1.5"/>')
        else:
            line_end = min(x_max, y_max)
            pieces.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(line_end):.1f}" y2="{sy(line_end):.1f}" stroke="#0f172a" stroke-dasharray="6 4"/>')
        for point in model["rows"]:
            truth = float(point["true_rul_cycles"])
            prediction = float(point["raw_prediction_cycles"])
            y_value = prediction - truth if residual else prediction
            pieces.append(
                f'<circle cx="{sx(truth):.2f}" cy="{sy(y_value):.2f}" r="3.4" '
                f'fill="{MODEL_COLORS[index]}" fill-opacity="0.7">'
                f'<title>Engine {point["unit_id"]}; true RUL {truth:g}; raw prediction {prediction:.4f}; '
                f'residual {prediction - truth:.4f} cycles</title></circle>'
            )
        pieces.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#334155"/>')
        pieces.append(_text(plot_left + plot_w / 2, top + panel_h - 23,
                            "True RUL (cycles)", size=12))
        pieces.append(f'<text x="{left + 18}" y="{plot_top + plot_h / 2}" transform="rotate(-90 {left + 18} {plot_top + plot_h / 2})" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#172033">{html.escape(y_label)}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def _bins_svg(models: Mapping[str, Any]) -> str:
    width, height = 1560, 650
    left, top, plot_w, plot_h = 105, 90, 1380, 430
    bin_names = [name for name, _, _ in RUL_BINS]
    grouped = [
        result["mae_cycles"]
        for model_data in models.values()
        for result in model_data["summary"]["by_true_rul_bin"].values()
    ]
    max_mae = max(float(value or 0.0) for value in grouped)
    y_max = max(10, math.ceil(max_mae / 10) * 10)
    sy = lambda value: top + (y_max - value) / y_max * plot_h
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(width / 2, 36, "Mean absolute error by true RUL bin (FD001/test)", size=23),
    ]
    for tick in range(0, y_max + 1, max(1, y_max // 5)):
        pieces.append(f'<line x1="{left}" y1="{sy(tick):.1f}" x2="{left + plot_w}" y2="{sy(tick):.1f}" stroke="#e2e8f0"/>')
        pieces.append(_text(left - 14, sy(tick) + 4, str(tick), size=12, anchor="end"))
    band_w = plot_w / len(bin_names)
    group_w = band_w * 0.78
    bar_w = group_w / (len(models) * 1.25)
    for group_index, bin_name in enumerate(bin_names):
        x0 = left + group_index * band_w + (band_w - group_w) / 2
        for model_index, (name, model_data) in enumerate(models.items()):
            summary = model_data["summary"]["by_true_rul_bin"][bin_name]
            value = summary["mae_cycles"]
            count = summary["count"]
            x = x0 + model_index * bar_w * 1.25
            if value is None:
                pieces.append(_text(x + bar_w / 2, top + plot_h + 19, f"n={count}", size=10))
                continue
            y = sy(float(value))
            bar_h = top + plot_h - y
            pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{MODEL_COLORS[model_index]}"><title>{html.escape(model_data["display_name"])}; {bin_name} bin; count {count}; MAE {float(value):.3f} cycles</title></rect>')
            pieces.append(_text(x + bar_w / 2, y - 5, f"{float(value):.1f}", size=10))
            pieces.append(_text(x + bar_w / 2, top + plot_h + 17, f"n={count}", size=9))
        label = "Low (0–30)" if bin_name == "low" else ("Medium (31–100)" if bin_name == "medium" else "High (101+)")
        pieces.append(_text(left + group_index * band_w + band_w / 2, top + plot_h + 44, label, size=14))
    pieces.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#334155"/>')
    pieces.append(_text(30, top + plot_h / 2, "MAE (cycles)", size=13))
    legend_y = height - 24
    for index, model_data in enumerate(models.values()):
        x = 180 + index * 270
        pieces.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{MODEL_COLORS[index]}"/>')
        pieces.append(_text(x + 22, legend_y, model_data["display_name"], size=12, anchor="start"))
    pieces.append("</svg>")
    return "\n".join(pieces)


def run_diagnostics(output_dir: Path) -> dict[str, Any]:
    report = load_verified_diagnostics()
    models = report.pop("_plot_rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "p22_s1_error_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "p22_s1_actual_vs_predicted.svg").write_text(
        _scatter_svg(models, residual=False) + "\n", encoding="utf-8"
    )
    (output_dir / "p22_s1_residuals.svg").write_text(
        _scatter_svg(models, residual=True) + "\n", encoding="utf-8"
    )
    (output_dir / "p22_s1_rul_bins.svg").write_text(_bins_svg(models) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    report = run_diagnostics(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
