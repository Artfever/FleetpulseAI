"""Compute and save final FD001/test errors and residual figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from math import fsum, sqrt
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from fleetpulse.evaluation.official_test_pairing import (
    DEFAULT_PAIRED_OUTPUT_PATH,
    PAIRING_SCHEMA,
)

DEFAULT_RESULTS_PATH = Path("artifacts/final/p13/final_test_results.json")
DEFAULT_RESIDUAL_FIGURE_PATH = Path("artifacts/final/p13/test_residual_histogram.html")
DEFAULT_PREDICTION_FIGURE_PATH = Path("artifacts/final/p13/test_prediction_errors.html")
RESULTS_SCHEMA = "p13-final-test-results-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def summarize_paired_results(
    paired_document: Mapping[str, Any],
    model_artifacts: Mapping[str, Mapping[str, Any]],
    *,
    paired_file_sha256: str,
) -> dict[str, Any]:
    """Build per-engine residuals and MAE/RMSE from an aligned pairing artifact."""
    paired_payload = paired_document.get("payload")
    if (
        not isinstance(paired_payload, Mapping)
        or paired_document.get("sha256") != _sha256(_canonical_json(paired_payload).encode())
        or paired_payload.get("schema_version") != PAIRING_SCHEMA
        or paired_payload.get("scores_computed") is not False
        or paired_payload.get("dataset_partition") != "FD001/test"
    ):
        raise ValueError("Input is not a verified, unscored FD001/test pairing artifact.")
    saved_models = paired_payload.get("models")
    if not isinstance(saved_models, list) or len(saved_models) != len(model_artifacts):
        raise ValueError("Paired model inventory and verified artifact inventory differ.")

    result_models: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for saved_model in saved_models:
        if not isinstance(saved_model, Mapping):
            raise TypeError("Paired artifact contains a malformed model.")
        name = saved_model.get("name")
        version = saved_model.get("version")
        metadata = model_artifacts.get(name)
        if name in seen_names or not isinstance(name, str) or not isinstance(metadata, Mapping):
            raise ValueError("Final model names must be unique and have verified artifacts.")
        seen_names.add(name)
        if metadata.get("version") != version:
            raise ValueError(f"Paired output model version differs from final artifact: {name}.")
        rows = saved_model.get("rows")
        count = saved_model.get("paired_count")
        if not isinstance(rows, list) or type(count) is not int or count != len(rows) or count < 1:
            raise ValueError(f"Model {name} has an incomplete paired result table.")

        engine_ids: list[int] = []
        residual_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError(f"Model {name} contains a malformed paired row.")
            unit_id = row.get("unit_id")
            observed_cycle = row.get("observed_cycle")
            answer_line = row.get("answer_line")
            true_rul = row.get("true_rul_cycles")
            prediction = row.get("raw_prediction_cycles")
            if (
                type(unit_id) is not int or unit_id < 1
                or type(observed_cycle) is not int or observed_cycle < 1
                or answer_line != unit_id
                or type(true_rul) is not int or true_rul < 0
                or type(prediction) not in (int, float) or not math.isfinite(prediction)
            ):
                raise ValueError(f"Model {name} contains invalid identity, label, or prediction data.")
            engine_ids.append(unit_id)
            residual = float(prediction) - true_rul
            residual_rows.append(
                {
                    "unit_id": unit_id,
                    "observed_cycle": observed_cycle,
                    "answer_line": answer_line,
                    "true_rul_cycles": true_rul,
                    "raw_prediction_cycles": float(prediction),
                    "residual_prediction_minus_true_cycles": residual,
                    "absolute_error_cycles": abs(residual),
                    "squared_error_cycles_squared": residual * residual,
                }
            )
        if engine_ids != sorted(set(engine_ids)):
            raise ValueError(f"Model {name} rows are not in unique ascending engine-ID order.")

        mae = fsum(row["absolute_error_cycles"] for row in residual_rows) / count
        rmse = sqrt(fsum(row["squared_error_cycles_squared"] for row in residual_rows) / count)
        if not math.isfinite(mae) or not math.isfinite(rmse):
            raise ValueError(f"Model {name} metrics are not finite.")
        result_models.append(
            {
                "name": name,
                "version": version,
                "display_name": metadata["display_name"],
                "artifact": dict(metadata),
                "sample_count": count,
                "metrics": {"mae_cycles": mae, "rmse_cycles": rmse},
                "residual_definition": "raw_prediction_cycles - true_rul_cycles",
                "rows": residual_rows,
            }
        )

    if seen_names != set(model_artifacts):
        raise ValueError("Final artifact metadata is missing or contains an unscored model.")
    return {
        "schema_version": RESULTS_SCHEMA,
        "dataset_partition": "FD001/test",
        "evaluation_unit": "one final observed prefix per engine",
        "sample_count_per_model": paired_payload["engine_count"],
        "prediction_source": {
            "pairing_payload_sha256": paired_document["sha256"],
            "pairing_file_sha256": paired_file_sha256,
            "test_history_sha256": paired_payload["test_history_sha256"],
            "rul_answer_file": paired_payload["answer_file"],
            "rul_answer_file_sha256": paired_payload["answer_file_sha256"],
        },
        "metric_definitions": {
            "mae_cycles": "mean(abs(raw prediction - true RUL))",
            "rmse_cycles": "sqrt(mean((raw prediction - true RUL)^2))",
            "predictions_clipped": False,
        },
        "results_use": "final reporting only; not used to select, tune, or refit models",
        "models": result_models,
    }


def load_verified_model_artifacts() -> dict[str, dict[str, Any]]:
    """Checksum-load frozen P12 bundles and collect their model/config hashes."""
    from fleetpulse.models.final_mlp_bundle import load_final_mlp_bundle
    from fleetpulse.models.final_tree_bundle import load_final_tree_bundle
    from fleetpulse.models.registry import FINAL_MODEL_REGISTRY

    artifacts: dict[str, dict[str, Any]] = {}
    for name, spec in FINAL_MODEL_REGISTRY.items():
        path = Path(spec.manifest_path)
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document.get("payload")
        if not isinstance(payload, Mapping) or document.get("sha256") != _sha256(
            _canonical_json(payload).encode()
        ):
            raise ValueError(f"Final model manifest checksum failed: {name}.")
        if name == "dense-rul-mlp-final":
            bundle = load_final_mlp_bundle(path)
            config = bundle.payload["training"]["feature_config"]
            model = bundle.payload["model"]
            verified_version = model["version"]
        else:
            bundle = load_final_tree_bundle(path)
            config = payload["training"]["feature_config"]
            model = payload["model"]
            verified_version = bundle.version
        if verified_version != spec.version:
            raise ValueError(f"Frozen final model version differs from registry: {name}.")
        weights_file = model.get("weights_file")
        weights_sha256 = model.get("weights_sha256")
        if weights_file is None:
            if weights_sha256 is not None:
                raise ValueError(f"Unexpected weights checksum for weightless model: {name}.")
        else:
            weights_path = path.parent / weights_file
            if _sha256(weights_path.read_bytes()) != weights_sha256:
                raise ValueError(f"Final model weights checksum failed: {name}.")
        artifacts[name] = {
            "version": spec.version,
            "display_name": spec.display_name,
            "manifest_path": spec.manifest_path,
            "manifest_file_sha256": _sha256(path.read_bytes()),
            "manifest_payload_sha256": document["sha256"],
            "weights_file": weights_file,
            "weights_sha256": weights_sha256,
            "feature_config": config,
            "feature_config_sha256": _sha256(_canonical_json(config).encode()),
            "feature_config_version": spec.feature_config_version,
            "feature_window_cycles": spec.feature_window_cycles,
            "feature_count": spec.feature_count,
            "artifact_role": spec.artifact_role,
        }
    return artifacts


def build_residual_figures(results: Mapping[str, Any]) -> tuple[str, str]:
    """Build self-contained interactive residual histogram and error scatter HTML."""
    models = results["models"]
    residual_figure = go.Figure()
    prediction_figure = go.Figure()
    for model in models:
        rows = model["rows"]
        residuals = [row["residual_prediction_minus_true_cycles"] for row in rows]
        true_rul = [row["true_rul_cycles"] for row in rows]
        residual_figure.add_trace(
            go.Histogram(x=residuals, name=model["display_name"], opacity=0.62, nbinsx=24)
        )
        prediction_figure.add_trace(
            go.Scatter(
                x=true_rul,
                y=residuals,
                mode="markers",
                name=model["display_name"],
                customdata=[row["unit_id"] for row in rows],
                hovertemplate="Engine %{customdata}<br>True RUL %{x:.2f} cycles"
                "<br>Residual %{y:.2f} cycles<extra>%{fullData.name}</extra>",
            )
        )
    residual_figure.update_layout(
        title="FD001/test residual distributions — frozen P12 models",
        xaxis_title="Residual: raw prediction − true RUL (cycles)",
        yaxis_title="Engine count",
        barmode="overlay",
    )
    prediction_figure.add_hline(y=0, line_dash="dash", line_color="#697586")
    prediction_figure.update_layout(
        title="FD001/test residual by true RUL — frozen P12 models",
        xaxis_title="True RUL at final observed cycle (cycles)",
        yaxis_title="Residual: raw prediction − true RUL (cycles)",
    )
    histogram_html = residual_figure.to_html(
        full_html=True, include_plotlyjs=True, div_id="p13-test-residual-histogram"
    )
    scatter_html = prediction_figure.to_html(
        full_html=True, include_plotlyjs=True, div_id="p13-test-residual-by-true-rul"
    )
    return histogram_html, scatter_html


def _write_immutable_json(payload: Mapping[str, Any], path: str | Path) -> str:
    canonical = _canonical_json(payload)
    digest = _sha256(canonical.encode())
    document = json.dumps(
        {"payload": payload, "sha256": digest}, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    _write_immutable_text(document, path)
    return digest


def _write_immutable_text(content: str, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"Refusing to overwrite different P13 results at {output}; use a new versioned path."
            )
        return
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(output)


def run_final_test_results(
    pairing_path: str | Path = DEFAULT_PAIRED_OUTPUT_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    residual_figure_path: str | Path = DEFAULT_RESIDUAL_FIGURE_PATH,
    prediction_figure_path: str | Path = DEFAULT_PREDICTION_FIGURE_PATH,
) -> tuple[dict[str, Any], str]:
    paired_file = Path(pairing_path)
    pairing_document = json.loads(paired_file.read_text(encoding="utf-8"))
    paired_file_sha256 = _sha256(paired_file.read_bytes())
    artifacts = load_verified_model_artifacts()
    results = summarize_paired_results(
        pairing_document,
        artifacts,
        paired_file_sha256=paired_file_sha256,
    )
    digest = _write_immutable_json(results, results_path)
    histogram_html, scatter_html = build_residual_figures(results)
    _write_immutable_text(histogram_html, residual_figure_path)
    _write_immutable_text(scatter_html, prediction_figure_path)
    return results, digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen FD001/test model results.")
    parser.add_argument("--pairings", type=Path, default=DEFAULT_PAIRED_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--residual-figure", type=Path, default=DEFAULT_RESIDUAL_FIGURE_PATH)
    parser.add_argument("--prediction-figure", type=Path, default=DEFAULT_PREDICTION_FIGURE_PATH)
    arguments = parser.parse_args()
    results, digest = run_final_test_results(
        arguments.pairings,
        arguments.output,
        arguments.residual_figure,
        arguments.prediction_figure,
    )
    print(f"dataset_partition: {results['dataset_partition']}")
    print(f"sample_count_per_model: {results['sample_count_per_model']}")
    for model in results["models"]:
        metrics = model["metrics"]
        print(
            f"{model['name']} version={model['version']} n={model['sample_count']} "
            f"MAE={metrics['mae_cycles']:.8f} RMSE={metrics['rmse_cycles']:.8f} cycles"
        )
    print(f"results: {arguments.output}")
    print(f"results_payload_sha256: {digest}")
    print(f"residual_figure: {arguments.residual_figure}")
    print(f"prediction_figure: {arguments.prediction_figure}")
    print("test_results_used_for_tuning: false")


if __name__ == "__main__":
    main()
