"""Generate frozen-model predictions for each FD001/test engine prefix.

This module deliberately reads sensor histories only.  Official RUL answers
are handled by a separate scoring step so predictions cannot accidentally be
conditioned on the labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.quality import validate_engine_history
from fleetpulse.features.config import FeatureConfig
from fleetpulse.features.engineering import engineer_features

PREDICTION_SCHEMA = "p13-official-test-predictions-v1"
TEST_PARTITION = "FD001/test"
DEFAULT_TEST_PATH = Path("data/raw/test_FD001.txt")
DEFAULT_OUTPUT_PATH = Path("artifacts/final/p13/official_test_predictions.json")


@dataclass(frozen=True)
class FinalPredictor:
    """One checksum-verified final model and its input contract."""

    name: str
    version: str
    feature_config: FeatureConfig
    predict: Callable[[np.ndarray], Sequence[float]]


def load_final_predictors() -> dict[str, FinalPredictor]:
    """Load every P12 final artifact through its checksum-verifying loader."""
    from fleetpulse.models.final_mlp_bundle import load_final_mlp_bundle, predict_final_mlp
    from fleetpulse.models.final_tree_bundle import (
        load_final_tree_bundle,
        predict_final_tree_bundle,
    )
    from fleetpulse.models.registry import FINAL_MODEL_REGISTRY

    predictors: dict[str, FinalPredictor] = {}
    for name, spec in FINAL_MODEL_REGISTRY.items():
        if name == "dense-rul-mlp-final":
            bundle = load_final_mlp_bundle(spec.manifest_path)
            config_document = bundle.payload["training"]["feature_config"]
            config = FeatureConfig(
                channels=tuple(config_document["channels"]),
                window_cycles=config_document["window_cycles"],
            )
            predict = lambda matrix, loaded=bundle: predict_final_mlp(loaded, matrix)
            version = bundle.payload["model"]["version"]
        else:
            bundle = load_final_tree_bundle(spec.manifest_path)
            predict = lambda matrix, loaded=bundle: predict_final_tree_bundle(loaded, matrix)
            config = bundle.feature_config
            version = bundle.version
        predictors[name] = FinalPredictor(name, version, config, predict)
    if not predictors:
        raise ValueError("No P12 final models are registered for official evaluation.")
    return predictors


def build_prediction_report(
    records: Iterable[Mapping[str, object]],
    predictors: Mapping[str, FinalPredictor],
    *,
    source_sha256: str,
    source_name: str,
    expected_engine_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Predict once per engine, using only that engine's observed test history.

    The function accepts no label values or RUL-answer path.  `expected_engine_ids`
    lets the official-data entry point pin the complete FD001 test identity set.
    """
    if not predictors:
        raise ValueError("At least one final predictor is required.")
    histories: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    rows = tuple(records)
    if not rows:
        raise ValueError("FD001/test history is empty.")
    quality = validate_engine_history(rows)
    if not quality.is_valid:
        codes = ", ".join(sorted({finding.code for finding in quality.errors}))
        raise ValueError(f"Cannot evaluate invalid FD001/test history: {codes}.")
    for row in rows:
        unit_id = row["unit_id"]
        if isinstance(unit_id, bool) or not isinstance(unit_id, int) or unit_id < 1:
            raise ValueError("FD001/test engine IDs must be positive integers.")
        histories[unit_id].append(row)

    engine_ids = sorted(histories)
    if expected_engine_ids is not None and engine_ids != list(expected_engine_ids):
        raise ValueError(
            "FD001/test engine IDs differ from the expected official identity sequence."
        )
    for predictor in predictors.values():
        if not isinstance(predictor, FinalPredictor):
            raise TypeError("predictors must contain FinalPredictor values.")

    engine_rows: list[dict[str, Any]] = []
    model_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in predictors
    }
    model_versions = {name: predictor.version for name, predictor in predictors.items()}

    for engine_id in engine_ids:
        history = histories[engine_id]
        last_cycle = max(int(row["cycle"]) for row in history)
        vectors = {
            name: engineer_features(
                history,
                dataset_partition=TEST_PARTITION,
                unit_id=engine_id,
                selected_cycle=last_cycle,
                config=predictor.feature_config,
            )
            for name, predictor in predictors.items()
        }
        engine_rows.append(
            {
                "unit_id": engine_id,
                "observed_cycle": last_cycle,
                "observed_row_count": len(history),
                "window_row_count": vectors[next(iter(vectors))].window_row_count,
            }
        )
        for name, predictor in predictors.items():
            vector = vectors[name]
            matrix = np.asarray([vector.values], dtype=np.float64)
            raw_values = tuple(predictor.predict(matrix))
            if len(raw_values) != 1:
                raise ValueError(f"{name} must return exactly one prediction per engine.")
            raw_prediction = float(raw_values[0])
            if not math.isfinite(raw_prediction):
                raise FloatingPointError(f"{name} returned a non-finite prediction.")
            model_results[name].append(
                {
                    "unit_id": engine_id,
                    "observed_cycle": last_cycle,
                    "raw_rul_cycles": raw_prediction,
                }
            )

    for name, predictions in model_results.items():
        prediction_ids = [item["unit_id"] for item in predictions]
        if prediction_ids != engine_ids or len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError(f"{name} did not produce exactly one row per test engine.")

    return {
        "schema_version": PREDICTION_SCHEMA,
        "dataset_partition": TEST_PARTITION,
        "source_name": source_name,
        "source_sha256": source_sha256,
        "prediction_policy": "one prediction per engine at its greatest observed test cycle",
        "feature_policy": "past-only-v1; 30-cycle window; missing cycles are not filled",
        "labels_read": False,
        "models": [
            {
                "name": name,
                "version": model_versions[name],
                "prediction_count": len(model_results[name]),
                "predictions": model_results[name],
            }
            for name in predictors
        ],
        "engine_count": len(engine_ids),
        "engines": engine_rows,
        "quality_warnings": sorted({finding.code for finding in quality.warnings}),
    }


def _write_immutable_report(report: Mapping[str, Any], output_path: str | Path) -> str:
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    document = json.dumps(
        {"payload": report, "sha256": payload_hash},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != document:
            raise FileExistsError(
                f"Refusing to overwrite different predictions at {output}; use a new versioned path."
            )
        return payload_hash
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8", newline="\n")
    temporary.replace(output)
    return payload_hash


def run_official_test_predictions(
    test_path: str | Path = DEFAULT_TEST_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[dict[str, Any], str]:
    """Load test sensor histories, generate final predictions, and save locally."""
    source = Path(test_path)
    records = load_engine_history(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    predictors = load_final_predictors()
    report = build_prediction_report(
        records,
        predictors,
        source_sha256=source_hash,
        source_name=source.name,
        expected_engine_ids=tuple(range(1, 101)),
    )
    digest = _write_immutable_report(report, output_path)
    return report, digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions from frozen P12 bundles using FD001/test histories only."
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    report, digest = run_official_test_predictions(arguments.test_data, arguments.output)
    print(f"dataset_partition: {report['dataset_partition']}")
    print(f"engine_count: {report['engine_count']}")
    print(f"labels_read: {report['labels_read']}")
    for model in report["models"]:
        values = [row["raw_rul_cycles"] for row in model["predictions"]]
        print(
            f"{model['name']} version={model['version']} predictions={len(values)} "
            f"range=[{min(values):.8f}, {max(values):.8f}] cycles"
        )
    print(f"report: {arguments.output}")
    print(f"payload_sha256: {digest}")


if __name__ == "__main__":
    main()
