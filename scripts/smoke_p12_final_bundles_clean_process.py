"""Run identical-history inference for every final P12 bundle in fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.split import TRAIN_PARTITION
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY

SOURCE_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
REPORT_PATH = PROJECT_ROOT / "artifacts/final/p12_clean_process_smoke.json"
REFERENCE_ENGINE = 4
REFERENCE_CYCLE = 164
MODEL_NAMES = tuple(FINAL_MODEL_REGISTRY)
_SHA256_LENGTH = 64


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _worker(model_name: str, history_path: Path) -> None:
    request_bytes = history_path.read_bytes()
    request = json.loads(request_bytes)
    history_sha256 = _sha256_bytes(request_bytes)
    spec = FINAL_MODEL_REGISTRY[model_name]
    manifest_path = PROJECT_ROOT / spec.manifest_path
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise TypeError(f"{model_name} manifest has no payload.")
    if document.get("sha256") != _sha256_bytes(_canonical(payload)):
        raise ValueError(f"{model_name} manifest checksum mismatch.")

    if model_name == "dense-rul-mlp-final":
        from fleetpulse.features.config import FeatureConfig
        from fleetpulse.features.engineering import engineer_features
        from fleetpulse.models.final_mlp_bundle import load_final_mlp_bundle, predict_final_mlp

        training = payload["training"]
        config_doc = training["feature_config"]
        config = FeatureConfig(
            channels=tuple(config_doc["channels"]), window_cycles=config_doc["window_cycles"]
        )
        bundle = load_final_mlp_bundle(manifest_path)
        feature_names = tuple(training["feature_names"])
        prediction_fn = lambda matrix: predict_final_mlp(bundle, matrix)
        weights_sha256 = payload["model"]["weights_sha256"]
        artifact_role = payload["artifact_role"]
        version = payload["model"]["version"]
    else:
        from fleetpulse.features.engineering import engineer_features
        from fleetpulse.models.final_tree_bundle import (
            load_final_tree_bundle,
            predict_final_tree_bundle,
        )

        bundle = load_final_tree_bundle(manifest_path)
        config = bundle.feature_config
        feature_names = bundle.feature_names
        prediction_fn = lambda matrix: predict_final_tree_bundle(bundle, matrix)
        model_info = payload["model"]
        weights_sha256 = model_info.get("weights_sha256")
        artifact_role = payload["artifact_role"]
        version = bundle.version

    if (
        document["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest()
        or version != spec.version
        or artifact_role != spec.artifact_role
        or config.version != spec.feature_config_version
        or config.window_cycles != spec.feature_window_cycles
        or len(feature_names) != spec.feature_count
        or tuple(feature_names) != config.output_schema
    ):
        raise ValueError(f"{model_name} bundle metadata differs from the final registry.")

    vector = engineer_features(
        request["history"],
        dataset_partition=request["dataset_partition"],
        unit_id=request["unit_id"],
        selected_cycle=request["selected_cycle"],
        config=config,
    )
    if vector.observed_row_count != request["observed_row_count"]:
        raise ValueError("The clean process did not receive the declared full input history.")
    matrix = np.asarray([vector.values], dtype=np.float64)
    prediction = prediction_fn(matrix)[0]
    if not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
        raise FloatingPointError(f"{model_name} produced a non-finite smoke prediction.")

    result = {
        "model_name": spec.name,
        "model_version": version,
        "stage": spec.stage,
        "artifact_role": artifact_role,
        "process_id": os.getpid(),
        "input_history_sha256": history_sha256,
        "dataset_partition": vector.dataset_partition,
        "unit_id": vector.unit_id,
        "observed_cycle": vector.selected_cycle,
        "observed_history_rows": vector.observed_row_count,
        "feature_window_rows": vector.window_row_count,
        "feature_config_version": config.version,
        "feature_window_cycles": config.window_cycles,
        "feature_count": len(feature_names),
        "manifest_payload_sha256": document["sha256"],
        "weights_sha256": weights_sha256,
        "raw_prediction_cycles": float(prediction),
        "training_or_fit_performed": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def _same_history_request() -> dict[str, Any]:
    histories = load_engine_history(SOURCE_PATH)
    rows = [
        row
        for row in histories
        if int(row["unit_id"]) == REFERENCE_ENGINE and int(row["cycle"]) <= REFERENCE_CYCLE
    ]
    if not rows or int(rows[-1]["cycle"]) != REFERENCE_CYCLE:
        raise ValueError("Reference FD001/train engine history does not reach cycle 164.")
    return {
        "dataset_partition": TRAIN_PARTITION,
        "unit_id": REFERENCE_ENGINE,
        "selected_cycle": REFERENCE_CYCLE,
        "observed_row_count": len(rows),
        "history": rows,
    }


def _run_parent() -> None:
    request_bytes = _canonical(_same_history_request())
    history_sha256 = _sha256_bytes(request_bytes)
    with tempfile.TemporaryDirectory(prefix="fleetpulse-p12-clean-process-") as temp_dir:
        history_path = Path(temp_dir) / "shared_reference_history.json"
        history_path.write_bytes(request_bytes)
        results = []
        for model_name in MODEL_NAMES:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    model_name,
                    "--history-file",
                    str(history_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Fresh-process smoke failed for {model_name} "
                    f"(exit={completed.returncode}): {completed.stderr.strip()}"
                )
            result = json.loads(completed.stdout)
            if result["input_history_sha256"] != history_sha256:
                raise ValueError(f"{model_name} did not receive the identical input history.")
            if result["process_id"] == os.getpid() or result[
                "training_or_fit_performed"
            ]:
                raise ValueError(f"{model_name} did not run as load-only fresh-process inference.")
            results.append(result)

    if len(results) != len(MODEL_NAMES) or len({item["input_history_sha256"] for item in results}) != 1:
        raise ValueError("The clean-process report is missing models or shared input identity.")
    report = {
        "schema_version": "p12-final-bundle-clean-process-smoke-v1",
        "source_partition": TRAIN_PARTITION,
        "training_source_sha256": _sha256_bytes(SOURCE_PATH.read_bytes()),
        "reference_engine": REFERENCE_ENGINE,
        "reference_cycle": REFERENCE_CYCLE,
        "input_history_sha256": history_sha256,
        "models": results,
        "all_models_used_identical_history": True,
        "all_models_loaded_in_separate_processes": True,
        "training_or_fit_performed": False,
        "official_test_data_opened": False,
        "official_test_labels_used": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"input_history_sha256: {history_sha256}")
    for item in results:
        print(
            f"{item['model_name']}: version={item['model_version']} "
            f"pid={item['process_id']} prediction={item['raw_prediction_cycles']:.8f} "
            f"cycles fresh_process=yes fit=no"
        )
    print(f"report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=MODEL_NAMES)
    parser.add_argument("--history-file", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.history_file is None:
            parser.error("--worker requires --history-file")
        _worker(args.worker, args.history_file)
    elif args.history_file is not None:
        parser.error("--history-file is only valid with --worker")
    else:
        _run_parent()


if __name__ == "__main__":
    main()
