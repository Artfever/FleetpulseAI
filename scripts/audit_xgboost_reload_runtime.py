"""Compare in-memory and reloaded XGBoost on the identical validation batch."""

from __future__ import annotations

import hashlib
import json
import time
from importlib.metadata import version

import joblib

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.train_xgboost import train_xgboost
from fleetpulse.models.xgboost_artifact import verify_xgboost_artifact

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
WARMUP_BATCHES = 5
TIMED_BATCHES = 200


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts/development"
    manifest_path = artifact_dir / "xgboost_validation.json"
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest_document["payload"]
    split = json.loads((PROJECT_ROOT / "artifacts/split_manifest.json").read_text(encoding="utf-8"))
    if split["source"]["sha256"] != TRAIN_SHA256:
        raise ValueError("Split manifest differs from the reviewed FD001/train source.")
    if hashlib.sha256((PROJECT_ROOT / "data/raw/train_FD001.txt").read_bytes()).hexdigest() != TRAIN_SHA256:
        raise ValueError("FD001/train source hash differs from the reviewed source.")

    table_paths = {
        role: artifact_dir / f"{role}_features.json" for role in ("fitting", "validation")
    }
    datasets = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=manifest_digest(split),
            expected_config=FeatureConfig(),
        )
        for role, path in table_paths.items()
    }
    validation = datasets["validation"]
    verify_xgboost_artifact(manifest_path, validation)
    table_sha256 = json.loads(table_paths["validation"].read_text(encoding="utf-8"))["sha256"]
    if table_sha256 != payload["training"]["validation_dataset_sha256"]:
        raise ValueError("Validation table digest differs from saved candidate provenance.")

    parameters = payload["training"]["parameters"]
    fresh = train_xgboost(
        datasets["fitting"],
        validation,
        random_seed=payload["training"]["random_seed"],
        **{
            key: parameters[key]
            for key in (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
            )
        },
    )
    loaded = joblib.load(artifact_dir / payload["model"]["filename"])
    batch = validation.X
    memory_values = tuple(map(float, fresh.model.predict(batch)))
    reloaded_values = tuple(map(float, loaded.predict(batch)))
    saved_values = tuple(
        row["prediction_cycles"] for row in payload["evaluation"]["predictions"]
    )
    if memory_values != reloaded_values or reloaded_values != saved_values:
        raise ValueError("In-memory, reloaded, and saved predictions are not exact matches.")

    for _ in range(WARMUP_BATCHES):
        fresh.model.predict(batch)
        loaded.predict(batch)
    durations = {"in_memory": 0.0, "reloaded": 0.0}
    for _ in range(TIMED_BATCHES):
        started = time.perf_counter()
        fresh.model.predict(batch)
        durations["in_memory"] += time.perf_counter() - started
        started = time.perf_counter()
        loaded.predict(batch)
        durations["reloaded"] += time.perf_counter() - started

    report = {
        "model_name": payload["model"]["name"],
        "model_version": payload["model"]["version"],
        "model_sha256": payload["model"]["sha256"],
        "source_sha256": TRAIN_SHA256,
        "split_manifest_sha256": manifest_digest(split),
        "validation_table_sha256": table_sha256,
        "batch_engine_count": len(validation.rows),
        "batch_feature_count": len(validation.feature_names),
        "warmup_batches_per_model": WARMUP_BATCHES,
        "timed_batches_per_model": TIMED_BATCHES,
        "total_prediction_seconds": durations,
        "mean_seconds_per_20_engine_batch": {
            key: value / TIMED_BATCHES for key, value in durations.items()
        },
        "max_raw_prediction_difference_cycles": max(
            abs(left - right) for left, right in zip(memory_values, reloaded_values, strict=True)
        ),
        "xgboost_version": version("xgboost"),
        "scope": "FD001/train held-out development-validation batch only",
        "timing_note": (
            "Both estimators received the same ordered 20x96 feature matrix in alternating "
            "calls in this process. Timings cover predict() only, exclude fitting, feature "
            "engineering, disk loading and service overhead, and are machine-specific."
        ),
    }
    report_path = PROJECT_ROOT / "reports/P09_A2_xgboost_reload_runtime.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, sort_keys=True))
    print(f"saved_audit: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
