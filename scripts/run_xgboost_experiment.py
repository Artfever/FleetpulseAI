"""Run and record one fixed XGBoost experiment on audited FD001/train tables."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.train_xgboost import train_xgboost

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts" / "development"
    manifest_path = PROJECT_ROOT / "artifacts" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"]["sha256"] != TRAIN_SHA256:
        raise ValueError("Split manifest does not identify the reviewed FD001/train source.")

    table_paths = {
        role: artifact_dir / f"{role}_features.json" for role in ("fitting", "validation")
    }
    expected_manifest_sha256 = manifest_digest(manifest)
    tables = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_config=FeatureConfig(),
        )
        for role, path in table_paths.items()
    }

    # This is the sole fit: fixed defaults, no sweep, no early stopping, CPU only.
    result = train_xgboost(tables["fitting"], tables["validation"])
    report = {
        "experiment": "P09-T2 single bounded XGBoost development-validation run",
        "model_version": "xgboost-rul-v1",
        "source_partition": "FD001/train",
        "source_sha256": TRAIN_SHA256,
        "split_manifest_sha256": expected_manifest_sha256,
        "feature_config": {
            "version": tables["fitting"].config.version,
            "channels": list(tables["fitting"].config.channels),
            "window_cycles": tables["fitting"].config.window_cycles,
            "feature_count": len(tables["fitting"].feature_names),
        },
        "table_sha256": {role: sha256(path) for role, path in table_paths.items()},
        "fitting_rows": len(tables["fitting"].rows),
        "fitting_engines": len({row.vector.unit_id for row in tables["fitting"].rows}),
        "validation_engines": len(tables["validation"].rows),
        "random_seed": result.random_seed,
        "parameters": result.parameters,
        "fit_and_validation_prediction_seconds": result.runtime_seconds,
        "metrics_cycles": asdict(result.metrics),
        "predictions": [asdict(prediction) for prediction in result.predictions],
        "versions": {
            "python": platform.python_version(),
            "xgboost": version("xgboost"),
        },
        "scope_note": (
            "Held-out FD001/train development-validation engines only; no official test "
            "histories or RUL answers used. One fixed configuration; no tuning or refit."
        ),
    }
    output = PROJECT_ROOT / "reports" / "P09_T2_xgboost_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: report[key] for key in (
        "experiment", "model_version", "fitting_rows", "fitting_engines",
        "validation_engines", "random_seed", "parameters",
        "fit_and_validation_prediction_seconds", "metrics_cycles", "versions",
    )}, indent=2, sort_keys=True))
    print(f"saved_run_record: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
