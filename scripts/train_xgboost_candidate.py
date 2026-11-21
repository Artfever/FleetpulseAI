"""Fit one fixed XGBoost candidate, persist its verified bundle, and replay service parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.train_xgboost import MODEL_VERSION, train_xgboost
from fleetpulse.models.xgboost_artifact import (
    save_xgboost_artifact,
    verify_xgboost_artifact,
)

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-record",
        default="reports/P09_T2_xgboost_run.json",
        help="optional P09-T2 JSON record to compare the candidate's validation predictions",
    )
    args = parser.parse_args()

    artifact_dir = PROJECT_ROOT / "artifacts/development"
    manifest_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"]["sha256"] != TRAIN_SHA256:
        raise ValueError("Split manifest differs from the reviewed FD001/train source.")
    source_path = PROJECT_ROOT / "data/raw/train_FD001.txt"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != TRAIN_SHA256:
        raise ValueError("FD001/train source hash differs from the reviewed source.")

    table_paths = {
        role: artifact_dir / f"{role}_features.json" for role in ("fitting", "validation")
    }
    tables = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=manifest_digest(manifest),
            expected_config=FeatureConfig(),
        )
        for role, path in table_paths.items()
    }
    result = train_xgboost(tables["fitting"], tables["validation"])

    prior_record = PROJECT_ROOT / args.run_record
    if prior_record.is_file():
        recorded = json.loads(prior_record.read_text(encoding="utf-8"))
        expected = [
            (row["dataset_partition"], row["unit_id"], row["cutoff_cycle"], row["prediction"])
            for row in recorded["predictions"]
        ]
        actual = [
            (row.dataset_partition, row.unit_id, row.cutoff_cycle, row.prediction)
            for row in result.predictions
        ]
        if recorded["model_version"] != MODEL_VERSION or actual != expected:
            raise ValueError("Fresh candidate does not reproduce the P09-T2 validation outputs.")

    table_digests = {
        role: json.loads(path.read_text(encoding="utf-8"))["sha256"]
        for role, path in table_paths.items()
    }
    saved = save_xgboost_artifact(
        result,
        tables["fitting"],
        tables["validation"],
        artifact_dir,
        fitting_dataset_sha256=table_digests["fitting"],
        validation_dataset_sha256=table_digests["validation"],
    )
    verified = verify_xgboost_artifact(saved["manifest_path"], tables["validation"])
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/verify_xgboost_service.py"),
            str(saved["manifest_path"]),
            str(source_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"model_version: {MODEL_VERSION}")
    print(f"model_sha256: {saved['model_sha256']}")
    print(f"validation_manifest: {saved['manifest_path']}")
    print(f"prediction_count: {verified['prediction_count']}")
    print(f"mae_cycles: {verified['mae_cycles']}")
    print(f"rmse_cycles: {verified['rmse_cycles']}")


if __name__ == "__main__":
    main()
