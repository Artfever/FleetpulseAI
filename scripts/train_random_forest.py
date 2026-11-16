"""Train one fixed forest, or save the same trusted A1 checkpoint without refitting."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import joblib

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.forest_artifact import save_random_forest_artifact
from fleetpulse.models.train_forest import RandomForestValidationResult, train_random_forest

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def load_recorded_candidate(checkpoint, run_record, table_paths, fitting, validation):
    """Reuse only this user's trusted, hash-checked local A1 result and provenance."""
    record = json.loads(Path(run_record).read_text(encoding="utf-8"))
    content = Path(checkpoint).read_bytes()
    if hashlib.sha256(content).hexdigest() != record["checkpoint_sha256"]:
        raise ValueError("Recorded candidate checkpoint hash mismatch.")
    actual_hashes = {
        role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in table_paths.items()
    }
    if (
        record["table_file_sha256"] != actual_hashes
        or record["source_sha256"] != fitting.source_sha256
        or record["split_manifest_sha256"] != fitting.split_manifest_sha256
        or record["feature_config"] != json.loads(json.dumps(asdict(fitting.config)))
    ):
        raise ValueError("Recorded run does not match current table/config/source inputs.")
    result = joblib.load(checkpoint)
    if (
        not isinstance(result, RandomForestValidationResult)
        or result.parameters != record["parameters"]
        or result.random_seed != record["random_seed"]
        or result.runtime_seconds != record["runtime_seconds"]
        or [asdict(row) for row in result.predictions] != record["predictions"]
        or result.metrics.mae != record["mae_cycles"]
        or result.metrics.rmse != record["rmse_cycles"]
        or len(validation.rows) != record["validation_engines"]
    ):
        raise ValueError("Candidate checkpoint does not match its run record.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-checkpoint", type=Path)
    parser.add_argument("--run-record", type=Path)
    args = parser.parse_args()
    if bool(args.reuse_checkpoint) != bool(args.run_record):
        parser.error("--reuse-checkpoint and --run-record must be supplied together")
    folder = PROJECT_ROOT / "artifacts/development"
    split = json.loads((PROJECT_ROOT / "artifacts/split_manifest.json").read_text(encoding="utf-8"))
    paths = {role: folder / f"{role}_features.json" for role in ("fitting", "validation")}
    tables = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=manifest_digest(split),
            expected_config=FeatureConfig(),
        )
        for role, path in paths.items()
    }
    if args.reuse_checkpoint:
        result = load_recorded_candidate(
            args.reuse_checkpoint, args.run_record, paths, tables["fitting"], tables["validation"]
        )
    else:
        result = train_random_forest(tables["fitting"], tables["validation"])
    hashes = {
        role: json.loads(path.read_text(encoding="utf-8"))["sha256"] for role, path in paths.items()
    }
    saved = save_random_forest_artifact(
        result,
        tables["fitting"],
        tables["validation"],
        folder,
        fitting_dataset_sha256=hashes["fitting"],
        validation_dataset_sha256=hashes["validation"],
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/verify_random_forest_artifact.py"),
            str(saved["manifest_path"]),
            str(paths["validation"]),
        ],
        check=True,
    )
    print(
        f"candidate source: {'existing A1 fit; no refit' if args.reuse_checkpoint else 'one new fixed fit'}"
    )
    print(f"model SHA-256: {saved['model_sha256']}")
    print(f"fit+prediction seconds: {result.runtime_seconds}")
    print(f"MAE: {result.metrics.mae}; RMSE: {result.metrics.rmse}")


if __name__ == "__main__":
    main()
