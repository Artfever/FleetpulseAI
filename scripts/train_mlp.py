"""Train the bounded P10 MLP once and save/verify its ignored development bundle."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.mlp import train_mlp
from fleetpulse.models.mlp_artifact import save_mlp_artifact

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/development/mlp"
    )
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    folder = PROJECT_ROOT / "artifacts/development"
    manifest = json.loads((PROJECT_ROOT / "artifacts/split_manifest.json").read_text(encoding="utf-8"))
    expected_manifest = manifest_digest(manifest)
    paths = {role: folder / f"{role}_features.json" for role in ("fitting", "validation")}
    tables = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=expected_manifest,
            expected_config=FeatureConfig(),
        )
        for role, path in paths.items()
    }
    result = train_mlp(
        tables["fitting"], tables["validation"], max_epochs=args.max_epochs,
        patience=args.patience,
    )
    hashes = {
        role: json.loads(path.read_text(encoding="utf-8"))["sha256"]
        for role, path in paths.items()
    }
    saved = save_mlp_artifact(
        result,
        tables["fitting"],
        tables["validation"],
        args.output_dir,
        fitting_dataset_sha256=hashes["fitting"],
        validation_dataset_sha256=hashes["validation"],
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/verify_mlp_artifact.py"),
            str(saved["manifest_path"]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"best_epoch: {result.best_epoch}/{len(result.training_losses)}")
    print(f"best_validation_mse_cycles_squared: {result.validation_losses[result.best_epoch - 1]:.12f}")
    print(f"runtime_seconds: {result.runtime_seconds:.6f}")
    print(f"weights_sha256: {saved['payload']['model']['sha256']}")
    print(f"artifact_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
