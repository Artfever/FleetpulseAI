"""Fresh-process verification for a saved P10 MLP development bundle."""

import argparse
import json
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.mlp_artifact import verify_mlp_artifact

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", nargs="?", type=Path,
        default=PROJECT_ROOT / "artifacts/development/mlp/mlp_training_manifest.json",
    )
    args = parser.parse_args()
    folder = PROJECT_ROOT / "artifacts/development"
    paths = {role: folder / f"{role}_features.json" for role in ("fitting", "validation")}
    split = json.loads((PROJECT_ROOT / "artifacts/split_manifest.json").read_text(encoding="utf-8"))
    tables = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=manifest_digest(split),
            expected_config=FeatureConfig(),
        )
        for role, path in paths.items()
    }
    hashes = {
        role: json.loads(path.read_text(encoding="utf-8"))["sha256"]
        for role, path in paths.items()
    }
    outcome = verify_mlp_artifact(
        args.manifest,
        tables["fitting"],
        tables["validation"],
        fitting_dataset_sha256=hashes["fitting"],
        validation_dataset_sha256=hashes["validation"],
    )
    print(json.dumps(outcome, sort_keys=True))


if __name__ == "__main__":
    main()
