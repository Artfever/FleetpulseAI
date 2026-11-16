"""Fresh-process replay of a saved forest against its unchanged validation table."""

import json
import sys
from pathlib import Path

from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.forest_artifact import verify_random_forest_artifact


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_random_forest_artifact.py MANIFEST VALIDATION_TABLE")
    manifest, table = map(Path, sys.argv[1:])
    training = json.loads(manifest.read_text(encoding="utf-8"))["payload"]["training"]
    document = json.loads(table.read_text(encoding="utf-8"))
    if document["sha256"] != training["validation_dataset_sha256"]:
        raise ValueError("Validation table payload hash differs from the forest manifest.")
    validation = load_feature_dataset(
        table,
        expected_source_sha256=training["source_sha256"],
        expected_manifest_sha256=training["split_manifest_sha256"],
    )
    print(json.dumps(verify_random_forest_artifact(manifest, validation), sort_keys=True))


if __name__ == "__main__":
    main()
