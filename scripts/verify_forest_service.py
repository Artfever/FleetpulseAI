"""Replay saved forest validation prefixes through shared inference; never fit."""

import hashlib
import json
import sys
from pathlib import Path

from fleetpulse.data.loader import load_engine_history
from fleetpulse.services import predict_rul


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_forest_service.py MANIFEST TRAIN_HISTORY")
    manifest, source = map(Path, sys.argv[1:])
    payload = json.loads(manifest.read_text(encoding="utf-8"))["payload"]
    if payload["model"]["name"] != "random-forest-rul":
        raise ValueError("Expected the saved forest candidate.")
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload["training"]["source_sha256"]:
        raise ValueError("Source history hash differs from the saved training provenance.")
    history = load_engine_history(source)
    checked = []
    for saved in payload["evaluation"]["predictions"]:
        # Labels and future observations never enter the service.
        prefix = [
            row
            for row in history
            if row["unit_id"] == saved["unit_id"] and row["cycle"] <= saved["cutoff_cycle"]
        ]
        result = predict_rul(
            prefix,
            dataset_partition=saved["dataset_partition"],
            unit_id=saved["unit_id"],
            manifest_path=manifest,
        )
        difference = abs(result.raw_rul_cycles - saved["prediction_cycles"])
        if difference != 0:
            raise ValueError(
                f"Service differs from offline prediction for engine {saved['unit_id']}."
            )
        checked.append(
            {
                "unit_id": saved["unit_id"],
                "cutoff_cycle": saved["cutoff_cycle"],
                "raw_rul_cycles": result.raw_rul_cycles,
                "absolute_difference": difference,
            }
        )
    if not checked:
        raise ValueError("No saved validation predictions to verify.")
    print(
        json.dumps(
            {
                "prediction_count": len(checked),
                "max_absolute_difference": max(row["absolute_difference"] for row in checked),
                "predictions": checked,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
