"""Replay saved XGBoost validation predictions through registered raw-history service."""

import hashlib
import json
import sys
from pathlib import Path

from fleetpulse.data.loader import load_engine_history
from fleetpulse.services import predict_rul


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_xgboost_service.py MANIFEST TRAIN_HISTORY")
    manifest, source = map(Path, sys.argv[1:])
    payload = json.loads(manifest.read_text(encoding="utf-8"))["payload"]
    if payload["model"]["name"] != "xgboost-rul":
        raise ValueError("Expected the saved XGBoost candidate.")
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload["training"]["source_sha256"]:
        raise ValueError("Training source hash differs from saved candidate provenance.")
    histories = load_engine_history(source)
    checked = []
    for saved in payload["evaluation"]["predictions"]:
        prefix = [
            row
            for row in histories
            if row["unit_id"] == saved["unit_id"] and row["cycle"] <= saved["cutoff_cycle"]
        ]
        prediction = predict_rul(
            prefix,
            dataset_partition=saved["dataset_partition"],
            unit_id=saved["unit_id"],
            manifest_path=manifest,
        )
        difference = abs(prediction.raw_rul_cycles - saved["prediction_cycles"])
        if difference != 0:
            raise ValueError(
                f"Shared service differs for engine {saved['unit_id']} at "
                f"cycle {saved['cutoff_cycle']}: {difference} cycles."
            )
        checked.append(
            {
                "unit_id": saved["unit_id"],
                "cutoff_cycle": saved["cutoff_cycle"],
                "raw_rul_cycles": prediction.raw_rul_cycles,
                "difference_cycles": difference,
            }
        )
    if not checked:
        raise ValueError("Candidate manifest contains no validation predictions.")
    print(
        json.dumps(
            {
                "model_version": payload["model"]["version"],
                "prediction_count": len(checked),
                "max_difference_cycles": max(row["difference_cycles"] for row in checked),
                "predictions": checked,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
