"""Smoke-test all frozen final models using only the tracked deployment demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fleetpulse.data.loader import load_engine_history
from fleetpulse.services.final_comparison import compare_final_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt"
REFERENCE_ENGINE = 4
REFERENCE_CYCLE = 164


def run_smoke() -> dict[str, object]:
    """Load final bundles and infer from the packaged prefix, without fitting."""
    demo_bytes = DEMO_PATH.read_bytes()
    rows = load_engine_history(DEMO_PATH)
    history = [
        row
        for row in rows
        if int(row["unit_id"]) == REFERENCE_ENGINE
        and int(row["cycle"]) <= REFERENCE_CYCLE
    ]
    if not history or int(history[-1]["cycle"]) != REFERENCE_CYCLE:
        raise ValueError("Packaged demo does not contain the required engine 4/cycle 164 prefix.")

    result = compare_final_models(
        history,
        dataset_partition="FD001/train",
        unit_id=REFERENCE_ENGINE,
        artifact_root=PROJECT_ROOT,
    )
    if result.failures or len(result.predictions) != 5:
        raise RuntimeError(
            "Expected all five final bundles to load; "
            f"predictions={len(result.predictions)}, failures={len(result.failures)}."
        )
    return {
        "schema_version": "p23-a1-release-smoke-v1",
        "demo_path": DEMO_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "demo_sha256": hashlib.sha256(demo_bytes).hexdigest(),
        "dataset_partition": result.dataset_partition,
        "unit_id": result.unit_id,
        "observed_cycle": result.observed_cycle,
        "history_rows": len(history),
        "final_models": [
            {
                "model_name": item.model_name,
                "model_version": item.model_version,
                "raw_rul_cycles": item.raw_rul_cycles,
                "manifest_payload_sha256": item.manifest_payload_sha256,
                "weights_sha256": item.weights_sha256,
            }
            for item in result.predictions
        ],
        "failures": [],
        "inference_only": True,
        "source_training_file_used": False,
    }


def main() -> None:
    result = run_smoke()
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
