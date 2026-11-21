"""Run all frozen P12 bundles on the same three packaged FD001/train prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY
from fleetpulse.services import compare_final_models

PREFIX_PATH = PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
EXPECTED_CUTOFFS = {4: 164, 5: 58, 12: 7}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(artifact_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Verify final-bundle identity and compare every model on identical prefixes."""
    records = load_engine_history(PREFIX_PATH)
    if {int(row["unit_id"]) for row in records} != set(EXPECTED_CUTOFFS):
        raise RuntimeError("Packaged prefix engine IDs differ from the checked P17-A1 inputs.")
    results = []
    expected_models = set(FINAL_MODEL_REGISTRY)
    for unit_id, cutoff in EXPECTED_CUTOFFS.items():
        history = [
            row for row in records if int(row["unit_id"]) == unit_id and int(row["cycle"]) <= cutoff
        ]
        comparison = compare_final_models(
            history,
            dataset_partition="FD001/train",
            unit_id=unit_id,
            artifact_root=artifact_root,
        )
        actual_models = {item.model_name for item in comparison.predictions}
        if actual_models != expected_models or comparison.failures:
            raise RuntimeError(
                f"Engine {unit_id}: expected all final bundles, got {sorted(actual_models)}; "
                f"failures={comparison.failures!r}"
            )
        if len({item.vector.values for item in comparison.predictions}) != 1:
            raise RuntimeError(f"Engine {unit_id}: final models received different feature values.")
        if len({item.feature_config_sha256 for item in comparison.predictions}) != 1:
            raise RuntimeError(f"Engine {unit_id}: final models used different feature configs.")
        if len({item.training_source_sha256 for item in comparison.predictions}) != 1:
            raise RuntimeError(f"Engine {unit_id}: final models have different training sources.")
        if len({item.selection_manifest_sha256 for item in comparison.predictions}) != 1:
            raise RuntimeError(f"Engine {unit_id}: final models have different selection lineage.")

        results.append(
            {
                "dataset_partition": comparison.dataset_partition,
                "unit_id": comparison.unit_id,
                "observed_cycle": comparison.observed_cycle,
                "input_history_sha256": comparison.input_history_sha256,
                "raw_min_rul_cycles": comparison.raw_min_rul_cycles,
                "raw_max_rul_cycles": comparison.raw_max_rul_cycles,
                "raw_spread_cycles": comparison.raw_spread_cycles,
                "feature_config_sha256": comparison.predictions[0].feature_config_sha256,
                "feature_count": len(comparison.predictions[0].vector.values),
                "window_row_count": comparison.predictions[0].vector.window_row_count,
                "training_source_sha256": comparison.predictions[0].training_source_sha256,
                "selection_manifest_sha256": comparison.predictions[0].selection_manifest_sha256,
                "predictions": [
                    {
                        "model_name": item.model_name,
                        "model_version": item.model_version,
                        "stage": item.stage,
                        "raw_rul_cycles": item.raw_rul_cycles,
                        "manifest_payload_sha256": item.manifest_payload_sha256,
                        "weights_sha256": item.weights_sha256,
                    }
                    for item in comparison.predictions
                ],
            }
        )

    return {
        "schema_version": "p17-a1-final-model-comparison-v1",
        "input_file": PREFIX_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "input_file_sha256": _sha256_file(PREFIX_PATH),
        "input_row_count": len(records),
        "bundle_root": str(artifact_root.resolve()),
        "models_compared": list(FINAL_MODEL_REGISTRY),
        "results": results,
        "limitations": [
            "The three inputs are packaged FD001/train prefixes, not the complete fleet.",
            "Raw RUL outputs are individual model estimates, not accuracy evidence.",
            "No true-RUL labels or future rows were passed to inference.",
            "The saved final bundle files are local artifacts and are not tracked by Git.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT,
        help="folder containing artifacts/final/... (defaults to this project checkout)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "P17_A1_final_model_comparison.json",
        help="where to write the verified result record",
    )
    args = parser.parse_args()
    report = build_report(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"PASS: compared {len(report['models_compared'])} final models on 3 histories")
    print(f"Input file SHA-256: {report['input_file_sha256']}")
    print(f"Result record: {args.output.resolve()}")


if __name__ == "__main__":
    main()
