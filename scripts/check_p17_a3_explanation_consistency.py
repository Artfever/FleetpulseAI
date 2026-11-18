"""Check final-model prediction/explanation parity on one packaged prefix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.models.final_tree_bundle import load_final_tree_bundle
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY
from fleetpulse.services import (
    build_fd001_training_background,
    check_final_explanation_consistency,
    compare_final_models,
    explain_final_tree_prediction,
)

PREFIX_PATH = PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
TRAINING_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
UNIT_ID = 4
OBSERVED_CYCLE = 164


def build_report(artifact_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Bind explanation identity, feature row and raw prediction to one prefix."""
    records = load_engine_history(PREFIX_PATH)
    history = [
        row
        for row in records
        if int(row["unit_id"]) == UNIT_ID and int(row["cycle"]) <= OBSERVED_CYCLE
    ]
    comparison = compare_final_models(
        history,
        dataset_partition="FD001/train",
        unit_id=UNIT_ID,
        artifact_root=artifact_root,
    )
    spec = FINAL_MODEL_REGISTRY["decision-tree-rul"]
    bundle = load_final_tree_bundle(artifact_root / spec.manifest_path)
    background = build_fd001_training_background(
        TRAINING_PATH,
        expected_training_source_sha256=bundle.payload["training"]["training_source_sha256"],
        feature_config=bundle.feature_config,
    )
    explanation = explain_final_tree_prediction(
        history,
        dataset_partition="FD001/train",
        unit_id=UNIT_ID,
        background=background,
        artifact_root=artifact_root,
    )
    consistency = check_final_explanation_consistency(comparison, explanation)
    if not consistency.is_consistent:
        raise RuntimeError(f"Prediction/explanation mismatch: {consistency.mismatch_codes}")
    matched_prediction = next(
        item for item in comparison.predictions if item.model_name == explanation.model_name
    )
    return {
        "schema_version": "p17-a3-explanation-consistency-v1",
        "consistency": asdict(consistency),
        "prediction": {
            "model_name": matched_prediction.model_name,
            "model_version": matched_prediction.model_version,
            "stage": matched_prediction.stage,
            "raw_rul_cycles": matched_prediction.raw_rul_cycles,
            "manifest_payload_sha256": matched_prediction.manifest_payload_sha256,
            "weights_sha256": matched_prediction.weights_sha256,
            "feature_config_sha256": matched_prediction.feature_config_sha256,
            "input_history_sha256": comparison.input_history_sha256,
            "feature_values": list(matched_prediction.vector.values),
            "feature_names": list(matched_prediction.vector.config.output_schema),
        },
        "explanation": {
            "model_name": explanation.model_name,
            "model_version": explanation.model_version,
            "raw_rul_cycles": explanation.raw_rul_cycles,
            "base_value_cycles": explanation.base_value_cycles,
            "manifest_payload_sha256": explanation.manifest_payload_sha256,
            "weights_sha256": explanation.weights_sha256,
            "feature_config_sha256": explanation.feature_config_sha256,
            "input_history_sha256": explanation.input_history_sha256,
            "background_sha256": explanation.background_sha256,
            "contribution_count": len(explanation.contributions),
            "additivity_error_cycles": explanation.additivity_error_cycles,
            "contributions": [asdict(item) for item in explanation.contributions],
        },
        "all_model_outputs_are_raw_unclipped": True,
        "limitations": [
            "This is one packaged FD001/train history and one frozen model bundle, not fleet-wide or accuracy evidence.",
            "The checker compares only the Decision Tree's raw prediction with its raw-output explanation; display-clamped values are not accepted.",
            "The model bundle, original training file and full report are local/ignored artifacts.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "P17_A3_explanation_consistency.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = report["consistency"]
    print(
        "PASS: prediction/explanation match for "
        f"{result['model_name']} {result['model_version']} "
        f"on engine {result['unit_id']} cycle {result['observed_cycle']}; "
        f"raw RUL={result['raw_rul_cycles']:.12g} cycles"
    )
    print(f"Consistency record: {args.output.resolve()}")


if __name__ == "__main__":
    main()
