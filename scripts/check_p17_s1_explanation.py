"""Explain the frozen final Decision Tree on one packaged FD001/train prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.models.final_tree_bundle import load_final_tree_bundle
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY
from fleetpulse.services.explanations import (
    build_fd001_training_background,
    explain_final_tree_prediction,
)

PREFIX_PATH = PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
TRAINING_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
UNIT_ID = 4
OBSERVED_CYCLE = 164


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    *,
    artifact_root: Path = PROJECT_ROOT,
    training_path: Path = TRAINING_PATH,
    prefix_path: Path = PREFIX_PATH,
) -> dict[str, object]:
    """Run exact bundle inference/explanation and return a provenance report."""
    spec = FINAL_MODEL_REGISTRY["decision-tree-rul"]
    manifest_path = artifact_root / spec.manifest_path
    bundle = load_final_tree_bundle(manifest_path)
    expected_source = bundle.payload["training"]["training_source_sha256"]
    background = build_fd001_training_background(
        training_path,
        expected_training_source_sha256=expected_source,
        feature_config=bundle.feature_config,
    )
    prefix_records = load_engine_history(prefix_path)
    history = [
        row
        for row in prefix_records
        if int(row["unit_id"]) == UNIT_ID and int(row["cycle"]) <= OBSERVED_CYCLE
    ]
    if not history or max(int(row["cycle"]) for row in history) != OBSERVED_CYCLE:
        raise RuntimeError("Packaged FD001/train history does not reach the selected cutoff.")
    result = explain_final_tree_prediction(
        history,
        dataset_partition="FD001/train",
        unit_id=UNIT_ID,
        background=background,
        artifact_root=artifact_root,
    )
    return {
        "schema_version": "p17-s1-final-tree-explanation-v1",
        "model_name": result.model_name,
        "model_version": result.model_version,
        "stage": "final",
        "dataset_partition": result.dataset_partition,
        "unit_id": result.unit_id,
        "observed_cycle": result.observed_cycle,
        "raw_rul_cycles": result.raw_rul_cycles,
        "base_value_cycles": result.base_value_cycles,
        "additivity_error_cycles": result.additivity_error_cycles,
        "feature_config_sha256": result.feature_config_sha256,
        "input_history_sha256": result.input_history_sha256,
        "manifest_payload_sha256": result.manifest_payload_sha256,
        "weights_sha256": result.weights_sha256,
        "training_source_sha256": result.training_source_sha256,
        "background": {
            "selection_policy": result.background_selection_policy,
            "row_count": result.background_row_count,
            "cycle": 90,
            "source": training_path.relative_to(PROJECT_ROOT).as_posix()
            if training_path.is_relative_to(PROJECT_ROOT)
            else str(training_path),
            "source_sha256": _sha256_file(training_path),
            "feature_matrix_sha256": result.background_sha256,
            "feature_names": list(bundle.feature_names),
        },
        "contributions": [asdict(item) for item in result.contributions],
        "global_summary_top_10": [asdict(item) for item in result.global_summary],
        "limitations": [
            "One packaged FD001/train prefix and one frozen Decision Tree were explained.",
            "SHAP values allocate this model's raw RUL output relative to the selected training reference; they are not causal effects, confidence intervals, or accuracy evidence.",
            "Interventional explanations use a fixed cycle-90 reference row for each of the 100 original training engines; correlated features can make masked combinations unrealistic.",
            "The local final model bundle and source training data are not tracked artifacts.",
            "No true-RUL labels, official-test histories, or future input rows were read for this explanation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "P17_S1_final_tree_explanation.json",
        help="where to write the verified result record",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS: explained final Decision Tree on FD001/train engine "
        f"{UNIT_ID} cycle {OBSERVED_CYCLE}; raw RUL={report['raw_rul_cycles']:.12g} cycles"
    )
    print(f"Explanation record: {args.output.resolve()}")


if __name__ == "__main__":
    main()
