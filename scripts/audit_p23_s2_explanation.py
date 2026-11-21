"""Reproduce one frozen-model comparison and SHAP example without fitting."""

from __future__ import annotations

import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.services.explanations import (
    check_final_explanation_consistency,
    explain_final_tree_history,
)
from fleetpulse.services.final_comparison import compare_final_models


def main() -> None:
    rows = [
        row
        for row in load_engine_history(PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt")
        if int(row["unit_id"]) == 4
    ]
    comparison = compare_final_models(
        rows,
        dataset_partition="FD001/train",
        unit_id=4,
        artifact_root=PROJECT_ROOT,
    )
    if comparison.failures:
        raise RuntimeError(f"Frozen model comparison failed: {comparison.failures!r}")
    explanation = explain_final_tree_history(
        rows,
        dataset_partition="FD001/train",
        unit_id=4,
        artifact_root=PROJECT_ROOT,
    )
    consistency = check_final_explanation_consistency(comparison, explanation)
    if not consistency.is_consistent:
        raise RuntimeError(f"Prediction/explanation mismatch: {consistency.mismatch_codes!r}")
    output = {
        "dataset_partition": explanation.dataset_partition,
        "unit_id": explanation.unit_id,
        "observed_cycle": explanation.observed_cycle,
        "predictions": [
            {
                "model": prediction.model_name,
                "version": prediction.model_version,
                "raw_rul_cycles": prediction.raw_rul_cycles,
            }
            for prediction in comparison.predictions
        ],
        "model_disagreement_spread_cycles": max(
            item.raw_rul_cycles for item in comparison.predictions
        )
        - min(item.raw_rul_cycles for item in comparison.predictions),
        "explanation": {
            "model": explanation.model_name,
            "version": explanation.model_version,
            "raw_rul_cycles": explanation.raw_rul_cycles,
            "baseline_cycles": explanation.base_value_cycles,
            "background_row_count": explanation.background_row_count,
            "background_sha256": explanation.background_sha256,
            "training_source_sha256": explanation.training_source_sha256,
            "feature_config_sha256": explanation.feature_config_sha256,
            "manifest_payload_sha256": explanation.manifest_payload_sha256,
            "weights_sha256": explanation.weights_sha256,
            "additivity_error_cycles": explanation.additivity_error_cycles,
            "consistency_check": consistency.is_consistent,
            "selected_contributions": {
                feature_name: next(
                    item.contribution_cycles
                    for item in explanation.contributions
                    if item.feature_name == feature_name
                )
                for feature_name in ("sensor_13_mean", "sensor_4_mean")
            },
        },
        "training_or_refit_performed": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
