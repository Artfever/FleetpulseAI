"""Verify P08-P11 candidate scores share FD001/train validation rows and labels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.features.build_dataset import manifest_digest

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
MANIFEST_PATH = PROJECT_ROOT / "artifacts/split_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
EXPECTED_SPLIT_SHA256 = manifest_digest(MANIFEST)
EXPECTED_ROWS = [
    {
        "key": [entry["dataset_partition"], entry["unit_id"], entry["cutoff_cycle"]],
        "true_rul_cycles": entry["target_rul"],
    }
    for entry in sorted(MANIFEST["validation_prefixes"], key=lambda value: value["unit_id"])
]
OUTPUT = PROJECT_ROOT / "artifacts/experiments/P11_T1_validation_comparability.json"

RUNS = [
    ("Mean constant baseline", "artifacts/development/constant_rul_validation.json"),
    ("P06 Decision Tree", "artifacts/development/decision_tree_validation.json"),
    ("P08 Random Forest default", "artifacts/development/random_forest_validation.json"),
    ("P09 XGBoost default", "artifacts/development/xgboost_validation.json"),
    ("P10 MLP baseline", "artifacts/development/mlp/mlp_training_manifest.json"),
    *((f"P11-S1 Decision Tree window {window}",
       f"artifacts/experiments/P11_S1_window_comparison/window_{window}/results.json")
      for window in (10, 20, 30)),
    *((f"P11-A1 {family} depth {depth}",
       f"artifacts/experiments/P11_A1_tree_search/{index:02d}_{family}_depth_{depth}/results.json")
      for index, family, depth in (
          (1, "random_forest", 8), (2, "random_forest", 12), (3, "random_forest", 16),
          (4, "xgboost", 3), (5, "xgboost", 4), (6, "xgboost", 6),
      )),
    *((f"P11-A2 {name}",
       f"artifacts/experiments/P11_A2_feature_ablation/{name}/results.json")
      for name in ("all_channels", "sensors_only", "settings_only")),
    *((f"P11-S2 MLP hidden {dims}",
       f"artifacts/experiments/P11_S2_mlp_settings/width_{dims}/p11_s2_results.json")
      for dims in ("32_16", "64_32", "128_64")),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first(document: dict, *names: str):
    for name in names:
        if name in document:
            return document[name]
    raise KeyError(f"None of the expected fields exists: {names}.")


def _normalize_prediction(row: dict) -> dict:
    return {
        "key": [
            row["dataset_partition"],
            int(row["unit_id"]),
            int(_first(row, "cutoff_cycle", "selected_cycle")),
        ],
        "true_rul_cycles": int(_first(row, "true_rul_cycles", "true_rul")),
        "prediction_cycles": float(_first(row, "prediction_cycles", "prediction")),
    }


def _metrics(rows: list[dict]) -> tuple[float, float]:
    errors = [row["prediction_cycles"] - row["true_rul_cycles"] for row in rows]
    return (
        math.fsum(abs(error) for error in errors) / len(errors),
        math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
    )


def _audit(name: str, relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload", document)
    protocol = payload.get("protocol", {})
    training = payload.get("training", {})
    evaluation = payload.get("evaluation", {})
    predictions = evaluation.get("predictions", payload.get("predictions"))
    if not isinstance(predictions, list):
        raise TypeError(f"Missing per-engine predictions in {relative_path}.")
    rows = [_normalize_prediction(row) for row in predictions]
    if len(rows) != len(EXPECTED_ROWS) or [row["key"] for row in rows] != [
        row["key"] for row in EXPECTED_ROWS
    ]:
        raise ValueError(f"Held-out engines/cutoffs differ for {name}.")
    if [row["true_rul_cycles"] for row in rows] != [
        row["true_rul_cycles"] for row in EXPECTED_ROWS
    ]:
        raise ValueError(f"Scoring targets differ for {name}.")

    containers = (protocol, training, evaluation, payload)
    source_sha256 = next(
        container[key]
        for container in containers
        for key in ("training_source_sha256", "source_sha256")
        if key in container
    )
    split_sha256 = next(
        container["split_manifest_sha256"]
        for container in containers
        if "split_manifest_sha256" in container
    )
    if source_sha256 != TRAIN_SHA256 or split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Source or split provenance differs for {name}.")

    saved_metrics = evaluation if "mae_cycles" in evaluation else payload.get("metrics", payload)
    mae, rmse = _metrics(rows)
    if "mae_cycles" in saved_metrics and not math.isclose(
        mae, float(saved_metrics["mae_cycles"]), rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError(f"Per-engine MAE differs from saved metric for {name}.")
    if "rmse_cycles" in saved_metrics and not math.isclose(
        rmse, float(saved_metrics["rmse_cycles"]), rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError(f"Per-engine RMSE differs from saved metric for {name}.")
    return {
        "name": name,
        "artifact_path": relative_path,
        "artifact_sha256": _sha256(path),
        "sample_count": len(rows),
        "same_engine_cutoffs_and_targets": True,
        "mae_cycles_recomputed": mae,
        "rmse_cycles_recomputed": rmse,
        "recorded_metric_pair_verified": "mae_cycles" in saved_metrics and "rmse_cycles" in saved_metrics,
        "engine4_cycle164": next(row for row in rows if row["key"] == ["FD001/train", 4, 164]),
    }


def main() -> None:
    candidates = [_audit(name, path) for name, path in RUNS]
    index = {
        "schema_version": "p11-t1-validation-comparability-v1",
        "training_source_sha256": TRAIN_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "validation_sample_unit": "engine-prefix",
        "validation_engine_count": len(EXPECTED_ROWS),
        "validation_keys_and_targets": EXPECTED_ROWS,
        "candidate_count": len(candidates),
        "all_candidate_keys_and_targets_match": True,
        "official_test_data_opened": False,
        "candidates": candidates,
    }
    OUTPUT.write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"candidate_artifacts: {len(candidates)}")
    print(f"validation_engine_prefixes: {len(EXPECTED_ROWS)}")
    print("same_keys_cutoffs_targets: all candidates matched")
    recorded_pairs = sum(candidate["recorded_metric_pair_verified"] for candidate in candidates)
    print(f"recorded_metric_pairs_recomputed: {recorded_pairs}/{len(candidates)} matched")
    print("P10 MLP metrics: recomputed from saved per-engine predictions (not stored in manifest)")
    print(f"P04_semantic_split_sha256: {EXPECTED_SPLIT_SHA256}")
    print(f"comparability_index: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
