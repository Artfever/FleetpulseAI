"""Verify common validation identities and summarize existing P11 candidates."""

from __future__ import annotations

import json
import math

from fleetpulse import PROJECT_ROOT

EXPECTED_SOURCE_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
EXPECTED_SPLIT_SHA256 = "6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92"
ARTIFACTS = PROJECT_ROOT / "artifacts"
OUTPUT = ARTIFACTS / "experiments/P11_S3_complexity_summary.json"


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def _normalize(predictions: list[dict], *, cycles_key: str, target_key: str) -> list[dict]:
    rows = []
    for row in predictions:
        rows.append({
            "key": [
                row["dataset_partition"],
                int(row["unit_id"]),
                int(row["cutoff_cycle"]),
            ],
            "true_rul_cycles": int(row[target_key]),
            "prediction_cycles": float(row[cycles_key]),
        })
    return rows


def _metrics(rows: list[dict]) -> tuple[float, float]:
    errors = [row["prediction_cycles"] - row["true_rul_cycles"] for row in rows]
    mae = math.fsum(abs(value) for value in errors) / len(errors)
    rmse = math.sqrt(math.fsum(value * value for value in errors) / len(errors))
    return mae, rmse


def main() -> None:
    constant = _load("artifacts/development/constant_rul_validation.json")["payload"]
    constant_eval = constant["evaluation"]
    candidates = [
        {
            "name": "Mean constant baseline",
            "family": "constant",
            "configuration": "fitting-label mean; no sensor features",
            "source": constant_eval["source_sha256"],
            "split": constant_eval["split_manifest_sha256"],
            "rows": _normalize(
                constant_eval["predictions"],
                cycles_key="prediction_cycles",
                target_key="true_rul_cycles",
            ),
            "recorded_mae": constant_eval["mae_cycles"],
            "recorded_rmse": constant_eval["rmse_cycles"],
            "results_path": "artifacts/development/constant_rul_validation.json",
        }
    ]
    sources = (
        (
            "Decision Tree, 30-cycle window",
            "artifacts/experiments/P11_S1_window_comparison/window_30/results.json",
            "prediction_cycles",
            "true_rul_cycles",
            "protocol",
        ),
        (
            "Random Forest, depth 16",
            "artifacts/experiments/P11_A1_tree_search/03_random_forest_depth_16/results.json",
            "prediction",
            "true_rul",
            "protocol",
        ),
        (
            "XGBoost, depth 4",
            "artifacts/experiments/P11_A1_tree_search/05_xgboost_depth_4/results.json",
            "prediction",
            "true_rul",
            "protocol",
        ),
        (
            "MLP, hidden (64, 32)",
            "artifacts/experiments/P11_S2_mlp_settings/width_64_32/p11_s2_results.json",
            "prediction_cycles",
            "true_rul_cycles",
            None,
        ),
        (
            "MLP, hidden (128, 64)",
            "artifacts/experiments/P11_S2_mlp_settings/width_128_64/p11_s2_results.json",
            "prediction_cycles",
            "true_rul_cycles",
            None,
        ),
    )
    for name, path, prediction_key, target_key, protocol_key in sources:
        document = _load(path)
        protocol = document[protocol_key] if protocol_key else document
        predictions = document["predictions"]
        metrics = document.get("metrics", document)
        candidates.append({
            "name": name,
            "family": document.get("family", document.get("model", {}).get("name", "MLP")),
            "configuration": document.get("parameters", document.get("model_parameters", document.get("hidden_dims"))),
            "source": protocol["training_source_sha256"] if protocol_key else document["source_sha256"],
            "split": protocol["split_manifest_sha256"] if protocol_key else document["split_manifest_sha256"],
            "rows": _normalize(predictions, cycles_key=prediction_key, target_key=target_key),
            "recorded_mae": metrics["mae_cycles"],
            "recorded_rmse": metrics["rmse_cycles"],
            "selected_validation_mse_cycles_squared": document.get(
                "selected_validation_mse_cycles_squared"
            ),
            "best_epoch": document.get("best_epoch"),
            "seed": document.get("random_seed", document.get("seed")),
            "fit_and_predict_seconds": metrics.get(
                "fit_and_predict_runtime_seconds", document.get("runtime_seconds")
            ),
            "results_path": path,
        })

    reference_keys = [row["key"] for row in candidates[0]["rows"]]
    reference_targets = [row["true_rul_cycles"] for row in candidates[0]["rows"]]
    if len(reference_keys) != 20 or len(set(map(tuple, reference_keys))) != 20:
        raise ValueError("Expected 20 unique validation engine/cutoff identities.")
    for candidate in candidates:
        if candidate["source"] != EXPECTED_SOURCE_SHA256 or candidate["split"] != EXPECTED_SPLIT_SHA256:
            raise ValueError(f"Training-source/split provenance mismatch: {candidate['name']}.")
        rows = candidate["rows"]
        if [row["key"] for row in rows] != reference_keys:
            raise ValueError(f"Validation engine/cutoff identities differ: {candidate['name']}.")
        if [row["true_rul_cycles"] for row in rows] != reference_targets:
            raise ValueError(f"Validation scoring labels differ: {candidate['name']}.")
        mae, rmse = _metrics(rows)
        if not math.isclose(mae, candidate["recorded_mae"], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Per-engine MAE does not match recorded metric: {candidate['name']}.")
        if not math.isclose(rmse, candidate["recorded_rmse"], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Per-engine RMSE does not match recorded metric: {candidate['name']}.")
        candidate["mae_cycles"] = mae
        candidate["rmse_cycles"] = rmse
        candidate["engine4_cycle164"] = next(
            row for row in rows if row["key"] == ["FD001/train", 4, 164]
        )
        del candidate["rows"]

    index = {
        "schema_version": "p11-s3-complexity-summary-v1",
        "training_source_sha256": EXPECTED_SOURCE_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "validation_engine_count": len(reference_keys),
        "validation_identities_match": True,
        "scoring_labels_match": True,
        "metric_recomputation_matches_records": True,
        "official_test_data_opened": False,
        "candidates": candidates,
    }
    OUTPUT.write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("candidates: 6")
    print("same_validation_engines_cutoffs_labels: 20/20")
    print("recorded_MAE_RMSE_recomputed: 6/6 match")
    for row in candidates:
        print(
            f"{row['name']}: MAE={row['mae_cycles']:.6f} "
            f"RMSE={row['rmse_cycles']:.6f}"
        )
    print(f"index: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
