"""Build a compact selection view from independently audited P11 result files."""

from __future__ import annotations

import hashlib
import json
import math

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.selection_manifest import SCHEMA_VERSION, load_selection_manifest

SUMMARY_PATH = PROJECT_ROOT / "artifacts/experiments/P11_S3_complexity_summary.json"
AUDIT_PATH = PROJECT_ROOT / "artifacts/experiments/P11_T1_validation_comparability.json"
OUTPUT = PROJECT_ROOT / "artifacts/selection_manifest.json"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_manifest():
    summary = _read(SUMMARY_PATH)
    audit = _read(AUDIT_PATH)
    if summary["schema_version"] != "p11-s3-complexity-summary-v1":
        raise ValueError("unexpected P11-S3 summary schema")
    if audit["schema_version"] != "p11-t1-validation-comparability-v1":
        raise ValueError("unexpected P11-T1 audit schema")
    if not audit["all_candidate_keys_and_targets_match"]:
        raise ValueError("P11-T1 audit did not confirm common validation identities")
    if not summary["validation_identities_match"] or not summary["scoring_labels_match"]:
        raise ValueError("P11-S3 summary did not confirm common validation identities")
    if summary["training_source_sha256"] != audit["training_source_sha256"]:
        raise ValueError("P11-S3 and P11-T1 source hashes differ")
    if summary["split_manifest_sha256"] != audit["split_manifest_sha256"]:
        raise ValueError("P11-S3 and P11-T1 split hashes differ")
    if summary["validation_engine_count"] != 20 or audit["validation_engine_count"] != 20:
        raise ValueError("unexpected validation cohort size")

    audited_runs = {row["artifact_path"]: row for row in audit["candidates"]}
    comparison = []
    for candidate in summary["candidates"]:
        path = candidate["results_path"]
        audit_row = audited_runs.get(path)
        if audit_row is None:
            raise ValueError(f"no P11-T1 result-file audit for {path}")
        result_path = PROJECT_ROOT / path
        if _sha256(result_path) != audit_row["artifact_sha256"]:
            raise ValueError(f"result artifact changed since P11-T1 audit: {path}")
        if audit_row["sample_count"] != 20 or not audit_row["same_engine_cutoffs_and_targets"]:
            raise ValueError(f"result artifact is not cohort-comparable: {path}")
        if not math.isclose(
            candidate["mae_cycles"], audit_row["mae_cycles_recomputed"], rel_tol=0, abs_tol=1e-9
        ) or not math.isclose(
            candidate["rmse_cycles"], audit_row["rmse_cycles_recomputed"], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"P11-S3 and P11-T1 metrics differ for {candidate['name']}")
        comparison.append(
            {
                "name": candidate["name"],
                "family": candidate["family"],
                "configuration": candidate["configuration"],
                "seed": candidate.get("seed"),
                "best_epoch": candidate.get("best_epoch"),
                "selected_validation_mse_cycles_squared": candidate.get(
                    "selected_validation_mse_cycles_squared"
                ),
                "mae_cycles": candidate["mae_cycles"],
                "rmse_cycles": candidate["rmse_cycles"],
                "fit_and_predict_seconds": candidate.get("fit_and_predict_seconds"),
                "sample_count": 20,
                "training_source_sha256": summary["training_source_sha256"],
                "split_manifest_sha256": summary["split_manifest_sha256"],
                "result_artifact_path": path,
                "result_artifact_sha256": audit_row["artifact_sha256"],
                "engine4_cycle164": candidate["engine4_cycle164"],
            }
        )

    min_mae = min(comparison, key=lambda row: row["mae_cycles"])
    min_rmse = min(comparison, key=lambda row: row["rmse_cycles"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_partition": "FD001/train",
        "validation_sample_unit": "engine-prefix",
        "validation_engine_count": 20,
        "validation_keys_and_targets": audit["validation_keys_and_targets"],
        "training_source_sha256": summary["training_source_sha256"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
        "candidate_count": len(comparison),
        "candidates": comparison,
        "decision": {
            "status": "team_choice_pending_p11_t3",
            "lowest_mae_candidate": min_mae["name"],
            "lowest_rmse_candidate": min_rmse["name"],
            "reason": (
                "The lowest MAE and lowest RMSE belong to different candidates, and all scores "
                "come from the repeatedly used development-validation selection cohort. Keep the "
                "trade-off explicit for the team decision in P11-T3."
            ),
        },
        "official_test_data_opened": False,
        "limitations": [
            "These are FD001/train development-validation selection results, not an untouched final estimate.",
            "Validation scores do not establish individual-engine accuracy or maintenance readiness.",
            "The source run files are Git-ignored; their paths and SHA-256 values are retained for local reproduction.",
        ],
    }


def main():
    payload = build_manifest()
    document = {"payload": payload, "sha256": _canonical_sha256(payload)}
    OUTPUT.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    loaded = load_selection_manifest(OUTPUT)
    print(f"comparison_candidates: {len(loaded['candidates'])}")
    print(f"common_validation_prefixes: {loaded['validation_engine_count']}/20")
    print(f"lowest_MAE: {loaded['decision']['lowest_mae_candidate']}")
    print(f"lowest_RMSE: {loaded['decision']['lowest_rmse_candidate']}")
    print(f"production_model_choice: {loaded['decision']['status']}")
    print(f"manifest_sha256: {document['sha256']}")
    print(f"output: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
