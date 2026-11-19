"""P22-A1 experiment matrix is fixed, repeatable and label-blind."""

import json

from scripts.run_p22_a1_fleet_scenarios import load_raw_predictions, run_experiment


def test_p22_a1_matrix_covers_fleet_team_and_buffer_values_without_worsening_ga():
    predictions = {unit_id: float(30 + unit_id) for unit_id in range(1, 21)}

    result = run_experiment(predictions)

    assert [(row["fleet_size"], row["team_count"]) for row in result["scenarios"]] == [
        (size, teams)
        for size in (5, 10, 20)
        for _buffer in (2, 120)
        for teams in (1, 2, 3)
    ]
    assert all(
        row["ga"]["weighted_tardiness_cycles"] <= row["edd"]["weighted_tardiness_cycles"]
        for row in result["scenarios"]
    )
    assert sum(result["ga_outcomes"].values()) == 18
    assert result["ga_outcomes"]["equal"] > 0
    assert result["ga_outcomes"]["better"] > 0
    assert result["scenarios"][0]["engine_ids"] == [1, 2, 3, 4, 5]
    assert result["duration_assumption_cycles"] == "1 + (engine_id mod 3)"
    assert result["priority_weight_assumption"] == "1 + ((engine_id * 7) mod 5)"
    assert result["buffer_levels_cycles"] == [2, 120]


def test_prediction_input_loader_does_not_require_or_expose_true_labels(tmp_path):
    report = {
        "payload": {
            "models": [
                {
                    "name": "decision-tree-rul",
                    "artifact": {
                        "version": "decision-tree-rul-v1",
                        "manifest_file_sha256": "manifest-hash",
                        "weights_sha256": "weights-hash",
                    },
                    "rows": [
                        {"unit_id": 1, "raw_prediction_cycles": 12.5, "true_rul_cycles": 999}
                    ],
                }
            ]
        }
    }
    report_path = tmp_path / "p13.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    predictions, provenance = load_raw_predictions(report_path)

    assert predictions == {1: 12.5}
    assert "true_rul_cycles" not in json.dumps(predictions)
    assert provenance["model_version"] == "decision-tree-rul-v1"
