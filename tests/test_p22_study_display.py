from __future__ import annotations

import hashlib
import json

import pytest
from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.p22_study_display import (
    load_error_analysis,
    load_scheduler_study,
    load_sensor_sensitivity,
    load_validation_bootstrap,
)


def test_saved_p22_model_studies_load_with_expected_versions_and_units() -> None:
    errors = load_error_analysis(PROJECT_ROOT / "docs/figures/p22_s1_error_analysis.json")
    bootstrap = load_validation_bootstrap(PROJECT_ROOT / "docs/p22_s2_validation_bootstrap.json")
    sensitivity = load_sensor_sensitivity(PROJECT_ROOT / "docs/p22_s3_sensor_sensitivity.json")

    assert errors["engine_count_per_model"] == 100
    assert errors["dataset_partition"] == "FD001/test"
    assert len(bootstrap["models"]) == 5
    assert bootstrap["method"]["replicates"] == 10_000
    assert len(sensitivity["aggregate_summary"]) == 10
    assert sensitivity["holdout_engine_count"] == 20


@pytest.mark.parametrize(
    ("filename", "schema"),
    [
        ("p22_a1_fixed_fleet_scenarios.json", "p22-a1-fixed-fleet-scenarios-v1"),
        ("p22_a2_seed_comparison.json", "p22-a2-seed-comparison-v1"),
    ],
)
def test_saved_scheduler_studies_are_checksummed_and_paired(filename: str, schema: str) -> None:
    study = load_scheduler_study(PROJECT_ROOT / "docs" / filename, expected_schema=schema)

    assert len(study["scenarios"]) == 18
    assert study["input_provenance"]["model_version"] == "decision-tree-rul-v1"
    if schema.endswith("a1-fixed-fleet-scenarios-v1"):
        assert study["ga_outcomes"] == {"equal": 12, "better": 6}
    else:
        assert study["outcome_counts"] == {"equal": 60, "better": 30}


def test_scheduler_loader_rejects_modified_snapshot(tmp_path) -> None:
    payload = {
        "schema": "p22-a1-fixed-fleet-scenarios-v1",
        "dataset_partition": "FD001/test",
        "input_provenance": {
            "report_sha256": "db420935dc14221c50f71f0ca3349b0667bea6b37df45b330026df3d3e8d40d4",
            "model_version": "decision-tree-rul-v1",
        },
        "scenarios": [{} for _ in range(18)],
        "ga_outcomes": {"equal": 12, "better": 6},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    envelope = {
        "payload": payload,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    payload["scenarios"][0]["edd"] = {"weighted_tardiness_cycles": 12.0}
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum is invalid"):
        load_scheduler_study(path, expected_schema="p22-a1-fixed-fleet-scenarios-v1")


def test_model_laboratory_displays_saved_p22_tables_and_figures() -> None:
    page = AppTest.from_file(
        str(PROJECT_ROOT / "app/modules/5_Model_Laboratory.py"), default_timeout=45
    ).run()

    assert not page.exception
    assert any(item.value == "P22 saved model studies" for item in page.subheader)
    assert any("P22-S1" in item.value for item in page.markdown)
    assert any("P22-S2" in item.value for item in page.markdown)
    assert any("P22-S3" in item.value for item in page.markdown)
    assert any("Observed MAE (cycles)" in item.value.columns for item in page.dataframe)
    assert any("Mean absolute raw RUL movement (cycles)" in item.value.columns for item in page.dataframe)


def test_planner_displays_saved_p22_scheduler_studies_separately() -> None:
    page = AppTest.from_file(
        str(PROJECT_ROOT / "app/modules/4_Maintenance_Planner.py"), default_timeout=45
    ).run()

    assert not page.exception
    assert any(item.value == "Saved P22 scheduling studies" for item in page.subheader)
    assert any("not use the selected fleet" in item.value for item in page.caption)
    assert any(
        "EDD weighted tardiness (weighted cycles)" in item.value.columns
        for item in page.dataframe
    )
    assert any("GA weighted tardiness by seed" in item.value.columns for item in page.dataframe)
