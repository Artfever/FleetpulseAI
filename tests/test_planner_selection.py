"""The Fleet Overview selection persists into the planner without fake health fields."""

from contextlib import contextmanager

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, deployment, services, ui


def test_selected_fleet_snapshot_is_available_on_planner_page(monkeypatch, tmp_path):
    manifest = tmp_path / "artifacts/development/decision_tree_validation.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("test manifest", encoding="utf-8")
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    records = [{"unit_id": 4, "cycle": 2}, {"unit_id": 5, "cycle": 2}]
    monkeypatch.setattr(ui, "render_packaged_training_subset", lambda: records)

    @contextmanager
    def use_test_manifest(path, supplied_bundle):
        yield path

    results = (
        services.FleetPredictionResult(
            services.EngineIdentity("FD001/train", 4),
            2,
            30.0,
            30.0,
            "Decision Tree",
            "decision-tree-rul-v1",
            2,
            (),
            (),
            None,
        ),
        services.FleetPredictionResult(
            services.EngineIdentity("FD001/train", 5),
            2,
            42.0,
            42.0,
            "Decision Tree",
            "decision-tree-rul-v1",
            2,
            ("cycle_gaps",),
            (),
            None,
        ),
    )
    monkeypatch.setattr(deployment, "prediction_manifest", use_test_manifest)
    monkeypatch.setattr(services, "predict_fleet", lambda *args, **kwargs: results)
    monkeypatch.setattr(
        services,
        "model_provenance_from_manifest",
        lambda path: {
            "model_name": "Decision Tree",
            "model_version": "decision-tree-rul-v1",
            "artifact_filename": "decision_tree_model.joblib",
            "artifact_sha256": "c" * 64,
            "manifest_sha256": "b" * 64,
            "manifest_payload_sha256": "d" * 64,
            "training_seed": 42,
            "training_parameters": {"max_depth": 8},
            "scikit_learn_version": "1.9.1",
            "feature_config": {"channels": ["sensor_1"], "window_cycles": 30},
            "feature_schema": ["sensor_1_latest"],
        },
    )

    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=30).run()
    fleet = app.switch_page("pages/1_Fleet_Overview.py").run()
    assert not fleet.exception
    fleet.multiselect[0].set_value([5]).run()
    fleet.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY] = "stale scenario"
    fleet.session_state[ui.P20_FLEET_COMPARISON_STATE_KEY] = "stale comparison"
    fleet.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY] = "stale artifact"
    fleet.button[0].click().run()

    assert not fleet.exception
    assert any("Saved 1 selected engine records" in item.value for item in fleet.success)
    saved_provenance = fleet.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY]
    assert saved_provenance["model"]["model_version"] == "decision-tree-rul-v1"
    assert len(saved_provenance["source"]["source_records_sha256"]) == 64
    assert ui.P20_FLEET_SCENARIO_STATE_KEY not in fleet.session_state
    assert ui.P20_FLEET_COMPARISON_STATE_KEY not in fleet.session_state
    assert ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY not in fleet.session_state

    planner = app.switch_page("pages/4_Maintenance_Planner.py").run()
    assert not planner.exception
    assert any("Loaded 1 selected engine records" in item.value for item in planner.success)
    saved = planner.dataframe[0].value
    assert saved["Dataset"].tolist() == ["FD001/train"]
    assert saved["Engine ID"].tolist() == [5]
    assert saved["Observed cycle"].tolist() == [2]
    assert saved["Predicted RUL (cycles)"].tolist() == [42.0]
    assert saved["Version"].tolist() == ["decision-tree-rul-v1"]
    assert saved["Warnings"].tolist() == ["cycle_gaps"]
    assert "health" not in " ".join(saved.columns).lower()
    assert any("positive synthetic planning assumptions" in item.value for item in planner.caption)

    planner.button[0].click().run()

    assert not planner.exception
    scenario = planner.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY]
    assert scenario.jobs[0].engine.dataset_partition == "FD001/train"
    assert scenario.jobs[0].engine.unit_id == 5
    assert scenario.jobs[0].predicted_rul_cycles == 42.0
    assert (scenario.jobs[0].duration_cycles, scenario.jobs[0].priority_weight) == (1, 1)
    assert (scenario.team_count, scenario.buffer_cycles, scenario.horizon_cycles) == (1, 2, 6)
    comparison = planner.session_state[ui.P20_FLEET_COMPARISON_STATE_KEY]
    assert comparison.edd.weighted_tardiness_cycles == 0
    assert comparison.ga.weighted_tardiness_cycles == 0
    assert comparison.edd.feasible and comparison.ga.feasible
    planning_table = planner.dataframe[1].value
    assert planning_table["Dataset"].tolist() == ["FD001/train"]
    assert planning_table["Engine ID"].tolist() == [5]
    assert planning_table["Planner RUL (cycles)"].tolist() == [42.0]
    assert planner.dataframe[2].value["Engine"].tolist() == ["FD001/train engine 5"]
    assert planner.dataframe[3].value["Engine"].tolist() == ["FD001/train engine 5"]
    assert len(planner.get("plotly_chart")) == 3
    artifact = planner.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY]
    assert artifact["schema_version"] == "p20-paired-schedule-comparison-v1"
    assert artifact["provenance"]["model"]["manifest_sha256"] == "b" * 64
    assert artifact["scenario"]["scenario_sha256"] == comparison.scenario_sha256
    assert [item.label for item in planner.get("download_button")] == [
        "Download paired fleet comparison JSON"
    ]


def test_planner_reports_saved_fleet_prediction_errors(monkeypatch, tmp_path):
    manifest = tmp_path / "artifacts/development/decision_tree_validation.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("test manifest", encoding="utf-8")
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    records = [{"unit_id": 4, "cycle": 2}]
    monkeypatch.setattr(ui, "render_packaged_training_subset", lambda: records)

    @contextmanager
    def use_test_manifest(path, supplied_bundle):
        yield path

    failed_prediction = services.FleetPredictionResult(
        services.EngineIdentity("FD001/train", 4),
        2,
        None,
        None,
        None,
        None,
        None,
        (),
        ("prediction_artifact_error",),
        "model artifact unavailable",
    )
    monkeypatch.setattr(deployment, "prediction_manifest", use_test_manifest)
    monkeypatch.setattr(services, "predict_fleet", lambda *args, **kwargs: (failed_prediction,))

    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=30).run()
    fleet = app.switch_page("pages/1_Fleet_Overview.py").run()
    fleet.multiselect[0].set_value([4]).run()
    fleet.button[0].click().run()
    planner = app.switch_page("pages/4_Maintenance_Planner.py").run()

    planner.button[0].click().run()

    assert not planner.exception
    assert any("no usable prediction" in error.value for error in planner.error)
    assert ui.P20_FLEET_SCENARIO_STATE_KEY not in planner.session_state


def test_planner_explains_empty_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=30).run()

    planner = app.switch_page("pages/4_Maintenance_Planner.py").run()

    assert not planner.exception
    assert not planner.dataframe
    assert any("No fleet is selected yet" in item.value for item in planner.info)
