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

    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=30).run()
    fleet = app.switch_page("pages/1_Fleet_Overview.py").run()
    assert not fleet.exception
    fleet.multiselect[0].set_value([5]).run()
    fleet.button[0].click().run()

    assert not fleet.exception
    assert any("Saved 1 selected engine records" in item.value for item in fleet.success)

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
    assert any("No maintenance schedule or health percentage" in item.value for item in planner.info)


def test_planner_explains_empty_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=30).run()

    planner = app.switch_page("pages/4_Maintenance_Planner.py").run()

    assert not planner.exception
    assert not planner.dataframe
    assert any("No fleet is selected yet" in item.value for item in planner.info)
