"""Fleet Overview displays ID-attached saved-model predictions and limits."""

import json
import math
from contextlib import contextmanager

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, deployment, services, ui


def _page(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    return AppTest.from_file(
        str(PROJECT_ROOT / "app/modules/1_Fleet_Overview.py"), default_timeout=30
    )


def test_fleet_overview_displays_prediction_identity_cycle_model_and_version(
    monkeypatch, tmp_path
):
    manifest = tmp_path / "artifacts/development/decision_tree_validation.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("test manifest", encoding="utf-8")
    page = _page(monkeypatch, tmp_path)
    records = [
        {"unit_id": 4, "cycle": 164},
        {"unit_id": 5, "cycle": 58},
        {"unit_id": 9, "cycle": 12},
    ]
    monkeypatch.setattr(ui, "render_packaged_training_subset", lambda: records)
    calls = []

    @contextmanager
    def use_test_manifest(path, supplied_bundle):
        assert supplied_bundle is None
        yield path

    def predict(records, **kwargs):
        calls.append((records, kwargs))
        return (
            services.FleetPredictionResult(
                services.EngineIdentity("FD001/train", 4),
                164,
                34.196,
                34.196,
                "Decision Tree",
                "decision-tree-rul-v1",
                30,
                (),
                (),
                None,
            ),
            services.FleetPredictionResult(
                services.EngineIdentity("FD001/train", 5),
                58,
                158.16,
                158.16,
                "Decision Tree",
                "decision-tree-rul-v1",
                30,
                ("cycle_gaps",),
                (),
                None,
            ),
            services.FleetPredictionResult(
                services.EngineIdentity("FD001/train", 9),
                None,
                None,
                None,
                None,
                None,
                None,
                (),
                ("non_finite_reading",),
                "Engine 9 history is invalid.",
            ),
        )

    monkeypatch.setattr(deployment, "prediction_manifest", use_test_manifest)
    monkeypatch.setattr(services, "predict_fleet", predict)

    page.run()

    assert not page.exception
    assert [heading.value for heading in page.title] == ["Fleet Overview"]
    assert calls == [
        (
            records,
            {
                "dataset_partition": "FD001/train",
                "manifest_path": tmp_path / "artifacts/development/decision_tree_validation.json",
            },
        )
    ]
    table = page.dataframe[0].value
    assert list(table) == ["Engine ID", "Observed cycle", "Predicted RUL (cycles)", "Threshold status"]
    assert table["Engine ID"].tolist() == [4, 5, 9]
    assert table["Observed cycle"].tolist()[:2] == [164, 58]
    assert math.isnan(table["Observed cycle"].tolist()[2])
    assert table["Predicted RUL (cycles)"].tolist()[:2] == [34.196, 158.16]
    assert math.isnan(table["Predicted RUL (cycles)"].tolist()[2])
    details = page.dataframe[1].value
    assert details["Model"].tolist()[:2] == ["Decision Tree"] * 2
    assert details["Version"].tolist()[:2] == ["decision-tree-rul-v1"] * 2
    assert details["Fit membership"].tolist() == ["Unknown (split unavailable)"] * 3
    assert details["Warnings"].tolist() == ["", "cycle_gaps", ""]
    assert "non_finite_reading" in details["Error"].tolist()[2]
    assert any("Missing estimates: 1" in item.value for item in page.warning)
    frame = page.get("iframe")[0].proto.srcdoc
    data = json.loads(frame.split('id="payload">', 1)[1].split("</script>", 1)[0])
    assert data["points"] == [
        {"engine": 4, "cycle": 164, "value": 34.196},
        {"engine": 5, "cycle": 58, "value": 158.16},
    ]


def test_fleet_overview_explains_missing_saved_model_without_fake_predictions(
    monkeypatch, tmp_path
):
    page = _page(monkeypatch, tmp_path)
    monkeypatch.setattr(ui, "render_packaged_training_subset", lambda: [{"unit_id": 4, "cycle": 1}])

    def should_not_predict(*args, **kwargs):
        raise AssertionError("prediction should not run without a trusted bundle")

    monkeypatch.setattr(services, "predict_fleet", should_not_predict)

    page.run()

    assert not page.exception
    assert not page.dataframe
    assert not page.get("iframe")
    assert any("Fleet predictions are unavailable" in item.value for item in page.warning)


def test_fleet_overview_sorts_selects_and_filters_without_losing_ids(monkeypatch, tmp_path):
    manifest = tmp_path / "artifacts/development/decision_tree_validation.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("test manifest", encoding="utf-8")
    page = _page(monkeypatch, tmp_path)
    records = [{"unit_id": engine, "cycle": 2} for engine in (4, 5, 9, 12)]
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
            100.0,
            100.0,
            "Decision Tree",
            "decision-tree-rul-v1",
            2,
            (),
            (),
            None,
        ),
        services.FleetPredictionResult(
            services.EngineIdentity("FD001/train", 12),
            2,
            20.0,
            20.0,
            "Decision Tree",
            "decision-tree-rul-v1",
            2,
            ("short_window",),
            (),
            None,
        ),
        services.FleetPredictionResult(
            services.EngineIdentity("FD001/train", 9),
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            ("non_finite_reading",),
            "Engine 9 history is invalid.",
        ),
    )
    monkeypatch.setattr(deployment, "prediction_manifest", use_test_manifest)
    monkeypatch.setattr(services, "predict_fleet", lambda *args, **kwargs: results)

    page.run()
    assert not page.exception
    assert page.dataframe[0].value["Engine ID"].tolist() == [12, 4, 5, 9]
    assert page.metric[0].value == "3 / 4"
    assert page.metric[1].value == "2"
    assert page.metric[2].value == "20.00 cycles"
    figure = json.loads(page.get("plotly_chart")[0].proto.spec)
    assert figure["data"][0]["y"] == ["Engine 12", "Engine 4", "Engine 5"]
    assert page.dataframe[0].value["Threshold status"].tolist() == [
        "At or below threshold",
        "At or below threshold",
        "Above threshold",
        "No estimate",
    ]

    page.number_input[0].set_value(35.0).run()
    page.checkbox[0].set_value(True).run()
    assert page.dataframe[0].value["Engine ID"].tolist() == [12, 4, 9]
    assert page.button(key="fp_fleet_save").label == "Save 4 engines to planner"
    assert any("including 1 hidden by the filter" in item.value for item in page.caption)
    page.button(key="fp_fleet_save").click().run()
    saved = page.session_state[ui.P15_SELECTED_FLEET_STATE_KEY]
    assert [row["Engine ID"] for row in saved] == [4, 5, 12, 9]

    page.multiselect[0].set_value([4, 9]).run()
    filtered = page.dataframe[0].value
    assert filtered["Engine ID"].tolist() == [4, 9]
    assert filtered["Predicted RUL (cycles)"].tolist()[0] == 30.0
    assert "non_finite_reading" in page.dataframe[1].value["Error"].tolist()[1]
