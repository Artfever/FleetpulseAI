"""Exercise an uploaded prediction through the shared maintenance-planner handoff."""

import csv
import io
from contextlib import contextmanager

import pytest
from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, services, ui
from fleetpulse.data.loader import HISTORY_COLUMNS, load_engine_history
from fleetpulse.services import RULPrediction


def _upload_rows():
    return [
        {
            field: unit_id if field == "unit_id" else cycle if field == "cycle" else 0.25
            for field in HISTORY_COLUMNS
        }
        for unit_id in (4, 5)
        for cycle in (1, 2, 3)
    ]


def _upload_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


_TREE_BUNDLE = PROJECT_ROOT / "artifacts" / "development"


@pytest.mark.skipif(
    not (_TREE_BUNDLE / "decision_tree_validation.json").is_file()
    or not (_TREE_BUNDLE / "decision_tree_model.joblib").is_file(),
    reason="the real development Decision Tree bundle is intentionally not tracked",
)
def test_real_uploaded_prefix_reaches_planner_with_saved_model_provenance():
    records = load_engine_history(PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt")
    selected = [row for row in records if row["unit_id"] == 4]
    contents = _upload_bytes(selected)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=60).run()
    upload_page = app.switch_page("modules/3_RUL_Prediction.py").run()
    upload_page.selectbox[0].select("Upload CSV").run()
    upload_page.file_uploader[0].set_value(("fd001_engine_4.csv", contents, "text/csv")).run()
    upload_page.button[0].click().run()

    assert not upload_page.exception
    result = upload_page.session_state[ui.P21_UPLOADED_PREDICTION_STATE_KEY]["prediction"]
    assert result["dataset"] == "user/upload"
    assert result["engine_id"] == 4
    assert result["observed_cycle"] == 164
    assert result["model_version"] == "decision-tree-rul-v1"
    assert result["raw_rul_cycles"] == pytest.approx(34.19607843137255, rel=1e-12)

    upload_page.button[1].click().run()
    planner = app.switch_page("modules/4_Maintenance_Planner.py").run()
    planner.button[0].click().run()
    assert not planner.exception
    scenario = planner.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY]
    artifact = planner.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY]
    assert [(job.engine.dataset_partition, job.engine.unit_id) for job in scenario.jobs] == [
        ("user/upload", 4)
    ]
    assert scenario.jobs[0].predicted_rul_cycles == pytest.approx(34.19607843137255)
    assert artifact["provenance"]["model"]["model_version"] == "decision-tree-rul-v1"
    assert artifact["provenance"]["source"]["source_record_count"] == 164


def test_uploaded_prediction_is_saved_with_provenance_and_reaches_both_schedulers(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)

    @contextmanager
    def use_test_manifest(manifest_path, supplied_bundle):
        assert supplied_bundle is None
        yield manifest_path

    model_provenance = {
        "model_name": "test-model",
        "model_version": "test-v1",
        "manifest_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "feature_config": {"window": 30},
        "feature_schema": ["sensor_1_latest"],
    }
    prediction_calls = []

    def predict_uploaded(contents, **kwargs):
        prediction_calls.append((contents, kwargs))
        return RULPrediction(12.5, 12.5, "test-model", "test-v1", 3, 3, ())

    monkeypatch.setattr(services, "predict_uploaded_csv", predict_uploaded)
    monkeypatch.setattr(services, "model_provenance_from_manifest", lambda path: model_provenance)
    monkeypatch.setattr("fleetpulse.deployment.prediction_manifest", use_test_manifest)

    contents = _upload_bytes(_upload_rows())
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=45).run()
    upload_page = app.switch_page("modules/3_RUL_Prediction.py").run()
    upload_page.selectbox[0].select("Upload CSV").run()
    upload_page.file_uploader[0].set_value(("histories.csv", contents, "text/csv")).run()
    upload_page.selectbox[2].select("4").run()
    upload_page.button[0].click().run()

    assert not upload_page.exception
    assert upload_page.metric[0].value == "12.50"
    assert prediction_calls == [
        (
            contents,
            {
                "dataset_partition": "user/upload",
                "unit_id": 4,
                "manifest_path": tmp_path / "artifacts/development/decision_tree_validation.json",
            },
        )
    ]
    upload_page.button[1].click().run()
    assert not upload_page.exception
    selected = upload_page.session_state[ui.P15_SELECTED_FLEET_STATE_KEY]
    assert selected == [
        {
            "Dataset": "user/upload",
            "Engine ID": 4,
            "Observed cycle": 3,
            "Predicted RUL (cycles)": 12.5,
            "Raw RUL (cycles)": 12.5,
            "Model": "test-model",
            "Version": "test-v1",
            "Feature rows": 3,
            "Warnings": "",
            "Error": None,
        }
    ]
    provenance = upload_page.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY]
    assert provenance["model"] == model_provenance
    assert provenance["source"]["dataset_partition"] == "user/upload"
    assert provenance["source"]["source_record_count"] == 3
    assert len(provenance["source"]["source_records_sha256"]) == 64
    assert provenance["source"]["source_file_sha256"]

    planner = app.switch_page("modules/4_Maintenance_Planner.py").run()
    assert not planner.exception
    planner.button[0].click().run()
    assert not planner.exception
    scenario = planner.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY]
    comparison = planner.session_state[ui.P20_FLEET_COMPARISON_STATE_KEY]
    artifact = planner.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY]
    assert [(job.engine.dataset_partition, job.engine.unit_id) for job in scenario.jobs] == [
        ("user/upload", 4)
    ]
    assert scenario.jobs[0].predicted_rul_cycles == 12.5
    assert comparison.edd.feasible and comparison.ga.feasible
    assert artifact["provenance"]["source"]["dataset_partition"] == "user/upload"
    assert artifact["provenance"]["prediction_snapshot"][0]["raw_rul_cycles"] == 12.5


def test_uploaded_prediction_is_invalidated_when_selected_engine_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    contents = _upload_bytes(_upload_rows())
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app/modules/3_RUL_Prediction.py"), default_timeout=30
    )
    page = app.run()
    page.selectbox[0].select("Upload CSV").run()
    page.file_uploader[0].set_value(("histories.csv", contents, "text/csv")).run()
    page.selectbox[2].select("4").run()
    page.session_state[ui.P21_UPLOADED_PREDICTION_STATE_KEY] = {
        "identity": {
            "upload_sha256": "old",
            "engine_id": 4,
            "model": "Decision Tree",
            "configured_manifest_sha256": None,
        },
        "prediction": {"raw_rul_cycles": 9.0},
    }
    page.session_state[ui.P15_SELECTED_FLEET_STATE_KEY] = [{"Dataset": "user/upload"}]
    page.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY] = {
        "source": {"dataset_partition": "user/upload"}
    }
    page.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY] = object()

    page.selectbox[2].select("5").run()

    assert not page.exception
    assert ui.P21_UPLOADED_PREDICTION_STATE_KEY not in page.session_state
    assert ui.P15_SELECTED_FLEET_STATE_KEY not in page.session_state
    assert ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY not in page.session_state
    assert ui.P20_FLEET_SCENARIO_STATE_KEY not in page.session_state
