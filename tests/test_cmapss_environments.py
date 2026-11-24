"""Exercise the released subset bundles and their normal Streamlit entry point."""

import csv
import io
import json
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui
from fleetpulse.data.loader import HISTORY_COLUMNS, load_engine_history
from fleetpulse.services.environments import NEW_ENVIRONMENTS, load_environment_model, sha256

BUNDLES = PROJECT_ROOT / "artifacts" / "environments"
DEMO = PROJECT_ROOT / "data" / "demo"


def _rows(subset):
    return load_engine_history(DEMO / f"{subset.lower()}_training_prefix.txt")


@pytest.fixture
def xgboost():
    return pytest.importorskip("xgboost")


@pytest.mark.parametrize("subset", NEW_ENVIRONMENTS)
def test_release_bundle_uses_matching_report_and_no_future_features(subset, xgboost):
    directory = BUNDLES / subset
    model = load_environment_model(directory, subset)
    report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    assert report["bundle_sha256"] == sha256(directory / "manifest.json")
    assert report["model_sha256"] == sha256(directory / "model.json")
    assert report["model"]["engines"] == {"FD002": 259, "FD003": 100, "FD004": 248}[subset]
    rows = _rows(subset)
    (prefix_result,) = model.predict(rows[:50], [50])
    future = [{**row, "sensor_1": 1e10, "setting_1": 50.0} for row in rows[50:]]
    (full_result,) = model.predict([*rows[:50], *future], [50])
    assert prefix_result == full_result
    assert prefix_result[0].window_row_count == 30


def test_loader_rejects_wrong_subset_and_corrupt_model(tmp_path, xgboost):
    shutil.copytree(BUNDLES / "FD002", tmp_path, dirs_exist_ok=True)
    with pytest.raises(ValueError, match="metadata"):
        load_environment_model(tmp_path, "FD004")
    with (tmp_path / "model.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_environment_model(tmp_path, "FD002")


def _csv(subset):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(_rows(subset)[:40])
    return output.getvalue().encode()


def test_prediction_page_routes_switches_and_validates_uploads(xgboost):
    page = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "modules" / "3_RUL_Prediction.py"), default_timeout=45
    ).run()
    assert page.radio(key="fp_rul_env").value == "FD001"
    for subset in NEW_ENVIRONMENTS:
        page.radio(key="fp_rul_env").set_value(subset).run()
        assert not page.exception
        assert not page.error
        assert page.text_input[0].value == f"XGBoost · {subset}"
        ((_, raw),) = load_environment_model(BUNDLES / subset, subset).predict(_rows(subset), [100])
        assert page.metric[0].value == f"{max(0, raw):.2f}"

    page.button(key="FD004_save").click().run()
    assert page.session_state[ui.P15_SELECTED_FLEET_STATE_KEY][0]["Dataset"] == "FD004/train"
    page.radio(key="fp_rul_env").set_value("FD002").run()
    assert ui.P15_SELECTED_FLEET_STATE_KEY not in page.session_state
    page.selectbox(key="FD002_source").select("Upload CSV").run()
    assert not page.metric
    page.file_uploader[0].set_value(("bad.csv", b"bad,header\n1,2", "text/csv")).run()
    assert page.error and not page.metric
    page.file_uploader[0].set_value(("good.csv", _csv("FD002"), "text/csv")).run()
    ((_, raw),) = load_environment_model(BUNDLES / "FD002", "FD002").predict(
        _rows("FD002")[:40], [40]
    )
    assert not page.exception and not page.error
    assert page.metric[0].value == f"{max(0, raw):.2f}"
    page.button(key="FD002_save").click().run()
    assert page.session_state[ui.P15_SELECTED_FLEET_STATE_KEY][0]["Dataset"] == "FD002/upload"
    page.radio(key="fp_rul_env").set_value("FD001").run()
    assert not page.exception
    assert ui.P15_SELECTED_FLEET_STATE_KEY not in page.session_state


def test_uploaded_subset_reaches_both_planner_schedulers(xgboost):
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=60).run()
    page = app.switch_page("modules/3_RUL_Prediction.py").run()
    page.radio(key="fp_rul_env").set_value("FD003").run()
    page.selectbox(key="FD003_source").select("Upload CSV").run()
    page.file_uploader[0].set_value(("fd003.csv", _csv("FD003"), "text/csv")).run()
    page.button(key="FD003_save").click().run()

    planner = app.switch_page("modules/4_Maintenance_Planner.py").run()
    planner.button[0].click().run()
    assert not planner.exception
    scenario = planner.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY]
    artifact = planner.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY]
    (job,) = scenario.jobs
    assert (job.engine.dataset_partition, job.engine.unit_id) == ("FD003/upload", 1)
    assert artifact["provenance"]["model"]["model_version"] == "fd003-xgboost-v1"
    assert artifact["provenance"]["source"]["source_record_count"] == 40


def test_missing_bundle_is_visible_and_produces_no_result(monkeypatch, tmp_path, xgboost):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    page = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "modules" / "3_RUL_Prediction.py"), default_timeout=45
    ).run()
    page.radio(key="fp_rul_env").set_value("FD003").run()
    assert not page.exception
    assert "FD003 model unavailable" in page.error[0].value
    assert not page.metric
