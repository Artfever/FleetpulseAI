"""Streamlit upload UI renders validated previews and actionable errors."""

import csv
import io
from contextlib import contextmanager

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, deployment, services, ui
from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.services import PredictionInputError, RULPrediction


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _row(unit_id, cycle):
    return {
        field: unit_id if field == "unit_id" else cycle if field == "cycle" else 0.25
        for field in HISTORY_COLUMNS
    }


def _page(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    return AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/3_RUL_Prediction.py"), default_timeout=30
    )


def test_upload_page_selects_engine_and_previews_only_its_rows_without_prediction(
    monkeypatch, tmp_path
):
    page = _page(monkeypatch, tmp_path)
    prediction_calls = []
    monkeypatch.setattr(
        services,
        "predict_rul_with_context",
        lambda *args, **kwargs: prediction_calls.append((args, kwargs)),
    )
    contents = _csv_bytes([_row(4, 1), _row(5, 1), _row(4, 2), _row(5, 2)])

    page.run()
    page.selectbox[0].select("Upload CSV").run()
    page.file_uploader[0].set_value(("histories.csv", contents, "text/csv")).run()

    assert not page.exception
    assert not page.error
    assert [widget.label for widget in page.selectbox] == ["History source", "RUL model", "Uploaded engine ID"]
    assert page.selectbox[2].options == ["4", "5"]
    page.selectbox[2].select("5").run()

    preview = page.dataframe[0].value
    assert set(preview["unit_id"]) == {5}
    assert list(preview["cycle"]) == [1.0, 2.0]
    assert any("2 of 4 rows" in element.value for element in page.success)
    assert prediction_calls == []


def test_upload_error_is_clear_and_clearing_file_clears_error(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    invalid = (PROJECT_ROOT / "tests/fixtures/p14_uploads/text_sensor.csv").read_bytes()

    page.run()
    page.selectbox[0].select("Upload CSV").run()
    page.file_uploader[0].set_value(("bad.csv", invalid, "text/csv")).run()

    assert not page.exception
    assert any("CSV upload rejected (invalid_number)" in item.value for item in page.error)
    assert any("sensor_1 must contain a numeric value" in item.value for item in page.error)
    assert not page.dataframe

    page.file_uploader[0].set_value(None).run()
    assert not page.exception
    assert not page.error


def test_upload_prediction_uses_shared_service_and_shows_short_history_limits(
    monkeypatch, tmp_path
):
    page = _page(monkeypatch, tmp_path)
    contents = _csv_bytes([_row(4, 1), _row(5, 1), _row(4, 2), _row(5, 2)])
    calls = []

    @contextmanager
    def use_test_manifest(manifest_path, supplied_bundle):
        assert supplied_bundle is None
        yield manifest_path

    def predict_uploaded(upload_bytes, **kwargs):
        calls.append((upload_bytes, kwargs))
        return RULPrediction(
            12.5, 12.5, "test-model", "test-v1", 2, 2, ("short_window", "cycle_gaps")
        )

    monkeypatch.setattr(deployment, "prediction_manifest", use_test_manifest)
    monkeypatch.setattr(services, "predict_uploaded_csv", predict_uploaded)

    page.run()
    page.selectbox[0].select("Upload CSV").run()
    page.file_uploader[0].set_value(("histories.csv", contents, "text/csv")).run()
    page.selectbox[2].select("5").run()
    assert any("outside the model's domain" in item.value for item in page.info)
    page.button[0].click().run()

    assert not page.exception
    assert calls == [
        (
            contents,
            {
                "dataset_partition": "user/upload",
                "unit_id": 5,
                "manifest_path": tmp_path / "artifacts/development/decision_tree_validation.json",
            },
        )
    ]
    assert page.metric[0].value == "12.50"
    assert any("test-model" in item.value and "test-v1" in item.value for item in page.markdown)
    assert any("short" in item.value for item in page.warning)
    assert any("gaps" in item.value for item in page.warning)


def test_upload_prediction_errors_are_shown_without_a_result(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    contents = _csv_bytes([_row(4, 1)])

    @contextmanager
    def use_test_manifest(manifest_path, supplied_bundle):
        yield manifest_path

    def fail_prediction(*args, **kwargs):
        raise PredictionInputError("selected history cannot be predicted")

    monkeypatch.setattr(deployment, "prediction_manifest", use_test_manifest)
    monkeypatch.setattr(services, "predict_uploaded_csv", fail_prediction)

    page.run()
    page.selectbox[0].select("Upload CSV").run()
    page.file_uploader[0].set_value(("history.csv", contents, "text/csv")).run()
    page.button[0].click().run()

    assert not page.exception
    assert not page.metric
    assert any(
        "Prediction is unavailable" in item.value
        and "selected history cannot be predicted" in item.value
        for item in page.warning
    )
