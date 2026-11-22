"""Selections shared between Engine Explorer and RUL Prediction stay explicit."""

import csv
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui
from fleetpulse.data.loader import HISTORY_COLUMNS


def test_engine_cycle_and_model_selection_survive_page_navigation(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "synthetic_engine_history.csv"
    with fixture.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rows += [{**row, "unit_id": "900002"} for row in rows]
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "train_FD001.txt").write_text(
        "\n".join(" ".join(row[field] for field in HISTORY_COLUMNS) for row in rows) + "\n",
        encoding="utf-8",
    )
    development = tmp_path / "artifacts" / "development"
    development.mkdir(parents=True)
    (development / "random_forest_validation.json").write_text("{}", encoding="utf-8")
    (development / "random_forest_model.joblib").write_bytes(b"synthetic placeholder")
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=30).run()
    explorer = app.switch_page("modules/2_Engine_Explorer.py").run()
    explorer.selectbox[0].set_value(900002).run()
    explorer.selectbox[2].set_value(2).run()

    prediction = app.switch_page("modules/3_RUL_Prediction.py").run()
    assert not prediction.exception
    assert prediction.selectbox[0].value == "Local FD001 training data"
    assert prediction.selectbox[1].value == "Decision Tree"
    assert prediction.selectbox[2].value == 900002
    assert prediction.select_slider[0].value == 2
    prediction.selectbox[1].set_value("Random Forest").run()
    assert not prediction.exception

    prediction.selectbox[2].set_value(900001).run()
    prediction.select_slider[0].set_value(1).run()
    explorer = app.switch_page("modules/2_Engine_Explorer.py").run()
    assert not explorer.exception
    assert explorer.selectbox[0].value == 900001
    assert explorer.selectbox[2].value == 1
    prediction = app.switch_page("modules/3_RUL_Prediction.py").run()
    assert not prediction.exception
    assert prediction.selectbox[1].value == "Random Forest"
    assert prediction.session_state[ui.P21_SELECTED_MODEL_STATE_KEY] == "Random Forest"
