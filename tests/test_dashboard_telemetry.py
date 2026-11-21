"""The dashboard preview follows the selected packaged engine and sensor."""

import json

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history


def test_dashboard_preview_uses_the_selected_engine_and_sensor():
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=30).run()
    app.selectbox(key="fp_preview_engine").set_value(12).run()
    app.selectbox(key="fp_preview_sensor").set_value("sensor_2").run()

    assert not app.exception
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    trace = figure["data"][0]
    records = load_engine_history(PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt")
    expected = sorted(
        (row for row in records if row["unit_id"] == 12), key=lambda row: row["cycle"]
    )
    assert trace["x"] == [row["cycle"] for row in expected]
    assert trace["y"] == [row["sensor_2"] for row in expected]
