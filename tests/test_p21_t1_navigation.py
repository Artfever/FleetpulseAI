"""The app provides a visible route through every principal screen."""

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui


def _links(app):
    return {(link.value, link.proto.page) for link in app.page_link}


def test_six_screen_journey_links_connect_all_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=30).run()

    assert not app.exception
    assert {
        ("Explore an engine history", "Engine_Explorer"),
        ("Review model evidence", "Model_Laboratory"),
    } <= _links(app)

    fleet = app.switch_page("modules/1_Fleet_Overview.py").run()
    assert not fleet.exception
    assert {
        ("Inspect an engine history", "Engine_Explorer"),
        ("Open RUL Prediction", "RUL_Prediction"),
    } <= _links(fleet)

    explorer = app.switch_page("modules/2_Engine_Explorer.py").run()
    assert not explorer.exception
    assert {
        ("Continue to RUL Prediction", "RUL_Prediction"),
        ("Open Fleet Overview", "Fleet_Overview"),
    } <= _links(explorer)

    prediction = app.switch_page("modules/3_RUL_Prediction.py").run()
    assert not prediction.exception
    assert {
        ("Continue to Fleet Overview", "Fleet_Overview"),
        ("Return to Engine Explorer", "Engine_Explorer"),
    } <= _links(prediction)

    planner = app.switch_page("modules/4_Maintenance_Planner.py").run()
    assert not planner.exception
    assert {
        ("Return to Fleet Overview", "Fleet_Overview"),
        ("Review model evidence", "Model_Laboratory"),
    } <= _links(planner)

    laboratory = app.switch_page("modules/5_Model_Laboratory.py").run()
    assert not laboratory.exception
    assert {
        ("Continue to RUL Prediction", "RUL_Prediction"),
        ("Return Home", "home"),
    } <= _links(laboratory)
