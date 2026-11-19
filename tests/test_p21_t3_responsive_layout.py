"""The app's column layouts become readable on narrow screens."""

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT


def test_home_registers_narrow_screen_column_stacking():
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=30).run()

    assert not app.exception
    style = next(
        item.value
        for item in app.markdown
        if "max-width: 900px" in item.value and "flex-direction: column" in item.value
    )
    assert "width: 100% !important" in style
