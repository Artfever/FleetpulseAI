"""FleetPulse AI landing page."""

from pathlib import Path

import streamlit as st

from fleetpulse import ui
from fleetpulse.data.loader import FD001LoadError, load_engine_history
from fleetpulse.ui_engine_replay import render_aircraft_preview
from fleetpulse.ui_presentation import context_tags, eyebrow
from fleetpulse.visualization import sensor_figure

_PAGES = Path(__file__).resolve().parents[1] / "pages"


def workflow_card(
    group: str, title: str, description: str, filename: str, action: str, icon: str
) -> None:
    with st.container(key=f"fp_card_{group.lower()}", border=True):
        st.markdown(f"### {title}")
        st.caption(description)
        st.page_link(st.Page(_PAGES / filename), label=action, icon=icon)


with st.container(key="fleetpulse_hero"):
    intro, instrument = st.columns([3, 2], vertical_alignment="center")
    with intro:
        eyebrow("NASA C-MAPSS / FD001")
        st.title("FleetPulse AI", anchor=False)
        st.markdown(
            '<p class="fp-lead">Follow an engine from sensor history to a remaining-life '
            "estimate and compare the resulting maintenance schedules.</p>",
            unsafe_allow_html=True,
        )
        context_tags("Simulated engine histories")
        st.page_link(
            st.Page(_PAGES / "2_Engine_Explorer.py"),
            label="Launch 3D engine replay",
            icon=":material/play_circle:",
        )
    with instrument:
        render_aircraft_preview()

st.subheader("Operations dashboard")
with st.container(key="fleetpulse_workflows"):
    overview, explorer, prediction = st.columns(3)
    with overview:
        workflow_card(
            "FLEET",
            "Fleet overview",
            "Review the fleet and compare available engine estimates.",
            "1_Fleet_Overview.py",
            "Open Fleet Overview",
            ":material/grid_view:",
        )
    with explorer:
        workflow_card(
            "TELEMETRY",
            "Engine explorer",
            "Orbit the aircraft in 3D and replay an engine’s recorded sensor history.",
            "2_Engine_Explorer.py",
            "Explore an engine history",
            ":material/timeline:",
        )
    with prediction:
        workflow_card(
            "INFERENCE",
            "Remaining useful life",
            "Estimate remaining cycles from an engine history.",
            "3_RUL_Prediction.py",
            "Open RUL Prediction",
            ":material/insights:",
        )

    planner, lab = st.columns(2)
    with planner:
        workflow_card(
            "SCHEDULING",
            "Maintenance planner",
            "Compare schedules under the same simulated constraints.",
            "4_Maintenance_Planner.py",
            "Open Maintenance Planner",
            ":material/event:",
        )
    with lab:
        workflow_card(
            "EVALUATION",
            "Model laboratory",
            "Inspect evaluation results and the artifacts behind them.",
            "5_Model_Laboratory.py",
            "Review model evidence",
            ":material/science:",
        )


sample_path = ui.PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
try:
    sample = load_engine_history(sample_path)
except FD001LoadError:
    sample = []
if sample:
    engine_ids = sorted({row["unit_id"] for row in sample})
    with st.container(key="fp_signal", border=True):
        st.subheader("Packaged telemetry")
        st.caption("Recorded readings from the packaged FD001 training prefixes.")
        engine_column, sensor_column = st.columns(2)
        engine_id = engine_column.selectbox("Engine", engine_ids, key="fp_preview_engine")
        sensor = sensor_column.selectbox(
            "Sensor channel", ("sensor_9", "sensor_2", "sensor_11"), key="fp_preview_sensor"
        )
        engine_rows = [row for row in sample if row["unit_id"] == engine_id]
        figure = sensor_figure(engine_rows, sensor)
        figure.update_layout(height=282)
        st.plotly_chart(
            figure,
            width="stretch",
            alt=f"{sensor} readings by cycle for packaged training engine {engine_id}",
        )

st.info(
    "Verified model bundles and evaluation reports appear only when matching artifacts are installed."
)
with st.expander("Training data preview"):
    ui.render_training_data()
