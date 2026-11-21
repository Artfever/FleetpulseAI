"""FleetPulse AI landing page."""

from pathlib import Path

import streamlit as st

from fleetpulse import ui
from fleetpulse.data.loader import FD001LoadError, load_engine_history
from fleetpulse.data.quality import validate_engine_history
from fleetpulse.ui_engine_replay import check_presenter_estimate, render_aircraft_preview
from fleetpulse.ui_presentation import eyebrow
from fleetpulse.visualization import sensor_figure

_PAGES = Path(__file__).resolve().parents[1] / "pages"


def workflow_card(
    group: str, title: str, description: str, filename: str, action: str, icon: str
) -> None:
    with st.container(key=f"fp_card_{group.lower()}", border=True):
        st.markdown(f"### {title}")
        st.caption(description)
        st.page_link(st.Page(_PAGES / filename), label=action, icon=icon)


sample_path = ui.PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
try:
    sample = load_engine_history(sample_path)
except FD001LoadError:
    sample = []
if sample and not validate_engine_history(sample).is_valid:
    sample = []
engine_ids = sorted({row["unit_id"] for row in sample})

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
        st.caption(
            "For the demo, replay Engine 4, inspect its remaining-life estimate, "
            "then add a team to the sandbox’s worked scheduling example."
        )
        with st.container(horizontal=True, gap="small"):
            with st.container(key="fp_home_primary", width="content"):
                st.page_link(
                    st.Page(_PAGES / "2_Engine_Explorer.py"),
                    label="Launch 3D engine replay",
                    icon=":material/play_circle:",
                    disabled=not sample,
                )
            with st.container(key="fp_home_secondary", width="content"):
                st.page_link(
                    st.Page(_PAGES / "4_Maintenance_Planner.py"),
                    label="Try the scheduling sandbox",
                    icon=":material/event:",
                    query_params={"sandbox": "1"},
                )
    with instrument:
        render_aircraft_preview()

with st.container(key="fp_home_readiness", border=True):
    telemetry, model = st.columns(2)
    telemetry.markdown("**Packaged telemetry**")
    model.markdown("**Engine 4 replay estimate**")
    if sample:
        telemetry.caption(f"Available · {len(engine_ids)} engines, {len(sample)} recorded cycles")
        checkpoint, problem = check_presenter_estimate(ui.PROJECT_ROOT, sample)
        if checkpoint is None:
            model.warning(problem or "The Engine 4 estimate is unavailable.")
        else:
            model.caption(f"Verified at cycle {checkpoint['cycle']} · `{checkpoint['version']}`")
    else:
        telemetry.warning("Packaged FD001 training histories are missing or invalid.")
        model.caption("The estimate check needs the packaged Engine 4 history.")

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


if sample:
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

training = st.expander(
    "Full FD001 training data (optional)", key="fp_home_full_training", on_change="rerun"
)
if training.open:
    with training:
        ui.render_training_data()
