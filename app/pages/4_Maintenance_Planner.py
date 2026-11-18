"""Maintenance Planner view with the selected fleet snapshot."""

from dataclasses import replace

import streamlit as st

from fleetpulse import ui
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario

st.title("Maintenance Planner")
st.caption("Review selected fleet estimates and configure a simulated maintenance scenario.")

selected_fleet = st.session_state.get(ui.P15_SELECTED_FLEET_STATE_KEY, [])
if selected_fleet:
    st.success(f"Loaded {len(selected_fleet)} selected engine records from Fleet Overview.")
    st.dataframe(selected_fleet, hide_index=True, width="stretch")
    st.info(
        "These saved estimates are context for a future planner. No maintenance schedule or "
        "health percentage is generated from this selection."
    )
else:
    st.info("No fleet is selected yet.")
    st.write("Select engines on Fleet Overview and save them for the planner.")

st.subheader("Synthetic maintenance scenario")
st.caption("RUL, durations, priorities, and planning limits are assumptions for this simulation.")

with st.form("synthetic_scenario"):
    capacity, buffer, horizon = st.columns(3)
    team_count = capacity.number_input("Maintenance teams", min_value=1, value=1)
    buffer_cycles = buffer.number_input("Buffer (cycles)", min_value=0.0, value=2.0)
    horizon_cycles = horizon.number_input("Soft horizon (cycles)", min_value=0.1, value=6.0)
    jobs = []
    for label, job in zip(("A", "B", "C"), appendix_b_jobs(), strict=True):
        rul, duration, priority = st.columns(3)
        jobs.append(
            replace(
                job,
                predicted_rul_cycles=rul.number_input(
                    f"{label} assumed RUL (cycles)", min_value=0.0, value=job.predicted_rul_cycles
                ),
                duration_cycles=duration.number_input(
                    f"{label} duration (cycles)", min_value=1, value=job.duration_cycles
                ),
                priority_weight=priority.number_input(
                    f"{label} priority weight", min_value=1, value=job.priority_weight
                ),
            )
        )
    submitted = st.form_submit_button("Save synthetic scenario")

if submitted:
    st.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY] = SchedulingScenario(
        tuple(jobs), team_count, buffer_cycles, horizon_cycles
    )

saved_scenario = st.session_state.get(ui.P18_SYNTHETIC_SCENARIO_STATE_KEY)
if saved_scenario is not None:
    st.success(
        f"Saved {len(saved_scenario.jobs)} synthetic jobs, {saved_scenario.team_count} teams, "
        f"a {saved_scenario.buffer_cycles:g}-cycle buffer and "
        f"a {saved_scenario.horizon_cycles:g}-cycle soft horizon."
    )
    st.dataframe(
        [
            {
                "Engine": label,
                "Assumed RUL (cycles)": job.predicted_rul_cycles,
                "Duration (cycles)": job.duration_cycles,
                "Priority weight": job.priority_weight,
            }
            for label, job in zip(("A", "B", "C"), saved_scenario.jobs, strict=True)
        ],
        hide_index=True,
        width="stretch",
    )
