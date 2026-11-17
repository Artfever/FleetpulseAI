"""Maintenance Planner view with the selected fleet snapshot."""

import streamlit as st

from fleetpulse import ui

st.title("Maintenance Planner")
st.caption("Review selected fleet predictions before a future scheduling scenario is configured.")

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
