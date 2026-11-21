"""FleetPulse AI view."""

from pathlib import Path

import streamlit as st

from fleetpulse.ui import render_page, render_training_data

render_page(
    "FleetPulse AI",
    "Simulated turbofan maintenance planning.",
    "Verified model bundles and evaluation reports appear only when matching artifacts are installed.",
    "Open Model Laboratory to see which verified results are available in this deployment.",
)
st.subheader("Start exploring")
_app_dir = Path(__file__).resolve().parents[1]
st.page_link(st.Page(_app_dir / "pages" / "2_Engine_Explorer.py"), label="Explore an engine history")
st.page_link(st.Page(_app_dir / "pages" / "5_Model_Laboratory.py"), label="Review model evidence")
render_training_data()
