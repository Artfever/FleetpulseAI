"""Streamlit entrypoint and shared page navigation."""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="FleetPulse AI",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)
stylesheet = Path(__file__).with_name("fleetpulse.css").read_text(encoding="utf-8")
st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)

navigation = st.navigation(
    [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("pages/1_Fleet_Overview.py", title="Fleet Overview", icon=":material/grid_view:"),
        st.Page("pages/2_Engine_Explorer.py", title="Engine Explorer", icon=":material/timeline:"),
        st.Page("pages/3_RUL_Prediction.py", title="RUL Prediction", icon=":material/insights:"),
        st.Page("pages/4_Maintenance_Planner.py", title="Maintenance Planner", icon=":material/event:"),
        st.Page("pages/5_Model_Laboratory.py", title="Model Laboratory", icon=":material/science:"),
    ]
)
st.sidebar.markdown(
    '<div class="fp-sidebar-brand"><span>FLEETPULSE AI</span>'
    '<small>NASA C-MAPSS / FD001 simulation</small></div>',
    unsafe_allow_html=True,
)
navigation.run()
