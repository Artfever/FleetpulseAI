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
    {
        "Workspace": [
            st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
            st.Page("modules/1_Fleet_Overview.py", title="Fleet Overview", icon=":material/space_dashboard:"),
            st.Page(
                "modules/2_Engine_Explorer.py", title="Engine Explorer", icon=":material/view_in_ar:"
            ),
        ],
        "Analysis": [
            st.Page("modules/3_RUL_Prediction.py", title="RUL Prediction", icon=":material/query_stats:"),
            st.Page(
                "modules/4_Maintenance_Planner.py",
                title="Maintenance Planner",
                icon=":material/event_available:",
            ),
            st.Page("modules/5_Model_Laboratory.py", title="Model Laboratory", icon=":material/biotech:"),
        ],
    },
)
st.sidebar.markdown(
    '<div class="fp-sidebar-brand"><span class="fp-brand-mark" aria-hidden="true">FP</span>'
    '<div><strong>FleetPulse AI</strong><small>FD001 simulation</small></div></div>',
    unsafe_allow_html=True,
)
navigation.run()
