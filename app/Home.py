"""Streamlit entrypoint and shared page navigation."""

import streamlit as st

st.set_page_config(page_title="FleetPulse AI", page_icon="\U0001f6e0\ufe0f", layout="wide")
st.markdown(
    """
    <style>
    @media (max-width: 900px) {
      [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
      }
      [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 0 !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

navigation = st.navigation(
    [
        st.Page("views/home.py", title="Home", default=True),
        st.Page("pages/1_Fleet_Overview.py", title="Fleet Overview"),
        st.Page("pages/2_Engine_Explorer.py", title="Engine Explorer"),
        st.Page("pages/3_RUL_Prediction.py", title="RUL Prediction"),
        st.Page("pages/4_Maintenance_Planner.py", title="Maintenance Planner"),
        st.Page("pages/5_Model_Laboratory.py", title="Model Laboratory"),
    ]
)
navigation.run()
