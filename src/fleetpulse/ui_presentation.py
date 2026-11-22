"""Presentation components shared by the Streamlit dashboard and modules."""

from html import escape
from pathlib import Path

import streamlit as st


def eyebrow(label: str) -> None:
    """Render a short context label above a page title."""
    st.markdown(f'<p class="fp-eyebrow">{escape(label)}</p>', unsafe_allow_html=True)


def module_links(*destinations: tuple[Path, str, str]) -> None:
    """Place the next workflow actions side by side on wide screens."""
    with st.container(key="fp_module_actions"):
        for column, (path, label, icon) in zip(st.columns(len(destinations)), destinations):
            column.page_link(st.Page(path), label=label, icon=icon, width="stretch")
