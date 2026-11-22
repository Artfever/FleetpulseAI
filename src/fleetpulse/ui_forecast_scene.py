"""Share a 3D viewport between fleet estimates and observed prediction checkpoints."""

import json
from collections.abc import Sequence
from functools import lru_cache

import streamlit as st

from fleetpulse import PROJECT_ROOT

_ASSETS = PROJECT_ROOT / "app" / "forecast_scene"


@lru_cache(maxsize=1)
def _document() -> str:
    return (
        (_ASSETS / "index.html").read_text(encoding="utf-8")
        .replace("/* STYLES */", (_ASSETS / "scene.css").read_text(encoding="utf-8"))
        .replace("/* SCRIPT */", (_ASSETS / "scene.js").read_text(encoding="utf-8"))
    )


def render_fleet_deck(
    estimates: Sequence[tuple[int, int, float]], *, threshold: float, total: int
) -> None:
    """Show ranked estimates without changing the displayed or saved fleet."""
    _render(
        {
            "mode": "fleet",
            "threshold": threshold,
            "total": total,
            "points": [
                {"engine": engine, "cycle": cycle, "value": value}
                for engine, cycle, value in estimates
            ],
        },
        description="3D ranked fleet estimates, with the review threshold in cycles",
    )


def render_prediction_trace(
    estimate_points: Sequence[tuple[int, float]], *, engine: int
) -> None:
    """Show exactly the observed cutoffs evaluated by the prediction service."""
    _render(
        {
            "mode": "prediction",
            "points": [
                {"engine": engine, "cycle": cycle, "value": value}
                for cycle, value in estimate_points
            ],
        },
        description="3D replay of saved-model estimates at observed cycle checkpoints",
    )


def _render(payload: dict, *, description: str) -> None:
    data = json.dumps(payload, allow_nan=False).replace("<", r"\u003c")
    st.iframe(_document().replace("/* DATA */", data), tab_index=0, alt=description)
