"""Render checked scheduling reports as an interactive 3D timeline."""

import json
from functools import lru_cache

import streamlit as st

from fleetpulse import PROJECT_ROOT
from fleetpulse.services.schedule_comparison import ScheduleComparisonResult

_ASSETS = PROJECT_ROOT / "app" / "planner_replay"


def build_planner_replay_payload(
    comparison: ScheduleComparisonResult, labels: dict[tuple[str, int], str], *, source: str
) -> dict:
    """Expose the checked job timings and score on one shared cycle scale."""
    schedules = {}
    for name, result in (("EDD", comparison.edd), ("GA", comparison.ga)):
        summary = result.report["summary"]
        schedules[name] = {
            "weighted": result.weighted_tardiness_cycles,
            "lateCount": summary["late_job_count"],
            "makespan": result.makespan_cycles,
            "teamCount": summary["team_count"],
            "horizon": summary["horizon_cycles"],
            "jobs": [
                {
                    "id": [row["dataset_partition"], row["unit_id"]],
                    "label": labels.get(
                        (row["dataset_partition"], row["unit_id"]),
                        f"{row['dataset_partition']} engine {row['unit_id']}",
                    ),
                    "team": row["team_id"],
                    "start": row["start_cycle"],
                    "finish": row["completion_cycle"],
                    "due": row["due_completion_cycle"],
                    "late": row["tardiness_cycles"],
                    "weight": row["priority_weight"],
                }
                for row in result.report["jobs"]
            ],
        }
    return {
        "source": source,
        "end": max(max(run["makespan"], run["horizon"]) for run in schedules.values()),
        "schedules": schedules,
    }


@lru_cache(maxsize=1)
def _document() -> str:
    return (
        (_ASSETS / "index.html").read_text(encoding="utf-8")
        .replace("/* STYLES */", (_ASSETS / "planner.css").read_text(encoding="utf-8"))
        .replace("/* SCRIPT */", (_ASSETS / "planner.js").read_text(encoding="utf-8"))
    )


def render_planner_replay(
    comparison: ScheduleComparisonResult, labels: dict[tuple[str, int], str], *, source: str
) -> None:
    payload = build_planner_replay_payload(comparison, labels, source=source)
    data = json.dumps(payload, allow_nan=False).replace("<", r"\u003c")
    st.iframe(
        _document().replace("/* DATA */", data),
        tab_index=0,
        alt="3D maintenance schedule replay with an accessible cycle scrubber and job details",
    )
