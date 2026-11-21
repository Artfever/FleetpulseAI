"""Compare changes to a simulated maintenance schedule without replacing its baseline."""

from dataclasses import replace

import plotly.graph_objects as go
import streamlit as st

from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services.schedule_comparison import ScheduleMethodResult, compare_schedules

DEMO_BASELINE = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
_METRICS = (
    ("Weighted tardiness (weighted cycles)", "weighted_tardiness_cycles"),
    ("Total tardiness (cycles)", "total_tardiness_cycles"),
    ("Late jobs", "late_job_count"),
    ("Makespan (cycles)", "makespan_cycles"),
    ("Horizon overflow (cycles)", "horizon_overflow_cycles"),
)


@st.cache_data(show_spinner=False, max_entries=16)
def _compare(scenario: SchedulingScenario):
    return compare_schedules(scenario)


def _timeline(result: ScheduleMethodResult, labels: dict, axis_end: float) -> go.Figure:
    figure = go.Figure()
    teams = sorted({row["team_id"] for row in result.report["jobs"]})
    for row in result.report["jobs"]:
        label = labels[(row["dataset_partition"], row["unit_id"])]
        late = row["tardiness_cycles"]
        figure.add_bar(
            y=[f"Team {row['team_id']}"],
            x=[row["duration_cycles"]],
            base=[row["start_cycle"]],
            text=[label],
            textposition="inside",
            insidetextanchor="middle",
            marker_color="#c99687" if late else "#a8c5df",
            insidetextfont={"color": "#17212b"},
            orientation="h",
            hovertemplate=(
                "%{text}<br>%{y}<br>"
                f"Start: {row['start_cycle']:g}<br>Finish: {row['completion_cycle']:g}<br>"
                f"Due: {row['due_completion_cycle']:g}<br>Late by: {late:g} cycles<extra></extra>"
            ),
        )
    figure.add_vline(x=result.report["summary"]["horizon_cycles"], line_dash="dot")
    figure.update_layout(
        barmode="overlay",
        showlegend=False,
        xaxis_title="Relative operating cycles",
        xaxis_range=[0, axis_end],
        yaxis_autorange="reversed",
        yaxis_categoryorder="array",
        yaxis_categoryarray=[f"Team {team}" for team in teams],
        height=max(230, min(390, 85 + 43 * result.report["summary"]["team_count"])),
        margin={"l": 0, "r": 12, "t": 10, "b": 0},
    )
    return figure


def _job_changes(before: ScheduleMethodResult, after: ScheduleMethodResult, labels: dict) -> list:
    key = ("dataset_partition", "unit_id")
    baseline = {(row[key[0]], row[key[1]]): row for row in before.report["jobs"]}
    changes = []
    for row in after.report["jobs"]:
        identity = (row[key[0]], row[key[1]])
        prior = baseline[identity]
        location = ("team_id", "start_cycle", "completion_cycle")
        moved = any(prior[field] != row[field] for field in location)
        was_late, is_late = prior["tardiness_cycles"] > 0, row["tardiness_cycles"] > 0
        changes.append(
            {
                "Job": labels[identity],
                "Moved": "Yes" if moved else "No",
                "Team before": prior["team_id"],
                "Team after": row["team_id"],
                "Start before": prior["start_cycle"],
                "Start after": row["start_cycle"],
                "Finish before": prior["completion_cycle"],
                "Finish after": row["completion_cycle"],
                "Due before": prior["due_completion_cycle"],
                "Due after": row["due_completion_cycle"],
                "Late by before": prior["tardiness_cycles"],
                "Late by after": row["tardiness_cycles"],
                "Deadline outcome": {
                    (False, True): "Newly late",
                    (True, False): "Now on time",
                    (True, True): "Still late",
                    (False, False): "On time in both",
                }[was_late, is_late],
            }
        )
    return changes


def render_scheduling_sandbox(saved_fleet: SchedulingScenario | None) -> None:
    """Show a live what-if comparison for the worked demo or saved fleet scenario."""
    opened = st.toggle(
        "Open scheduling sandbox",
        value=st.query_params.get("sandbox") == "1",
        key="fp_schedule_sandbox",
    )
    if not opened:
        return

    with st.container(key="fp_sandbox_panel", border=True):
        st.subheader("Scheduling sandbox")
        options = {"Appendix B demo": DEMO_BASELINE}
        if saved_fleet is not None:
            options["Saved fleet scenario"] = saved_fleet
        source = st.radio("Baseline scenario", tuple(options), horizontal=True)
        baseline = options[source]
        baseline_result = _compare(baseline)
        widget_scope = baseline_result.scenario_sha256
        demo = source == "Appendix B demo"
        labels = {
            (job.engine.dataset_partition, job.engine.unit_id): (
                f"Job {chr(65 + i)}" if demo else f"{job.engine.dataset_partition} #{job.engine.unit_id}"
            )
            for i, job in enumerate(baseline.jobs)
        }
        st.caption(
            f"Baseline: {baseline.team_count} teams, {baseline.buffer_cycles:g}-cycle buffer, "
            f"{baseline.horizon_cycles:g}-cycle soft horizon, {len(baseline.jobs)} jobs. "
            "The Appendix B jobs are synthetic. The saved fleet uses model estimates with "
            "assumed service durations and priorities."
        )
        team, buffer = st.columns(2)
        teams = team.number_input(
            "Trial teams", min_value=1, max_value=max(20, baseline.team_count),
            value=2 if demo else baseline.team_count, key=f"fp_sandbox_teams_{widget_scope}",
        )
        reserve = buffer.number_input(
            "Trial buffer (cycles)", min_value=0.0, value=baseline.buffer_cycles,
            step=1.0, key=f"fp_sandbox_buffer_{widget_scope}",
        )
        with st.expander("Change individual job priorities"):
            weights = st.data_editor(
                [
                    {
                        "Job": labels[job.engine.dataset_partition, job.engine.unit_id],
                        "Baseline weight": job.priority_weight,
                        "Trial weight": job.priority_weight,
                    }
                    for job in baseline.jobs
                ],
                disabled=["Job", "Baseline weight"],
                column_config={
                    "Trial weight": st.column_config.NumberColumn(min_value=1, step=1, required=True)
                },
                hide_index=True,
                key=f"fp_sandbox_weights_{widget_scope}",
            )
        try:
            trial = replace(
                baseline, team_count=teams, buffer_cycles=reserve,
                jobs=tuple(
                    replace(job, priority_weight=row["Trial weight"])
                    for job, row in zip(baseline.jobs, weights, strict=True)
                ),
            )
        except ValueError:
            st.error("Use a positive whole number for every trial priority weight.")
            return
        trial_result = _compare(trial)
        st.markdown("#### Compare both schedulers")
        summaries = []
        methods = {"EDD": "edd", "Genetic algorithm": "ga"}
        cards = st.columns(len(methods))
        for card, (method, attribute) in zip(cards, methods.items(), strict=True):
            before_method = getattr(baseline_result, attribute)
            after_method = getattr(trial_result, attribute)
            old_score = before_method.weighted_tardiness_cycles
            new_score = after_method.weighted_tardiness_cycles
            card.metric(
                f"{method} weighted tardiness", f"{old_score:g} → {new_score:g}",
                delta=f"{new_score - old_score:+g} weighted cycles",
                delta_color="inverse",
            )
            before = before_method.report["summary"]
            after = after_method.report["summary"]
            for label, key in _METRICS:
                summaries.append(
                    {
                        "Scheduler": method,
                        "Measure": label,
                        "Baseline": before[key],
                        "Trial": after[key],
                        "Change (trial − baseline)": after[key] - before[key],
                    }
                )
        with st.expander("See all score and timing changes"):
            st.dataframe(summaries, hide_index=True)
        st.caption(
            "Lower weighted tardiness is better. Adjusting the buffer also changes completion "
            "deadlines. Adjusting a priority changes the weights used to score the schedules."
        )
        method = st.radio("Timeline scheduler", tuple(methods), horizontal=True)
        before = getattr(baseline_result, methods[method])
        after = getattr(trial_result, methods[method])
        axis_end = max(before.makespan_cycles, after.makespan_cycles, baseline.horizon_cycles)
        left, right = st.columns(2)
        for column, title, result in ((left, "Baseline", before), (right, "Trial", after)):
            column.markdown(f"#### {title}")
            column.plotly_chart(
                _timeline(result, labels, axis_end), key=f"fp_sandbox_{title.lower()}",
            )
        st.caption("Clay bars finish late. The dotted line is the shared soft horizon.")
        changes = _job_changes(before, after, labels)
        moved = sum(row["Moved"] == "Yes" for row in changes)
        st.markdown(f"#### Job changes · {moved} of {len(changes)} moved")
        st.dataframe(changes, hide_index=True)
