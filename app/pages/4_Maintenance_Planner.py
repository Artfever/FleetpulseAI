"""Maintenance Planner view with the selected fleet snapshot."""

from dataclasses import replace

import plotly.graph_objects as go
import streamlit as st

from fleetpulse import ui
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.ga import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_SEED,
    MAX_GENERATIONS,
    MAX_POPULATION_SIZE,
    MAX_SEED,
    schedule_ga,
)
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.reporting import build_schedule_report
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

examples = dict(zip(("A", "B", "C"), appendix_b_jobs(), strict=True))
labels = {job.engine: label for label, job in examples.items()}
selected_engines = st.multiselect("Scenario engines", tuple(examples), default=tuple(examples))

with st.form("synthetic_scenario"):
    capacity, buffer, horizon = st.columns(3)
    team_count = capacity.number_input(
        "Maintenance teams", min_value=0, value=1, help="At least one team is required."
    )
    buffer_cycles = buffer.number_input("Buffer (cycles)", min_value=0.0, value=2.0)
    horizon_cycles = horizon.number_input("Soft horizon (cycles)", min_value=0.1, value=6.0)
    job_inputs = []
    for label, job in examples.items():
        if label not in selected_engines:
            continue
        rul, duration, priority = st.columns(3)
        job_inputs.append(
            (
                job,
                rul.number_input(
                    f"{label} assumed RUL (cycles)", min_value=0.0, value=job.predicted_rul_cycles
                ),
                duration.number_input(
                    f"{label} duration (cycles)", min_value=1, value=job.duration_cycles
                ),
                priority.number_input(
                    f"{label} priority weight", min_value=1, value=job.priority_weight
                ),
            )
        )
    method = st.selectbox("Scheduling method", ("EDD", "Genetic Algorithm"))
    with st.expander("Advanced genetic algorithm settings"):
        st.caption("These settings apply when Genetic Algorithm is selected.")
        seed = st.number_input("Random seed", 0, MAX_SEED, DEFAULT_SEED)
        population = st.number_input(
            "Population size", 2, MAX_POPULATION_SIZE, DEFAULT_POPULATION_SIZE
        )
        generations = st.number_input("Generations", 0, MAX_GENERATIONS, DEFAULT_GENERATIONS)
    submitted = st.form_submit_button("Save synthetic scenario")

if submitted:
    try:
        if not job_inputs:
            raise ValueError("Select at least one engine.")
        if team_count < 1:
            raise ValueError("Enter at least one maintenance team.")
        jobs = tuple(
            replace(
                job, predicted_rul_cycles=rul, duration_cycles=duration, priority_weight=priority
            )
            for job, rul, duration, priority in job_inputs
        )
        scenario = SchedulingScenario(jobs, team_count, buffer_cycles, horizon_cycles)
        search_result = (
            schedule_ga(scenario, population_size=population, generations=generations, seed=seed)
            if method == "Genetic Algorithm"
            else None
        )
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY] = scenario
        st.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY] = search_result

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
                "Engine": labels[job.engine],
                "Assumed RUL (cycles)": job.predicted_rul_cycles,
                "Duration (cycles)": job.duration_cycles,
                "Priority weight": job.priority_weight,
            }
            for job in saved_scenario.jobs
        ],
        hide_index=True,
        width="stretch",
    )

    saved_search = st.session_state.get(ui.P19_GENETIC_SEARCH_STATE_KEY)
    schedule = saved_search.schedule if saved_search is not None else schedule_edd(saved_scenario)
    report = build_schedule_report(saved_scenario, schedule)
    summary = report["summary"]
    row_labels = {
        (engine.dataset_partition, engine.unit_id): label for engine, label in labels.items()
    }
    if saved_search is None:
        st.subheader("EDD baseline schedule")
        st.caption("Earliest completion deadline first. All timings are simulated relative cycles.")
    else:
        st.subheader("Genetic algorithm schedule")
        st.caption("Best schedule found within the selected budget. Timings are simulated cycles.")
    cost, makespan, overflow = st.columns(3)
    cost.metric("Weighted tardiness", f"{summary['weighted_tardiness_cycles']:g}")
    makespan.metric("Makespan (cycles)", f"{summary['makespan_cycles']:g}")
    overflow.metric("Horizon overflow (cycles)", f"{summary['horizon_overflow_cycles']:g}")
    st.dataframe(
        [
            {
                "Engine": row_labels[row["dataset_partition"], row["unit_id"]],
                "Team": row["team_id"],
                "Start": row["start_cycle"],
                "Finish": row["completion_cycle"],
                "Deadline": row["due_completion_cycle"],
                "Late by": row["tardiness_cycles"],
                "Past horizon": row["horizon_overflow_cycles"],
            }
            for row in report["jobs"]
        ],
        hide_index=True,
        width="stretch",
    )

    timeline = go.Figure()
    for row in report["jobs"]:
        label = row_labels[row["dataset_partition"], row["unit_id"]]
        timeline.add_bar(
            name=label,
            y=[f"Team {row['team_id']}"],
            x=[row["duration_cycles"]],
            base=[row["start_cycle"]],
            orientation="h",
            text=[label],
            hovertemplate=(
                f"Engine {label}<br>Start {row['start_cycle']:g}<br>"
                f"Finish {row['completion_cycle']:g}<extra></extra>"
            ),
        )
    timeline.add_vline(
        x=summary["horizon_cycles"], line_dash="dash", annotation_text="Soft horizon"
    )
    timeline.update_layout(
        barmode="overlay",
        xaxis_title="Relative cycles",
        yaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(timeline, width="stretch")

    if saved_search is not None:
        st.subheader("Search progress")
        st.write(
            f"Seed {saved_search.seed} ? Population {saved_search.population_size} ? "
            f"{saved_search.generations} generations ? {saved_search.evaluations} distinct orders "
            f"evaluated in {saved_search.runtime_seconds:.3f} s"
        )
        st.caption(
            f"EDD reference score is {saved_search.edd_penalty:g}. "
            "Each generation keeps the best order found so far. A larger search can sometimes "
            "find a better schedule."
        )
        history = go.Figure(
            go.Scatter(
                x=list(range(len(saved_search.best_penalties))),
                y=list(saved_search.best_penalties),
                mode="lines+markers",
                line_shape="hv",
                name="Best so far",
                hovertemplate="Generation %{x}<br>Weighted tardiness %{y:g}<extra></extra>",
            )
        )
        history.add_hline(y=saved_search.edd_penalty, line_dash="dash", annotation_text="EDD")
        history.update_layout(
            xaxis_title="Generation (0 = initial population)",
            yaxis_title="Best weighted tardiness",
            showlegend=False,
        )
        st.plotly_chart(history, width="stretch")
