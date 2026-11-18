"""Maintenance Planner view with the selected fleet snapshot."""

import json
from dataclasses import replace

import plotly.graph_objects as go
import streamlit as st

from fleetpulse import ui
from fleetpulse.optimization.ga import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_SEED,
    MAX_GENERATIONS,
    MAX_POPULATION_SIZE,
    MAX_SEED,
)
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services import (
    EngineIdentity,
    FleetPredictionResult,
    PlannerJobAssumption,
    build_paired_comparison_artifact,
    build_planning_scenario,
    compare_schedules,
)

st.title("Maintenance Planner")
st.caption("Review selected fleet estimates and configure a simulated maintenance scenario.")


def _prediction_from_saved_row(row: dict) -> FleetPredictionResult:
    """Restore a typed prediction record from Fleet Overview's saved snapshot."""
    try:
        identity = EngineIdentity(str(row["Dataset"]), int(row["Engine ID"]))
        warning_text = str(row.get("Warnings") or "")
        error_text = str(row.get("Error") or "")
        return FleetPredictionResult(
            identity=identity,
            observed_cycle=row.get("Observed cycle"),
            raw_rul_cycles=row.get("Raw RUL (cycles)"),
            displayed_rul_cycles=row.get("Predicted RUL (cycles)"),
            model_name=row.get("Model"),
            model_version=row.get("Version"),
            input_window_rows=row.get("Feature rows"),
            warnings=tuple(value.strip() for value in warning_text.split(",") if value.strip()),
            error_codes=("saved_fleet_prediction_error",) if error_text else (),
            error=error_text or None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("A saved fleet row is missing valid engine prediction fields.") from error


def _run_schedule_comparison(
    scenario: SchedulingScenario, *, population_size: int, generations: int, seed: int
):
    """Run both schedulers against the same immutable scenario value."""
    return compare_schedules(
        scenario,
        population_size=population_size,
        generations=generations,
        seed=seed,
    )


def _render_method_result(
    column, result, method_name, labels, *, key_prefix, time_axis_max
):
    report = result.report
    summary = report["summary"]
    column.subheader(method_name)
    column.caption("All times use relative cycles. Both methods use the same scenario.")
    metrics = column.columns(2)
    metrics[0].metric("Weighted tardiness (weighted cycles)", f"{summary['weighted_tardiness_cycles']:g}")
    metrics[1].metric("Total tardiness (cycles)", f"{summary['total_tardiness_cycles']:g}")
    metrics = column.columns(2)
    metrics[0].metric("Makespan (cycles)", f"{summary['makespan_cycles']:g}")
    metrics[1].metric("Horizon overflow (cycles)", f"{summary['horizon_overflow_cycles']:g}")
    metrics = column.columns(2)
    metrics[0].metric("Late jobs", summary["late_job_count"])
    metrics[1].metric("Runtime (seconds)", f"{result.runtime_seconds:.6f}")
    column.dataframe(
        [
            {
                "Engine": labels.get(
                    (row["dataset_partition"], row["unit_id"]),
                    f"{row['dataset_partition']} engine {row['unit_id']}",
                ),
                "Team": row["team_id"],
                "Start (cycles)": row["start_cycle"],
                "Finish (cycles)": row["completion_cycle"],
                "Deadline (cycles)": row["due_completion_cycle"],
                "Late by (cycles)": row["tardiness_cycles"],
                "Beyond horizon (cycles)": row["horizon_overflow_cycles"],
            }
            for row in report["jobs"]
        ],
        hide_index=True,
        width="stretch",
    )
    timeline = go.Figure()
    for row in report["jobs"]:
        engine = labels.get(
            (row["dataset_partition"], row["unit_id"]),
            f"{row['dataset_partition']} engine {row['unit_id']}",
        )
        timeline.add_bar(
            name=engine,
            y=[f"Team {row['team_id']}"],
            x=[row["duration_cycles"]],
            base=[row["start_cycle"]],
            orientation="h",
            text=[engine],
            hovertemplate=(
                f"{engine}<br>Start {row['start_cycle']:g}<br>"
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
        xaxis_range=[0, time_axis_max],
    )
    column.plotly_chart(timeline, width="stretch", key=f"{key_prefix}-timeline")
    search = result.search
    if search is not None:
        column.caption(
            f"GA search: seed {search.seed}, population {search.population_size}, "
            f"{search.generations} generations, {search.evaluations} distinct orders. "
            f"EDD reference weighted tardiness: {search.edd_penalty:g} cycles."
        )
        history = go.Figure(
            go.Scatter(
                x=list(range(len(search.best_penalties))),
                y=list(search.best_penalties),
                mode="lines+markers",
                line_shape="hv",
                name="Best so far",
                hovertemplate="Generation %{x}<br>Weighted tardiness %{y:g}<extra></extra>",
            )
        )
        history.add_hline(y=search.edd_penalty, line_dash="dash", annotation_text="EDD")
        history.update_layout(
            xaxis_title="Generation (0 = initial population)",
            yaxis_title="Best weighted tardiness (weighted cycles)",
            showlegend=False,
        )
        column.plotly_chart(history, width="stretch", key=f"{key_prefix}-history")


def _render_comparison(scenario, comparison, labels, *, heading):
    st.subheader(heading)
    st.caption(
        "These results use one fixed scenario and one identical set of engine jobs. "
        "Fleet RUL values come from the saved model; synthetic-section RUL values are example inputs. "
        "Durations, priority weights, team count, buffer, and horizon are scenario assumptions."
    )
    st.info(
        "Due completion = max(0, planner RUL − buffer). A job is late when it finishes after its "
        "due completion; weighted tardiness adds each job's late cycles times its assumed priority. "
        "The horizon is a soft reporting limit: all selected jobs stay in both schedules, and work "
        "finishing past the horizon is reported as overflow separately, not added to weighted tardiness."
    )
    columns = st.columns(2)
    key_prefix = heading.lower().replace(" ", "-")
    time_axis_max = max(
        comparison.edd.makespan_cycles,
        comparison.ga.makespan_cycles,
        scenario.horizon_cycles,
    )
    _render_method_result(
        columns[0], comparison.edd, "EDD", labels,
        key_prefix=f"{key_prefix}-edd", time_axis_max=time_axis_max,
    )
    _render_method_result(
        columns[1], comparison.ga, "Genetic Algorithm", labels,
        key_prefix=f"{key_prefix}-ga", time_axis_max=time_axis_max,
    )


selected_fleet = st.session_state.get(ui.P15_SELECTED_FLEET_STATE_KEY, [])
if selected_fleet:
    st.success(f"Loaded {len(selected_fleet)} selected engine records from Fleet Overview.")
    st.dataframe(selected_fleet, hide_index=True, width="stretch")
    st.caption(
        "Fleet estimates come from the saved model. Durations and priority weights below are "
        "positive synthetic planning assumptions, not FD001 measurements."
    )

    with st.form("selected_fleet_scenario"):
        settings = st.columns(3)
        fleet_team_count = settings[0].number_input(
            "Fleet planning teams", min_value=1, value=1
        )
        fleet_buffer_cycles = settings[1].number_input(
            "Fleet RUL buffer (cycles)", min_value=0.0, value=2.0
        )
        fleet_horizon_cycles = settings[2].number_input(
            "Fleet soft horizon (cycles)", min_value=0.1, value=6.0
        )
        per_engine_settings = []
        for position, row in enumerate(selected_fleet):
            label = f"{row.get('Dataset', 'unknown')} engine {row.get('Engine ID', position + 1)}"
            duration = st.number_input(
                f"{label} assumed service duration (cycles)",
                min_value=1,
                value=1,
                key=f"p20_duration_{position}_{row.get('Dataset')}_{row.get('Engine ID')}",
            )
            weight = st.number_input(
                f"{label} assumed priority weight",
                min_value=1,
                value=1,
                key=f"p20_weight_{position}_{row.get('Dataset')}_{row.get('Engine ID')}",
            )
            per_engine_settings.append((duration, weight))
        use_selected_fleet = st.form_submit_button("Build selected-fleet planning inputs")

    if use_selected_fleet:
        try:
            fleet_predictions = tuple(_prediction_from_saved_row(row) for row in selected_fleet)
            assumptions = {
                prediction.identity: PlannerJobAssumption(duration, weight)
                for prediction, (duration, weight) in zip(
                    fleet_predictions, per_engine_settings, strict=True
                )
            }
            fleet_scenario = build_planning_scenario(
                fleet_predictions,
                assumptions,
                team_count=fleet_team_count,
                buffer_cycles=fleet_buffer_cycles,
                horizon_cycles=fleet_horizon_cycles,
            )
            comparison = _run_schedule_comparison(
                fleet_scenario,
                population_size=DEFAULT_POPULATION_SIZE,
                generations=DEFAULT_GENERATIONS,
                seed=DEFAULT_SEED,
            )
            provenance = st.session_state.get(ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY)
            if not isinstance(provenance, dict):
                raise TypeError(
                    "Selected fleet provenance is missing; save the engine selection again from Fleet Overview."
                )
            paired_artifact = build_paired_comparison_artifact(
                fleet_predictions,
                fleet_scenario,
                comparison,
                source_provenance=provenance.get("source", {}),
                model_provenance=provenance.get("model", {}),
            )
        except (TypeError, ValueError) as error:
            st.error(f"Selected fleet cannot be planned: {error}")
        else:
            st.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY] = fleet_scenario
            st.session_state[ui.P20_FLEET_COMPARISON_STATE_KEY] = comparison
            st.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY] = paired_artifact
            st.success("Selected fleet predictions and explicit assumptions are ready for scheduling.")

    saved_fleet_scenario = st.session_state.get(ui.P20_FLEET_SCENARIO_STATE_KEY)
    if saved_fleet_scenario is not None:
        st.caption(
            f"{saved_fleet_scenario.team_count} teams; "
            f"{saved_fleet_scenario.buffer_cycles:g}-cycle buffer; "
            f"{saved_fleet_scenario.horizon_cycles:g}-cycle soft horizon. "
            "One saved scenario can be shared by both scheduling methods."
        )
        st.dataframe(
            [
                {
                    "Dataset": job.engine.dataset_partition,
                    "Engine ID": job.engine.unit_id,
                    "Planner RUL (cycles)": job.predicted_rul_cycles,
                    "Assumed duration (cycles)": job.duration_cycles,
                    "Assumed priority weight": job.priority_weight,
                }
                for job in saved_fleet_scenario.jobs
            ],
            hide_index=True,
            width="stretch",
        )
        saved_artifact = st.session_state.get(ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY)
        if saved_artifact is not None:
            st.download_button(
                "Download paired fleet comparison JSON",
                data=json.dumps(saved_artifact, indent=2, allow_nan=False),
                file_name="fleetpulse_paired_schedule_comparison.json",
                mime="application/json",
            )
        fleet_labels = {
            (job.engine.dataset_partition, job.engine.unit_id):
                f"{job.engine.dataset_partition} engine {job.engine.unit_id}"
            for job in saved_fleet_scenario.jobs
        }
        _render_comparison(
            saved_fleet_scenario,
            st.session_state[ui.P20_FLEET_COMPARISON_STATE_KEY],
            fleet_labels,
            heading="Selected fleet: compare schedules",
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
    with st.expander("Advanced genetic algorithm settings"):
        st.caption("Both schedules are compared; these settings control the genetic search.")
        seed = st.number_input("Random seed", 0, MAX_SEED, DEFAULT_SEED)
        population = st.number_input(
            "Population size", 2, MAX_POPULATION_SIZE, DEFAULT_POPULATION_SIZE
        )
        generations = st.number_input("Generations", 0, MAX_GENERATIONS, DEFAULT_GENERATIONS)
    submitted = st.form_submit_button("Save scenario and compare both schedules")

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
        comparison = _run_schedule_comparison(
            scenario, population_size=population, generations=generations, seed=seed
        )
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY] = scenario
        st.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY] = comparison.ga.search
        st.session_state[ui.P20_SYNTHETIC_COMPARISON_STATE_KEY] = comparison

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

    comparison = st.session_state.get(ui.P20_SYNTHETIC_COMPARISON_STATE_KEY)
    if comparison is not None:
        row_labels = {
            (engine.dataset_partition, engine.unit_id): label
            for engine, label in labels.items()
        }
        _render_comparison(
            saved_scenario,
            comparison,
            row_labels,
            heading="Compare schedules on the saved synthetic scenario",
        )
