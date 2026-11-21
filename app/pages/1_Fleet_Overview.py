"""Review fleet estimates and send a selected snapshot to Maintenance Planner."""

import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import deployment, ui
from fleetpulse.models.usage_scope import classify_training_engine
from fleetpulse.services import (
    FleetPredictionResult,
    PredictionArtifactError,
    PredictionInputError,
    model_provenance_from_manifest,
    predict_fleet,
)
from fleetpulse.ui_presentation import module_header, module_links

_APP_DIR = Path(__file__).resolve().parents[1]
_PACKAGED = "Packaged FD001/train parity subset"
_LOCAL = "Local FD001 training data"
_CHART_LIMIT = 12
_DISPLAY_COLUMNS = ("Engine ID", "Observed cycle", "Predicted RUL (cycles)", "Threshold status")


def _fit_membership_label(split_manifest: dict | None, unit_id: int) -> str:
    if split_manifest is None:
        return "Unknown (split unavailable)"
    try:
        membership = classify_training_engine(
            split_manifest, dataset_partition="FD001/train", unit_id=unit_id
        )
    except (TypeError, ValueError):
        membership = "unknown"
    return {
        "fitting": "Seen during fitting",
        "validation": "Not seen during fitting (development validation engine)",
        "unknown": "Unknown (not listed in saved split)",
    }[membership]


def _snapshot_row(result: FleetPredictionResult, split_manifest: dict | None) -> dict:
    return {
        "Dataset": result.identity.dataset_partition,
        "Engine ID": result.identity.unit_id,
        "Observed cycle": result.observed_cycle,
        "Fit membership": _fit_membership_label(split_manifest, result.identity.unit_id),
        "Predicted RUL (cycles)": result.displayed_rul_cycles,
        "Raw RUL (cycles)": result.raw_rul_cycles,
        "Model": result.model_name,
        "Version": result.model_version,
        "Feature rows": result.input_window_rows,
        "Warnings": ", ".join(result.warnings),
        "Error": f"{', '.join(result.error_codes)}: {result.error}" if result.error else None,
    }


def _review_status(value: float | None, threshold: float) -> str:
    if value is None:
        return "No estimate"
    return "At or below threshold" if value <= threshold else "Above threshold"


def _load_fleet_snapshot(records, source: str, source_path: Path, details):
    artifact_dir = ui.PROJECT_ROOT / "artifacts" / "development"
    manifest = artifact_dir / "decision_tree_validation.json"
    try:
        supplied_bundle = st.secrets.get("deployment_model")
    except StreamlitSecretNotFoundError:
        supplied_bundle = None
    if not manifest.is_file() and supplied_bundle is None:
        return None, "Fleet predictions are unavailable. Install the verified Decision Tree bundle."

    try:
        with deployment.prediction_manifest(manifest, supplied_bundle) as active_manifest:
            predictions = predict_fleet(
                records, dataset_partition="FD001/train", manifest_path=active_manifest
            )
            try:
                model_provenance = model_provenance_from_manifest(active_manifest)
            except ValueError as error:
                model_provenance = None
                details.warning(f"Comparison provenance is unavailable: {error}")
    except (PredictionArtifactError, PredictionInputError, ValueError) as error:
        return None, f"Fleet predictions are unavailable: {error}"

    split_path = ui.PROJECT_ROOT / "artifacts" / "split_manifest.json"
    try:
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(split_manifest, dict):
            raise TypeError("the saved split manifest must be an object")
    except (OSError, ValueError, TypeError) as error:
        split_manifest = None
        details.warning(f"Development split membership is unavailable: {error}")
    canonical_records = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    source_provenance = {
        "source_name": source,
        "source_path": source_path.relative_to(ui.PROJECT_ROOT).as_posix(),
        "dataset_partition": "FD001/train",
        "source_records_sha256": hashlib.sha256(canonical_records).hexdigest(),
        "source_record_count": len(records),
        "source_file_sha256": (
            hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
        ),
    }
    rows = [_snapshot_row(result, split_manifest) for result in predictions]
    return (rows, {"source": source_provenance, "model": model_provenance}), None


def _ranking(rows: list[dict], threshold: float) -> go.Figure:
    estimates = [row["Predicted RUL (cycles)"] for row in rows]
    engine_labels = [f"Engine {row['Engine ID']}" for row in rows]
    axis_end = max(max(estimates) * 1.18, 1.0)
    fig = go.Figure(
        go.Bar(
            x=estimates,
            y=engine_labels,
            orientation="h",
            marker_color=["#d5ae77" if value <= threshold else "#a8c5df" for value in estimates],
            text=[f"{value:.2f}" for value in estimates],
            textposition="outside",
            customdata=[row["Observed cycle"] for row in rows],
            hovertemplate=(
                "%{y}<br>Observed cycle %{customdata}<br>Estimate %{x:.2f} cycles<extra></extra>"
            ),
        )
    )
    if threshold <= axis_end:
        fig.add_vline(x=threshold, line_dash="dot")
    fig.update_layout(
        height=max(200, min(405, 105 + 25 * len(rows))),
        margin={"l": 0, "r": 19, "t": 12, "b": 30},
        xaxis={"title": "Estimated remaining cycles", "range": [0, axis_end]},
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    return fig


def _save_selection(selected: list[dict], provenance: dict) -> None:
    count = len(selected)
    noun = "engine" if count == 1 else "engines"
    if st.button(
        f"Save {count} {noun} to planner", key="fp_fleet_save", type="primary", disabled=not count
    ):
        st.session_state[ui.P15_SELECTED_FLEET_STATE_KEY] = selected
        st.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY] = provenance
        st.session_state.pop(ui.P20_FLEET_SCENARIO_STATE_KEY, None)
        st.session_state.pop(ui.P20_FLEET_COMPARISON_STATE_KEY, None)
        st.session_state.pop(ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY, None)
        st.success(f"Saved {count} selected engine records for the planner.")
        st.page_link(
            st.Page(_APP_DIR / "pages" / "4_Maintenance_Planner.py"),
            label="Open Maintenance Planner",
            icon=":material/event:",
        )


def _render_snapshot(rows: list[dict], provenance: dict, details) -> None:
    selection, review = st.columns([2, 1])
    engine_ids = [row["Engine ID"] for row in rows]
    selected_ids = selection.multiselect("Engine IDs", engine_ids, default=engine_ids)
    threshold = review.number_input("Review threshold (cycles)", min_value=0.0, value=50.0, step=5.0)
    threshold_only = st.checkbox(
        "Show only estimates at or below the threshold",
        help="Rows without an estimate stay visible for review.",
    )

    selected_set = set(selected_ids)
    chosen = [row for row in rows if row["Engine ID"] in selected_set]
    estimated = sorted(
        (row for row in chosen if row["Predicted RUL (cycles)"] is not None),
        key=lambda row: (row["Predicted RUL (cycles)"], row["Engine ID"]),
    )
    without_estimate = sorted(
        (row for row in chosen if row["Predicted RUL (cycles)"] is None),
        key=lambda row: row["Engine ID"],
    )
    below = [row for row in estimated if row["Predicted RUL (cycles)"] <= threshold]
    shown_estimates = below if threshold_only else estimated
    visible = shown_estimates + without_estimate

    counts = st.columns(3)
    counts[0].metric("Selected with estimates", f"{len(estimated)} / {len(chosen)}")
    counts[1].metric("At or below threshold", len(below))
    lowest = estimated[0] if estimated else None
    if lowest:
        counts[2].metric(
            f"Lowest estimate · Engine {lowest['Engine ID']}",
            f"{lowest['Predicted RUL (cycles)']:.2f} cycles",
        )
    else:
        counts[2].metric("Lowest estimate", "—")

    if shown_estimates:
        st.subheader("Remaining life by engine")
        if len(shown_estimates) > _CHART_LIMIT:
            st.caption(f"Showing the {_CHART_LIMIT} lowest visible estimates. All rows are below.")
        st.plotly_chart(
            _ranking(shown_estimates[:_CHART_LIMIT], threshold),
            key="fp_fleet_ranking",
            alt="Selected engine estimates ordered from lowest to highest remaining cycles",
        )

    if visible:
        visible_rows = [
            {**row, "Threshold status": _review_status(row["Predicted RUL (cycles)"], threshold)}
            for row in visible
        ]
        st.dataframe(
            [{column: row[column] for column in _DISPLAY_COLUMNS} for row in visible_rows],
            hide_index=True,
            column_config={
                "Predicted RUL (cycles)": st.column_config.NumberColumn(
                    "Estimated RUL (cycles)", format="%.2f",
                ),
                "Threshold status": "Review status",
            },
        )
        details.dataframe(visible_rows, hide_index=True)
    else:
        st.info("No engines match the current selection and filter.")
    if without_estimate:
        st.warning(f"Missing estimates: {len(without_estimate)}. Check Source and model details.")
    hidden = len(chosen) - len(visible)
    if hidden:
        st.caption(f"The save action includes all selected engines, including {hidden} hidden by the filter.")
    elif chosen:
        st.caption("The save action includes all selected engines.")
    _save_selection(chosen, provenance)


module_header(
    "fleet",
    "FD001 / Fleet",
    "Fleet Overview",
    "Review simulated remaining-life estimates and select engines for the planner.",
    "Saved Decision Tree",
)
source = st.selectbox("History source", [_PACKAGED, _LOCAL])
overview = st.container()
details = st.expander("Source and model details")
with details:
    st.caption(
        "The estimate is in operating cycles. The review threshold is a display filter for "
        "simulated histories and does not establish a physical fault or a maintenance rule."
    )
    if source == _PACKAGED:
        source_path = ui.PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
        records = ui.render_packaged_training_subset()
    else:
        source_path = ui.PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
        records = ui.render_training_data()

with overview:
    if not records:
        st.warning("No validated records for this source. Check Source and model details.")
    else:
        snapshot, problem = _load_fleet_snapshot(records, source, source_path, details)
        if snapshot is None:
            st.warning(problem)
        else:
            _render_snapshot(*snapshot, details)

module_links(
    (_APP_DIR / "pages" / "2_Engine_Explorer.py", "Inspect an engine history", ":material/timeline:"),
    (_APP_DIR / "pages" / "3_RUL_Prediction.py", "Open RUL Prediction", ":material/insights:"),
)
