"""Show saved-model fleet predictions with their engine identities."""

import hashlib
import json
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import deployment, ui
from fleetpulse.models.usage_scope import classify_training_engine
from fleetpulse.services import (
    PredictionArtifactError,
    PredictionInputError,
    model_provenance_from_manifest,
    predict_fleet,
)

st.title("Fleet Overview")
st.caption(
    "Review saved-model RUL estimates for observed engine histories. Estimates are in operating "
    "cycles and do not confirm physical faults or maintenance needs."
)
_app_dir = Path(__file__).resolve().parents[1]
st.page_link(st.Page(_app_dir / "pages" / "2_Engine_Explorer.py"), label="Inspect an engine history")
st.page_link(st.Page(_app_dir / "pages" / "3_RUL_Prediction.py"), label="Open RUL Prediction")


def _fit_membership_label(split_manifest: dict | None, unit_id: int) -> str:
    """Describe saved development-split membership without inferring from absence."""
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

source = st.selectbox(
    "History source",
    ["Packaged FD001/train parity subset", "Local FD001 training data"],
)
if source == "Packaged FD001/train parity subset":
    source_path = ui.PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
    records = ui.render_packaged_training_subset()
else:
    source_path = ui.PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
    records = ui.render_training_data()

if records:
    artifact_dir = ui.PROJECT_ROOT / "artifacts" / "development"
    manifest_path = artifact_dir / "decision_tree_validation.json"
    split_path = ui.PROJECT_ROOT / "artifacts" / "split_manifest.json"
    try:
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(split_manifest, dict):
            raise TypeError("the saved split manifest must be an object")
    except (OSError, ValueError, TypeError) as error:
        split_manifest = None
        st.warning(
            "Training-engine fit membership cannot be verified from the saved development "
            f"split ({error}); affected rows are labeled unknown."
        )
    try:
        supplied_bundle = st.secrets.get("deployment_model")
    except StreamlitSecretNotFoundError:
        supplied_bundle = None

    if not manifest_path.is_file() and supplied_bundle is None:
        st.warning(
            "Fleet predictions are unavailable: provide the trusted Decision Tree bundle "
            "through Streamlit secrets or install the verified local development bundle."
        )
    else:
        try:
            with deployment.prediction_manifest(manifest_path, supplied_bundle) as active_manifest:
                predictions = predict_fleet(
                    records,
                    dataset_partition="FD001/train",
                    manifest_path=active_manifest,
                )
                try:
                    model_provenance = model_provenance_from_manifest(active_manifest)
                except ValueError as error:
                    model_provenance = None
                    st.warning(f"Comparison provenance is unavailable: {error}")
        except (PredictionArtifactError, PredictionInputError, ValueError) as error:
            st.warning(f"Fleet predictions are unavailable: {error}")
        else:
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
                    hashlib.sha256(source_path.read_bytes()).hexdigest()
                    if source_path.is_file()
                    else None
                ),
            }
            st.caption(
                "The displayed rows are one snapshot from the same saved Decision Tree model "
                "and version. The packaged option contains only three training-history prefixes; "
                "it is not a complete fleet or model-quality evidence."
            )
            engine_ids = [result.identity.unit_id for result in predictions]
            selected_engine_ids = st.multiselect(
                "Engine IDs",
                options=engine_ids,
                default=engine_ids,
            )
            threshold = st.number_input(
                "Review threshold (cycles)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                help="A display filter for review; it is not a fault alarm or maintenance rule.",
            )
            threshold_only = st.checkbox(
                "Show only predictions at or below the threshold",
                value=False,
                help="Rows with validation or prediction errors remain visible because they have no estimate.",
            )
            selected_set = set(selected_engine_ids)
            saved_selection = [
                {
                    "Dataset": result.identity.dataset_partition,
                    "Engine ID": result.identity.unit_id,
                    "Observed cycle": result.observed_cycle,
                    "Fit membership": _fit_membership_label(
                        split_manifest, result.identity.unit_id
                    ),
                    "Predicted RUL (cycles)": result.displayed_rul_cycles,
                    "Raw RUL (cycles)": result.raw_rul_cycles,
                    "Model": result.model_name,
                    "Version": result.model_version,
                    "Feature rows": result.input_window_rows,
                    "Warnings": ", ".join(result.warnings),
                    "Error": (
                        f"{', '.join(result.error_codes)}: {result.error}"
                        if result.error and result.error_codes
                        else result.error
                    ),
                }
                for result in predictions
                if result.identity.unit_id in selected_set
            ]
            if st.button(
                "Save selected engines for Maintenance Planner",
                disabled=not selected_engine_ids,
            ):
                st.session_state[ui.P15_SELECTED_FLEET_STATE_KEY] = saved_selection
                st.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY] = {
                    "source": source_provenance,
                    "model": model_provenance,
                }
                st.session_state.pop(ui.P20_FLEET_SCENARIO_STATE_KEY, None)
                st.session_state.pop(ui.P20_FLEET_COMPARISON_STATE_KEY, None)
                st.session_state.pop(ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY, None)
                st.success(f"Saved {len(saved_selection)} selected engine records for the planner.")
                st.page_link(st.Page(_app_dir / "pages" / "4_Maintenance_Planner.py"), label="Open Maintenance Planner")

            visible_results = [
                result
                for result in predictions
                if result.identity.unit_id in selected_set
                and (
                    not threshold_only
                    or result.displayed_rul_cycles is None
                    or result.displayed_rul_cycles <= threshold
                )
            ]
            visible_results.sort(
                key=lambda result: (
                    result.displayed_rul_cycles is None,
                    result.displayed_rul_cycles
                    if result.displayed_rul_cycles is not None
                    else 0.0,
                    result.identity.dataset_partition,
                    result.identity.unit_id,
                )
            )
            rows = [
                {
                    "Dataset": result.identity.dataset_partition,
                    "Engine ID": result.identity.unit_id,
                    "Observed cycle": result.observed_cycle,
                    "Fit membership": _fit_membership_label(
                        split_manifest, result.identity.unit_id
                    ),
                    "Predicted RUL (cycles)": result.displayed_rul_cycles,
                    "Raw RUL (cycles)": result.raw_rul_cycles,
                    "Model": result.model_name,
                    "Version": result.model_version,
                    "Feature rows": result.input_window_rows,
                    "Threshold status": (
                        "No estimate"
                        if result.displayed_rul_cycles is None
                        else "At or below threshold"
                        if result.displayed_rul_cycles <= threshold
                        else "Above threshold"
                    ),
                    "Warnings": ", ".join(result.warnings),
                    "Error": (
                        f"{', '.join(result.error_codes)}: {result.error}"
                        if result.error and result.error_codes
                        else result.error
                    ),
                }
                for result in visible_results
            ]
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.info("No engines match the current selection and threshold filter.")
            if any(result.error for result in predictions):
                st.warning(
                    "Rows with a validation or prediction error have no RUL estimate. "
                    "Their original engine IDs remain visible."
                )
            else:
                st.success(f"Displayed {len(visible_results)} ID-attached engine predictions.")
else:
    st.info("Choose a history source with validated engine records to view fleet predictions.")
