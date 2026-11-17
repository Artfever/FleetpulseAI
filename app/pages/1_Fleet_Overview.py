"""Show saved-model fleet predictions with their engine identities."""

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import deployment, ui
from fleetpulse.services import (
    PredictionArtifactError,
    PredictionInputError,
    predict_fleet,
)

st.title("Fleet Overview")
st.caption(
    "Review saved-model RUL estimates for observed engine histories. Estimates are in operating "
    "cycles and do not confirm physical faults or maintenance needs."
)

source = st.selectbox(
    "History source",
    ["Packaged FD001/train parity subset", "Local FD001 training data"],
)
if source == "Packaged FD001/train parity subset":
    records = ui.render_packaged_training_subset()
else:
    records = ui.render_training_data()

if records:
    artifact_dir = ui.PROJECT_ROOT / "artifacts" / "development"
    manifest_path = artifact_dir / "decision_tree_validation.json"
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
        except (PredictionArtifactError, PredictionInputError) as error:
            st.warning(f"Fleet predictions are unavailable: {error}")
        else:
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
                st.success(f"Saved {len(saved_selection)} selected engine records for the planner.")
                st.page_link("pages/4_Maintenance_Planner.py", label="Open Maintenance Planner")

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
