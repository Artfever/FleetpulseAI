"""Predict remaining cycles for a selected simulated training-engine history."""

import json

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import ui
from fleetpulse.data.loader import FD001LoadError, load_engine_history
from fleetpulse.data.quality import validate_engine_history
from fleetpulse.deployment import prediction_manifest
from fleetpulse.models.usage_scope import classify_training_engine
from fleetpulse.services import (
    PredictionArtifactError,
    PredictionInputError,
    predict_rul_with_context,
)

st.title("RUL Prediction")
st.caption(
    "Estimate remaining useful life in operating cycles using the saved P06 development model. "
    "This page does not load a final refitted model."
)

source = st.selectbox(
    "History source",
    [
        "Local FD001 training data",
        "Packaged FD001/train parity subset",
        "Illustrative synthetic example",
    ],
)
partition = "FD001/train"
if source == "Local FD001 training data":
    records = ui.render_training_data()
elif source == "Packaged FD001/train parity subset":
    records = ui.render_packaged_training_subset()
else:
    partition = "illustrative/synthetic"
    st.warning(
        "Artificial example, not NASA data or a real engine history. The model was not "
        "validated on these invented readings; its output is only a software demonstration."
    )
    try:
        records = load_engine_history(ui.PROJECT_ROOT / "data/demo/illustrative_history.txt")
        if not validate_engine_history(records).is_valid:
            raise FD001LoadError("The illustrative example has invalid history records.")
    except FD001LoadError as error:
        st.error(f"Illustrative example unavailable: {error}")
        records = None

if records is not None:
    engine_ids = sorted({int(row["unit_id"]) for row in records})
    engine = st.selectbox("Engine ID", engine_ids)
    history = sorted(
        (row for row in records if row["unit_id"] == engine), key=lambda row: row["cycle"]
    )
    if partition == "FD001/train":
        split_path = ui.PROJECT_ROOT / "artifacts" / "split_manifest.json"
        try:
            split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
            usage = classify_training_engine(
                split_manifest, dataset_partition=partition, unit_id=engine
            )
        except (OSError, ValueError, TypeError) as error:
            st.warning(
                "Training-set membership for this model cannot be verified from the saved "
                f"development split ({error}). Do not treat this replay as held-out evidence."
            )
        else:
            if usage == "fitting":
                st.warning(
                    "In-sample illustration: according to the saved development split, this engine "
                    "was used to fit the saved development model. Its prediction is not held-out evidence."
                )
            elif usage == "validation":
                st.info(
                    "Held-out development engine: according to the saved development split, this "
                    "engine was not used to fit the saved development model. This is development "
                    "evidence, not official-test performance."
                )
            else:
                st.warning(
                    "This engine is not listed in the saved development split, so its fit/holdout "
                    "status is unknown. Do not treat this replay as held-out evidence."
                )
    else:
        st.info(
            "Illustrative synthetic example: it is not a NASA engine, held-out evaluation, "
            "or maintenance result."
        )
    cycle = st.selectbox("Observed cycle", [int(row["cycle"]) for row in history])
    st.caption(
        "Only readings through the selected cycle are used; this is not a maintenance guarantee."
    )
    if st.button("Predict remaining cycles"):
        prefix = [row for row in history if row["cycle"] <= cycle]
        manifest = ui.PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
        try:
            supplied = None
            if not manifest.exists():
                try:
                    supplied = st.secrets.get("deployment_model")
                except StreamlitSecretNotFoundError:
                    pass
            with prediction_manifest(manifest, supplied) as active_manifest:
                result = predict_rul_with_context(
                    prefix,
                    dataset_partition=partition,
                    unit_id=engine,
                    manifest_path=active_manifest,
                )
        except (PredictionArtifactError, PredictionInputError) as error:
            st.warning(f"Prediction is unavailable: {error}")
            st.write("Check the saved development model bundle and the selected training history.")
        else:
            st.metric("Estimated remaining cycles", f"{result.displayed_rul_cycles:.2f}")
            st.write(f"Model: {result.model_name} ({result.model_version})")
            st.caption(
                f"{partition} engine {engine}, observed cycle {result.observed_cycle}; "
                f"{result.input_window_rows} observed rows in the feature window. "
                f"Raw estimate: {result.raw_rul_cycles:.2f} cycles."
            )
            if "short_window" in result.warnings:
                st.warning(
                    "The available feature window is short; the estimate uses fewer observations."
                )
            if "cycle_gaps" in result.warnings:
                st.warning("The observed history contains gaps between cycles.")
            if "display_clipped_at_zero" in result.warnings:
                st.warning(
                    "The displayed estimate is clipped to zero; the raw estimate is shown above."
                )
else:
    st.write("Choose an engine once the selected history is available.")
