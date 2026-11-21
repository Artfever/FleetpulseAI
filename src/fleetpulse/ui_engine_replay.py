"""Feed packaged engine histories and verified model checkpoints to the 3D replay."""

import json
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import FD001LoadError, load_engine_history

ASSETS = PROJECT_ROOT / "app" / "engine_replay"
SENSORS = ("sensor_9", "sensor_2", "sensor_11")


def build_replay_payload(root: Path) -> dict:
    """Bundle observed samples and past-only predictions at up to seven cutoffs per engine."""
    from fleetpulse.deployment import prediction_manifest
    from fleetpulse.services import PredictionArtifactError, PredictionInputError
    from fleetpulse.services.history import replay_engine_history

    records = load_engine_history(root / "data" / "demo" / "fd001_training_prefixes.txt")
    manifest = root / "artifacts" / "development" / "decision_tree_validation.json"
    supplied = None
    if not manifest.is_file():
        try:
            supplied = st.secrets.get("deployment_model")
        except StreamlitSecretNotFoundError:
            pass
    engines = []
    for unit_id in sorted({row["unit_id"] for row in records}):
        rows = sorted(
            (row for row in records if row["unit_id"] == unit_id), key=lambda row: row["cycle"]
        )
        cycles = [row["cycle"] for row in rows]
        checkpoints = []
        problem = None
        if manifest.is_file() or supplied is not None:
            last = len(cycles) - 1
            cutoffs = [cycles[i * last // 6] for i in range(7)] if last >= 6 else cycles
            try:
                with prediction_manifest(manifest, supplied) as active_manifest:
                    replay = replay_engine_history(
                        rows,
                        dataset_partition="FD001/train",
                        unit_id=unit_id,
                        selected_cycle=cycles[-1],
                        replay_cutoffs=cutoffs,
                        manifest_path=active_manifest,
                    )
                checkpoints = [
                    {
                        "cycle": point.cutoff_cycle,
                        "rul": point.prediction.displayed_rul_cycles,
                        "version": point.prediction.model_version,
                        "warnings": list(point.prediction.warnings),
                    }
                    for point in replay.points
                ]
            except (PredictionArtifactError, PredictionInputError):
                problem = "The saved Decision Tree could not produce replay estimates."
        else:
            problem = "Install the verified development Decision Tree bundle to show estimates."
        engines.append(
            {
                "id": unit_id,
                "cycles": cycles,
                "sensors": {sensor: [row[sensor] for row in rows] for sensor in SENSORS},
                "checkpoints": checkpoints,
                "problem": problem,
            }
        )
    return {"engines": engines, "sensors": SENSORS}


def _mount_replay(payload: dict, *, description: str) -> None:
    """Mount a self-contained browser scene without third-party scripts."""
    document = (ASSETS / "index.html").read_text(encoding="utf-8")
    css = (ASSETS / "replay.css").read_text(encoding="utf-8")
    script = (ASSETS / "replay.js").read_text(encoding="utf-8")
    data = json.dumps(payload, allow_nan=False).replace("<", r"\u003c")
    html = document.replace("/* STYLES */", css).replace("/* SCRIPT */", script)
    st.iframe(html.replace("/* DATA */", data), tab_index=0, alt=description)


def render_aircraft_preview() -> None:
    """Show the illustrative aircraft on the dashboard without loading a model."""
    _mount_replay(
        {"preview": True, "engines": [], "sensors": []},
        description="Rotatable aircraft illustration",
    )


def render_engine_replay(root: Path) -> bool:
    """Show the complete replay when the packaged histories are available."""
    try:
        payload = build_replay_payload(root)
    except FD001LoadError:
        return False
    if not payload["engines"]:
        return False
    st.subheader("3D Engine Replay")
    st.caption("Orbit the aircraft, select either engine, then follow the packaged FD001 history.")
    _mount_replay(payload, description="Interactive 3D aircraft, turbofan, and cycle replay")
    return True
