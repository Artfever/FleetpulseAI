"""Check the data and model boundaries of the 3D replay."""

from pathlib import Path
from shutil import copyfile

import pytest

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.ui_engine_replay import SENSORS, build_replay_payload, check_presenter_estimate

_SUBSET = Path("data/demo/fd001_training_prefixes.txt")
_MANIFEST = Path("artifacts/development/decision_tree_validation.json")


def test_replay_keeps_recorded_readings_when_estimates_are_unavailable(tmp_path, monkeypatch):
    target = tmp_path / _SUBSET
    target.parent.mkdir(parents=True)
    copyfile(PROJECT_ROOT / _SUBSET, target)
    monkeypatch.setattr("fleetpulse.ui_engine_replay.st.secrets", {})

    payload = build_replay_payload(tmp_path)

    assert [engine["id"] for engine in payload["engines"]] == [4, 5, 12]
    engine = payload["engines"][0]
    assert engine["cycles"] == list(range(1, 165))
    assert all(len(engine["sensors"][name]) == 164 for name in SENSORS)
    assert engine["checkpoints"] == []
    assert "Install the verified" in engine["problem"]


@pytest.mark.skipif(
    not (PROJECT_ROOT / _MANIFEST).is_file(), reason="local development bundle unavailable"
)
def test_replay_uses_verified_cutoff_estimates_including_the_final_cycle():
    engine = build_replay_payload(PROJECT_ROOT)["engines"][0]
    checkpoints = engine["checkpoints"]

    assert len(checkpoints) == 7
    assert [point["cycle"] for point in checkpoints] == [1, 28, 55, 82, 109, 136, 164]
    assert checkpoints[-1]["version"] == "decision-tree-rul-v1"
    assert checkpoints[-1]["rul"] == pytest.approx(34.19607843137255)
    records = load_engine_history(PROJECT_ROOT / _SUBSET)
    assert check_presenter_estimate(PROJECT_ROOT, records) == (checkpoints[-1], None)
