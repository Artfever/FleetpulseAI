"""Synthetic P14 uploads demonstrate valid and rejected FD001 histories."""

from pathlib import Path

import pytest

from fleetpulse.data.loader import FD001LoadError, load_engine_history
from fleetpulse.data.quality import validate_engine_history

FIXTURES = Path(__file__).parent / "fixtures" / "p14_uploads"


def _load(name: str):
    return load_engine_history(FIXTURES / name)


def test_valid_synthetic_upload_loads_and_passes_quality_checks():
    records = _load("valid_synthetic_history.txt")

    assert len(records) == 3
    assert {record["unit_id"] for record in records} == {910001}
    assert [record["cycle"] for record in records] == [1, 2, 3]
    assert validate_engine_history(records).is_valid


def test_empty_upload_is_rejected_by_loader():
    with pytest.raises(FD001LoadError, match="no sensor records"):
        _load("empty_history.txt")


def test_duplicate_cycle_upload_is_flagged_by_quality_validation():
    report = validate_engine_history(_load("duplicate_cycle.txt"))

    assert not report.is_valid
    assert "duplicate_engine_cycle" in {finding.code for finding in report.errors}


def test_missing_channel_upload_is_rejected_by_loader():
    with pytest.raises(FD001LoadError, match="has 25 fields; expected 26"):
        _load("missing_channel.txt")


def test_text_sensor_upload_is_rejected_by_loader():
    with pytest.raises(FD001LoadError, match="contains invalid numbers"):
        _load("text_sensor.txt")


def test_nonfinite_sensor_upload_is_flagged_by_quality_validation():
    report = validate_engine_history(_load("nonfinite_sensor.txt"))

    assert not report.is_valid
    assert "non_finite_reading" in {finding.code for finding in report.errors}


def test_shuffled_cycle_upload_is_not_reordered_and_is_flagged():
    records = _load("shuffled_cycles.txt")

    assert [record["cycle"] for record in records] == [1, 3, 2]
    report = validate_engine_history(records)
    assert not report.is_valid
    assert "cycle_order" in {finding.code for finding in report.errors}
