"""Validation for synthetic maintenance-scenario job inputs."""

from dataclasses import FrozenInstanceError

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.jobs import MaintenanceJob, appendix_b_jobs


def test_appendix_b_inputs_have_qualified_identity_and_explicit_synthetic_values():
    jobs = appendix_b_jobs()

    assert [job.engine for job in jobs] == [
        EngineIdentity("synthetic/appendix-b-v1", 1),
        EngineIdentity("synthetic/appendix-b-v1", 2),
        EngineIdentity("synthetic/appendix-b-v1", 3),
    ]
    assert [job.predicted_rul_cycles for job in jobs] == [4.0, 5.0, 8.0]
    assert [job.duration_cycles for job in jobs] == [3, 1, 2]
    assert [job.priority_weight for job in jobs] == [1, 5, 1]
    with pytest.raises(FrozenInstanceError):
        jobs[0].duration_cycles = 99


@pytest.mark.parametrize(
    ("engine", "rul", "duration", "weight", "message"),
    [
        (EngineIdentity(" ", 1), 4, 1, 1, "dataset_partition"),
        (EngineIdentity("FD001/train", 0), 4, 1, 1, "unit_id"),
        (EngineIdentity("FD001/train", True), 4, 1, 1, "unit_id"),
        (EngineIdentity("FD001/train", 1), -0.1, 1, 1, "nonnegative"),
        (EngineIdentity("FD001/train", 1), float("nan"), 1, 1, "finite"),
        (EngineIdentity("FD001/train", 1), float("inf"), 1, 1, "finite"),
        (EngineIdentity("FD001/train", 1), 4, 0, 1, "duration_cycles"),
        (EngineIdentity("FD001/train", 1), 4, 1.5, 1, "duration_cycles"),
        (EngineIdentity("FD001/train", 1), 4, 1, 0, "priority_weight"),
        (EngineIdentity("FD001/train", 1), 4, 1, True, "priority_weight"),
    ],
)
def test_job_inputs_reject_invalid_identity_and_values(engine, rul, duration, weight, message):
    with pytest.raises(ValueError, match=message):
        MaintenanceJob(engine, rul, duration, weight)


def test_same_numeric_engine_id_remains_distinct_across_partitions():
    fitting = MaintenanceJob(EngineIdentity("FD001/train", 7), 10, 2, 3)
    testing = MaintenanceJob(EngineIdentity("FD001/test", 7), 10, 2, 3)

    assert fitting.engine != testing.engine
