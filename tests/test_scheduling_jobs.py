"""Validation for synthetic maintenance-scenario job inputs."""

from dataclasses import FrozenInstanceError

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.jobs import (
    MaintenanceJob,
    appendix_b_jobs,
    calculate_job_timing,
)


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


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ((0, 1, 2), ((2.0, 0, 3, 1.0), (3.0, 3, 4, 1.0), (6.0, 4, 6, 0.0))),
        ((1, 0, 2), ((3.0, 0, 1, 0.0), (2.0, 1, 4, 2.0), (6.0, 4, 6, 0.0))),
    ],
)
def test_appendix_b_completion_deadline_and_tardiness_example(order, expected):
    jobs = appendix_b_jobs()
    start = 0
    actual = []
    for index in order:
        timing = calculate_job_timing(jobs[index], buffer_cycles=2, start_cycle=start)
        actual.append(
            (
                timing.due_completion_cycle,
                timing.start_cycle,
                timing.completion_cycle,
                timing.tardiness_cycles,
            )
        )
        start = timing.completion_cycle

    assert tuple(actual) == expected


def test_due_cycle_is_clamped_at_zero_and_tardiness_uses_completion():
    job = MaintenanceJob(EngineIdentity("synthetic/test", 1), 1, 1, 1)

    timing = calculate_job_timing(job, buffer_cycles=2)

    assert timing.due_completion_cycle == 0
    assert timing.start_cycle == 0
    assert timing.completion_cycle == 1
    assert timing.tardiness_cycles == 1


@pytest.mark.parametrize("buffer", [-1, float("nan"), float("inf"), True])
def test_timing_rejects_invalid_buffers(buffer):
    with pytest.raises((TypeError, ValueError), match="buffer_cycles"):
        calculate_job_timing(appendix_b_jobs()[0], buffer_cycles=buffer)


@pytest.mark.parametrize("start", [-1, 1.0, True])
def test_timing_rejects_invalid_start_cycles(start):
    with pytest.raises(ValueError, match="start_cycle"):
        calculate_job_timing(appendix_b_jobs()[0], buffer_cycles=0, start_cycle=start)
