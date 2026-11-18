"""Validate the P20 planner boundary using synthetic fleet results."""

from dataclasses import replace

import pytest

from fleetpulse.optimization.jobs import calculate_job_timing
from fleetpulse.services import (
    EngineIdentity,
    FleetPredictionResult,
    PlannerJobAssumption,
    build_planning_scenario,
)


def _prediction(unit_id, raw, displayed=None, *, model="Random Forest", version="final-v1"):
    return FleetPredictionResult(
        EngineIdentity("FD001/train", unit_id),
        20,
        raw,
        max(0.0, raw) if displayed is None else displayed,
        model,
        version,
        20,
        (),
        (),
        None,
    )


def _assumptions(*unit_ids):
    return {
        EngineIdentity("FD001/train", unit_id): PlannerJobAssumption(3, unit_id)
        for unit_id in unit_ids
    }


def test_planner_uses_single_prediction_snapshot_and_separate_assumptions():
    predictions = (_prediction(5, 8.125), _prediction(4, -0.25))
    assumptions = _assumptions(4, 5)

    scenario = build_planning_scenario(
        predictions,
        assumptions,
        team_count=2,
        buffer_cycles=1.5,
        horizon_cycles=10,
    )

    assert [job.engine.unit_id for job in scenario.jobs] == [4, 5]
    assert [job.predicted_rul_cycles for job in scenario.jobs] == [0.0, 8.125]
    assert [job.duration_cycles for job in scenario.jobs] == [3, 3]
    assert [job.priority_weight for job in scenario.jobs] == [4, 5]
    assert (scenario.team_count, scenario.buffer_cycles, scenario.horizon_cycles) == (2, 1.5, 10)
    # The raw model output remains available for evaluation and was not mutated.
    assert [result.raw_rul_cycles for result in predictions] == [8.125, -0.25]


def test_planner_deadlines_use_unrounded_raw_rul_not_display_value():
    predictions = (
        _prediction(4, 2.0000004, displayed=2.0),
        _prediction(5, -0.0000004, displayed=0.0),
    )
    scenario = build_planning_scenario(
        predictions,
        _assumptions(4, 5),
        team_count=1,
        buffer_cycles=2.0,
        horizon_cycles=6.0,
    )

    jobs = {job.engine.unit_id: job for job in scenario.jobs}
    above_buffer = calculate_job_timing(jobs[4], buffer_cycles=scenario.buffer_cycles)
    below_zero = calculate_job_timing(jobs[5], buffer_cycles=scenario.buffer_cycles)

    assert jobs[4].predicted_rul_cycles == 2.0000004
    assert above_buffer.due_completion_cycle == pytest.approx(0.0000004)
    assert jobs[5].predicted_rul_cycles == 0.0
    assert below_zero.due_completion_cycle == 0.0
    assert [row.raw_rul_cycles for row in predictions] == [2.0000004, -0.0000004]


@pytest.mark.parametrize("raw", [None, float("nan"), float("inf"), -float("inf"), True])
def test_planner_requires_finite_nonboolean_raw_rul(raw):
    prediction = _prediction(4, raw, displayed=0.0)
    with pytest.raises(ValueError, match="finite raw RUL"):
        build_planning_scenario(
            (prediction,),
            _assumptions(4),
            team_count=1,
            buffer_cycles=2,
            horizon_cycles=6,
        )


@pytest.mark.parametrize(
    "predictions,assumptions,message",
    [
        ((), {}, "At least one fleet prediction"),
        ((_prediction(4, 1), _prediction(4, 2)), _assumptions(4), "identities must be unique"),
        ((_prediction(4, 1, model="Tree"), _prediction(5, 2)), _assumptions(4, 5), "one saved model"),
        ((_prediction(4, 1),), {}, "explicit duration and priority"),
        ((_prediction(4, 1),), _assumptions(4, 5), "match the prediction snapshot"),
        ((replace(_prediction(4, 1), error="missing"),), _assumptions(4), "no usable prediction"),
    ],
)
def test_planner_rejects_incomplete_or_mixed_snapshots(predictions, assumptions, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_planning_scenario(
            predictions,
            assumptions,
            team_count=1,
            buffer_cycles=2,
            horizon_cycles=6,
        )


@pytest.mark.parametrize("duration,weight", [(0, 1), (1, 0), (1.5, 1), (1, True)])
def test_planner_rejects_nonpositive_or_nonintegral_assumptions(duration, weight):
    assumptions = {EngineIdentity("FD001/train", 4): PlannerJobAssumption(duration, weight)}
    with pytest.raises(ValueError, match="positive integer"):
        build_planning_scenario(
            (_prediction(4, 1),),
            assumptions,
            team_count=1,
            buffer_cycles=2,
            horizon_cycles=6,
        )
