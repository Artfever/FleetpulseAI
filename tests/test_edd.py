"""EDD is a deterministic deadline baseline, not a weighted-optimal claim."""

from dataclasses import replace

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.edd import edd_order, schedule_edd
from fleetpulse.optimization.jobs import MaintenanceJob, appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario


@pytest.mark.parametrize("teams,cost,completions", [(1, 6, (3, 4, 6)), (2, 1, (3, 1, 3))])
def test_appendix_b_edd_baseline_matches_reviewed_case(teams, cost, completions):
    example = SchedulingScenario(appendix_b_jobs(), teams, 2, 6)
    assert [engine.unit_id for engine in edd_order(example)] == [1, 2, 3]
    result = schedule_edd(example)
    assert result.weighted_tardiness_cycles == cost
    assert tuple(row.timing.completion_cycle for row in result.scheduled_jobs) == completions
    assert result == decode_order(example, edd_order(example))


def test_input_job_storage_order_does_not_change_edd_and_is_not_mutated():
    jobs = appendix_b_jobs()
    first = SchedulingScenario(jobs, 1, 2, 6)
    reverse = SchedulingScenario(tuple(reversed(jobs)), 1, 2, 6)
    assert schedule_edd(first) == schedule_edd(reverse)
    assert first.jobs == jobs and reverse.jobs == tuple(reversed(jobs))


def test_equal_clamped_deadlines_break_ties_by_qualified_engine_identity():
    jobs = (
        MaintenanceJob(EngineIdentity("train", 2), 1, 9, 99),
        MaintenanceJob(EngineIdentity("train", 1), 0, 1, 1),
        MaintenanceJob(EngineIdentity("test", 1), 1, 2, 5),
    )
    example = SchedulingScenario(jobs, 1, 2, 6)
    assert edd_order(example) == (
        EngineIdentity("test", 1),
        EngineIdentity("train", 1),
        EngineIdentity("train", 2),
    )


def test_fractional_deadlines_are_not_rounded_before_sorting():
    first = MaintenanceJob(EngineIdentity("synthetic/fractional", 9), 2.4, 1, 1)
    second = MaintenanceJob(EngineIdentity("synthetic/fractional", 1), 2.5, 1, 1)
    example = SchedulingScenario((second, first), 1, 2, 6)
    assert edd_order(example) == (first.engine, second.engine)
    assert schedule_edd(example).weighted_tardiness_cycles == pytest.approx(2.1)


def test_horizon_overflow_is_reported_without_changing_edd_order_or_cost():
    example = SchedulingScenario(appendix_b_jobs(), 1, 2, 2)
    result = schedule_edd(example)
    assert [row.job.engine.unit_id for row in result.scheduled_jobs] == [1, 2, 3]
    assert result.weighted_tardiness_cycles == 6
    assert result.horizon_overflow_cycles == 4 and result.overflow_job_count == 3


def test_baseline_does_not_claim_weighted_optimality():
    example = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
    alternatives = (example.jobs[1].engine, example.jobs[0].engine, example.jobs[2].engine)
    assert schedule_edd(example).weighted_tardiness_cycles == 6
    assert decode_order(example, alternatives).weighted_tardiness_cycles == 2


def test_order_is_deadline_only_even_when_durations_or_priorities_change():
    jobs = appendix_b_jobs()
    changed = (
        replace(jobs[0], duration_cycles=100),
        replace(jobs[1], priority_weight=1000),
        jobs[2],
    )
    assert edd_order(SchedulingScenario(changed, 1, 2, 6)) == tuple(job.engine for job in jobs)


def test_edd_calls_shared_decoder_with_the_unchanged_scenario(monkeypatch):
    import fleetpulse.optimization.edd as service

    example = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
    calls = []

    def observed(scenario, ordering):
        calls.append((scenario, ordering))
        return decode_order(scenario, ordering)

    monkeypatch.setattr(service, "decode_order", observed)
    assert service.schedule_edd(example).weighted_tardiness_cycles == 6
    assert calls == [(example, edd_order(example))]


def test_invalid_scenario_has_clear_error():
    with pytest.raises(TypeError, match="SchedulingScenario"):
        schedule_edd(None)
