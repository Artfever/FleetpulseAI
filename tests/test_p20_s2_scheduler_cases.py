"""P20-S2 edge cases on one explicitly synthetic, fixed scheduling scenario."""

from fleetpulse.optimization.edd import edd_order, schedule_edd
from fleetpulse.optimization.ga import schedule_ga
from scripts.check_p20_s2_scheduler_cases import load_scenario


def test_p20_zero_deadline_precedes_later_due_jobs_and_all_work_remains_visible():
    scenario = load_scenario()

    assert [engine.unit_id for engine in edd_order(scenario)] == [1, 2, 3, 4]
    result = schedule_edd(scenario)
    rows = {row.job.engine.unit_id: row for row in result.scheduled_jobs}

    assert rows[1].timing.due_completion_cycle == 0
    assert rows[1].timing.tardiness_cycles == 3
    assert len(rows) == len(scenario.jobs) == 4
    assert result.late_job_count == 4


def test_p20_insufficient_near_term_capacity_reports_lateness_and_horizon_overflow():
    scenario = load_scenario()
    edd = schedule_edd(scenario)
    ga = schedule_ga(scenario, population_size=16, generations=20, seed=42)

    for result in (edd, ga.schedule):
        assert len(result.scheduled_jobs) == 4
        assert result.late_job_count > 0
        assert result.makespan_cycles == 10
        assert result.horizon_overflow_cycles == 4
        assert result.overflow_job_count > 0


def test_p20_heterogeneous_durations_and_tied_priorities_are_explicit_and_repeatable():
    scenario = load_scenario()
    by_id = {job.engine.unit_id: job for job in scenario.jobs}
    assert [by_id[unit_id].duration_cycles for unit_id in (1, 2, 3, 4)] == [3, 1, 4, 2]
    assert by_id[3].priority_weight == by_id[4].priority_weight == 1
    assert [engine.unit_id for engine in edd_order(scenario)] == [1, 2, 3, 4]

    first = schedule_ga(scenario, population_size=16, generations=20, seed=42)
    second = schedule_ga(scenario, population_size=16, generations=20, seed=42)
    assert first.order == second.order
    assert first.schedule == second.schedule
    assert first.best_penalties == second.best_penalties
    assert {row.job.engine for row in first.schedule.scheduled_jobs} == {
        job.engine for job in scenario.jobs
    }
