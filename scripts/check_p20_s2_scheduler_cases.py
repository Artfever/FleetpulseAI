"""Run the fixed synthetic P20-S2 urgent/capacity-pressure example."""

from __future__ import annotations

import json
from pathlib import Path

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.ga import schedule_ga
from fleetpulse.optimization.jobs import MaintenanceJob
from fleetpulse.optimization.scoring import SchedulingScenario

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/p20_s2_capacity_pressure.json"


def load_scenario() -> SchedulingScenario:
    """Load the documented, synthetic job mix from its checked-in fixture."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    jobs = tuple(
        MaintenanceJob(
            EngineIdentity(payload["dataset_partition"], row["unit_id"]),
            row["predicted_rul_cycles"],
            row["duration_cycles"],
            row["priority_weight"],
        )
        for row in payload["jobs"]
    )
    return SchedulingScenario(
        jobs,
        team_count=payload["team_count"],
        buffer_cycles=payload["buffer_cycles"],
        horizon_cycles=payload["horizon_cycles"],
    )


def _summary(result):
    return {
        "order": [row.job.engine.unit_id for row in result.scheduled_jobs],
        "completion_cycles": [row.timing.completion_cycle for row in result.scheduled_jobs],
        "due_completion_cycles": [row.timing.due_completion_cycle for row in result.scheduled_jobs],
        "tardiness_cycles": [row.timing.tardiness_cycles for row in result.scheduled_jobs],
        "weighted_tardiness_cycles": result.weighted_tardiness_cycles,
        "late_job_count": result.late_job_count,
        "makespan_cycles": result.makespan_cycles,
        "horizon_overflow_cycles": result.horizon_overflow_cycles,
        "overflow_job_count": result.overflow_job_count,
    }


def main() -> None:
    scenario = load_scenario()
    edd = schedule_edd(scenario)
    ga = schedule_ga(scenario, population_size=16, generations=20, seed=42)
    assert len(edd.scheduled_jobs) == len(ga.schedule.scheduled_jobs) == 4
    assert edd.makespan_cycles == ga.schedule.makespan_cycles == 10
    assert edd.horizon_overflow_cycles == ga.schedule.horizon_overflow_cycles == 4
    print(
        json.dumps(
            {
                "provenance": "synthetic fixture; all durations, weights, team capacity, buffer, and horizon are assumptions",
                "scenario": {
                    "team_count": scenario.team_count,
                    "buffer_cycles": scenario.buffer_cycles,
                    "horizon_cycles": scenario.horizon_cycles,
                    "jobs": [
                        {
                            "engine": job.engine.unit_id,
                            "rul_cycles": job.predicted_rul_cycles,
                            "duration_cycles": job.duration_cycles,
                            "priority_weight": job.priority_weight,
                        }
                        for job in scenario.jobs
                    ],
                },
                "edd": _summary(edd),
                "ga": {
                    **_summary(ga.schedule),
                    "seed": ga.seed,
                    "population_size": ga.population_size,
                    "generations": ga.generations,
                    "evaluations": ga.evaluations,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
