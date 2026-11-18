"""Verified, JSON-safe schedule rows before table/chart presentation (P18-A3)."""

import hashlib
import json
from dataclasses import asdict

from fleetpulse.optimization.jobs import JobTiming, MaintenanceJob
from fleetpulse.optimization.scoring import (
    JobAssignment,
    ScheduledJob,
    ScheduleEvaluation,
    ScheduleFeasibilityError,
    SchedulingScenario,
    evaluate_schedule,
)

SCHEDULE_REPORT_SCHEMA = "p18-schedule-report-v1"


def build_schedule_report(scenario: SchedulingScenario, schedule: ScheduleEvaluation) -> dict:
    """Recheck assignments and all derived fields before exposing plot records.

    Never trust an externally constructed/stale ScheduleEvaluation just because
    its dataclass type is correct. Rebuild timing and score from the scenario,
    enforce shared feasibility, then reject any differing source job, timing,
    overflow or aggregate. Returned rows/summary are plain JSON-safe values.
    """
    if not isinstance(scenario, SchedulingScenario) or not isinstance(schedule, ScheduleEvaluation):
        raise TypeError("Expected SchedulingScenario and ScheduleEvaluation values.")
    if any(
        not isinstance(row, ScheduledJob)
        or not isinstance(row.job, MaintenanceJob)
        or not isinstance(row.timing, JobTiming)
        for row in schedule.scheduled_jobs
    ):
        raise TypeError("schedule must contain typed ScheduledJob records.")
    assignments = tuple(
        JobAssignment(row.job.engine, row.team_id, row.timing.start_cycle)
        for row in schedule.scheduled_jobs
    )
    verified = evaluate_schedule(scenario, assignments)
    if verified != schedule:
        raise ScheduleFeasibilityError(
            "Schedule contents or summaries differ from the scenario/shared calculation."
        )
    scenario_bytes = json.dumps(
        asdict(scenario), sort_keys=True, separators=(",", ":"), allow_nan=False, default=int
    ).encode()
    return {
        "schema_version": SCHEDULE_REPORT_SCHEMA,
        "time_unit": "relative_cycles",
        "objective": "sum(priority_weight * tardiness_cycles)",
        "feasibility_checked": True,
        "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
        "summary": {
            "team_count": scenario.team_count,
            "buffer_cycles": scenario.buffer_cycles,
            "horizon_cycles": scenario.horizon_cycles,
            "scheduled_job_count": len(verified.scheduled_jobs),
            "weighted_tardiness_cycles": verified.weighted_tardiness_cycles,
            "total_tardiness_cycles": verified.total_tardiness_cycles,
            "late_job_count": verified.late_job_count,
            "makespan_cycles": verified.makespan_cycles,
            "horizon_overflow_cycles": verified.horizon_overflow_cycles,
            "overflow_job_count": verified.overflow_job_count,
        },
        "jobs": [
            {
                "dataset_partition": row.job.engine.dataset_partition,
                "unit_id": int(row.job.engine.unit_id),
                "team_id": row.team_id,
                "predicted_rul_cycles": row.job.predicted_rul_cycles,
                "duration_cycles": row.job.duration_cycles,
                "priority_weight": row.job.priority_weight,
                "due_completion_cycle": row.timing.due_completion_cycle,
                "start_cycle": row.timing.start_cycle,
                "completion_cycle": row.timing.completion_cycle,
                "tardiness_cycles": row.timing.tardiness_cycles,
                "horizon_overflow_cycles": row.horizon_overflow_cycles,
            }
            for row in verified.scheduled_jobs
        ],
    }
