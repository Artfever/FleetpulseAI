"""P22-A2 seed comparison checks feasibility, hand scores and repeatability."""

from scripts.run_p22_a2_seed_comparison import run_comparison


def test_p22_a2_runs_every_seed_for_each_fixed_scenario_and_checks_equal_and_better():
    predictions = {unit_id: float(30 + unit_id) for unit_id in range(1, 21)}

    result = run_comparison(predictions)

    assert result["case_count"] == 18
    assert result["ga_run_count"] == 90
    assert result["ga_seeds"] == [0, 7, 42, 2026, 2027]
    assert result["all_schedules_feasible"]
    assert result["all_seed_repeats_match"]
    assert result["outcome_counts"]["equal"] > 0
    assert result["outcome_counts"]["better"] > 0
    assert all(
        seed["ga"]["weighted_tardiness_cycles"] <= row["edd"]["weighted_tardiness_cycles"]
        for row in result["scenarios"]
        for seed in row["ga_by_seed"]
    )


def test_p22_a2_retains_hand_calculated_appendix_b_reference():
    predictions = {unit_id: 50.0 + unit_id for unit_id in range(1, 21)}

    result = run_comparison(predictions)

    reference = result["appendix_b_hand_check"]
    assert reference["edd"]["weighted_tardiness_cycles"] == 6
    assert reference["edd"]["late_job_count"] == 2
    assert reference["ga_seed_42"]["weighted_tardiness_cycles"] == 2
    assert reference["edd"]["feasible"] and reference["ga_seed_42"]["feasible"]
