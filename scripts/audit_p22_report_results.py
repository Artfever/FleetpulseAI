"""Audit the report-ready P22 summary against committed evidence artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from fleetpulse.evaluation.p22_study_display import (
    load_error_analysis,
    load_scheduler_study,
    load_sensor_sensitivity,
    load_validation_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/p22_report_results.md"


def _number(value: float) -> str:
    return f"{value:,.2f}"


def audit_report() -> tuple[int, int]:
    report = REPORT.read_text(encoding="utf-8-sig")
    s1 = load_error_analysis(ROOT / "docs/figures/p22_s1_error_analysis.json")
    s2 = load_validation_bootstrap(ROOT / "docs/p22_s2_validation_bootstrap.json")
    s3 = load_sensor_sensitivity(ROOT / "docs/p22_s3_sensor_sensitivity.json")
    a1 = load_scheduler_study(
        ROOT / "docs/p22_a1_fixed_fleet_scenarios.json",
        expected_schema="p22-a1-fixed-fleet-scenarios-v1",
    )
    a2 = load_scheduler_study(
        ROOT / "docs/p22_a2_seed_comparison.json",
        expected_schema="p22-a2-seed-comparison-v1",
    )

    checked = 0
    for model in s1["models"].values():
        summary = model["summary"]
        for bin_name in ("low", "medium", "high"):
            assert _number(summary["by_true_rul_bin"][bin_name]["mae_cycles"]) in report
            checked += 1
        for metric in ("mae_cycles", "rmse_cycles"):
            assert _number(summary["overall"][metric]) in report
            checked += 1

    for model in s2["models"].values():
        assert _number(model["observed_mae_cycles"]) in report
        assert all(_number(value) in report for value in model["percentile_95_ci_mae_cycles"])
        checked += 3

    for row in s3["aggregate_summary"]:
        assert _number(row["mean_absolute_prediction_change_cycles"]) in report
        checked += 1
    largest_mlp_change = max(
        row["max_absolute_prediction_change_cycles"]
        for row in s3["aggregate_summary"]
        if row["model_version"] == "dense-rul-mlp-v1"
    )
    assert _number(largest_mlp_change) in report
    checked += 1

    assert a1["ga_outcomes"] == {"equal": 12, "better": 6}
    assert "12 EDD/GA ties and 6 GA improvements" in report
    assert a2["ga_run_count"] == 90
    assert a2["outcome_counts"] == {"equal": 60, "better": 30}
    assert a2["all_schedules_feasible"] is True
    assert a2["all_seed_repeats_match"] is True
    assert "60 ties, 30 improvements and no regressions" in report
    checked += 5

    replay_report = (ROOT / "docs/p22_a3_replay_profile.md").read_text(encoding="utf-8")
    for value in ("66.43", "53.41", "33.49", "18.77"):
        assert value in replay_report and value in report
        checked += 1

    images = re.findall(r"!\[([^\]]+)\]\(([^)]+\.svg)\)", report)
    assert len(images) == 3
    for alt_text, relative_path in images:
        assert alt_text.strip()
        assert (REPORT.parent / relative_path).is_file()

    local_links = [
        target
        for target in re.findall(r"\]\(([^)]+)\)", report)
        if not target.startswith(("http://", "https://", "#"))
    ]
    missing = [
        target
        for target in local_links
        if not (REPORT.parent / target.split("#", 1)[0]).is_file()
    ]
    assert not missing, f"missing local report links: {missing}"
    dashboard_note = (ROOT / "docs/p22_dashboard_studies.md").read_text(encoding="utf-8-sig")
    assert "p22_report_results.md" in dashboard_note
    return checked, len(local_links)


def main() -> None:
    claims, links = audit_report()
    print(
        f"PASS: checked {claims} report claims against P22 JSON/source reports; "
        f"verified 3 labeled SVGs and {links} local links."
    )


if __name__ == "__main__":
    main()
