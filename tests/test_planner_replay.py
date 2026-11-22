"""Ensure the hangar faithfully shows the selected schedule and score."""

import json
from dataclasses import replace

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT
from fleetpulse.services.schedule_comparison import compare_schedules
from fleetpulse.ui_planner_replay import build_planner_replay_payload
from fleetpulse.ui_schedule_sandbox import DEMO_BASELINE


def _hangar_payload(page):
    frame = next(item for item in page.get("iframe") if "Maintenance Hangar" in item.proto.srcdoc)
    payload = frame.proto.srcdoc.split('id="payload">', 1)[1].split("</script>", 1)[0]
    return json.loads(payload)


def test_replay_preserves_job_b_deadline_and_weight_when_capacity_changes():
    before = build_planner_replay_payload(
        compare_schedules(DEMO_BASELINE), {}, source="One team"
    )
    after = build_planner_replay_payload(
        compare_schedules(replace(DEMO_BASELINE, team_count=2)), {}, source="Two teams"
    )

    assert [before["schedules"][method]["weighted"] for method in ("EDD", "GA")] == [6, 2]
    assert [after["schedules"][method]["weighted"] for method in ("EDD", "GA")] == [1, 1]
    job_b = DEMO_BASELINE.jobs[1].engine
    key = [job_b.dataset_partition, job_b.unit_id]
    old, new = [
        next(job for job in payload["schedules"]["EDD"]["jobs"] if job["id"] == key)
        for payload in (before, after)
    ]
    assert (old["team"], old["start"], old["finish"], old["due"], old["late"], old["weight"]) == (
        1, 3, 4, 3, 1, 5,
    )
    assert (new["team"], new["start"], new["finish"], new["due"], new["late"], new["weight"]) == (
        2, 0, 1, 3, 0, 5,
    )


def test_planner_replays_saved_custom_scenario_after_submission():
    page = AppTest.from_file(
        str(PROJECT_ROOT / "app/modules/4_Maintenance_Planner.py"), default_timeout=35
    ).run()
    assert _hangar_payload(page)["source"] == "Appendix B demo"

    next(widget for widget in page.number_input if widget.label == "Maintenance teams").set_value(2)
    next(button for button in page.button if "Save scenario" in button.label).click().run()

    assert not page.exception
    replay = _hangar_payload(page)
    assert replay["source"] == "Custom simulation"
    assert replay["schedules"]["EDD"]["teamCount"] == 2
    assert replay["schedules"]["EDD"]["weighted"] == 1
    page.selectbox(key="fp_planner_replay_source").select("Appendix B demo").run()
    assert _hangar_payload(page)["schedules"]["EDD"]["weighted"] == 6
