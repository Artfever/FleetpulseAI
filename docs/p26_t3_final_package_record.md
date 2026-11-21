# P26-T3 final package and submission procedure

Issue: [P26-T3 #263](https://github.com/Artfever/FleetpulseAI/issues/263)

## Exact tested package

| Field | Value |
| --- | --- |
| Candidate file | `dist/fleetpulse-submission.zip` |
| Archive contents | 487 tracked project files; excludes owner handoffs, this package record, ignored local files, virtual environments, caches, credentials, development artifacts, and raw datasets |
| Size | 9,498,405 bytes |
| SHA-256 | `b0c2159bf9a3499fceb3d2edf5c117f049b9b3a3ac1ff8e3b687d2c66ea706eb` |
| Archive hygiene | `scripts/audit_submission_archive.py` passed with zero findings |
| Extracted run | Python 3.12.16; editable install with `.[dev,xgboost,mlp]`; Ruff passed; pytest 740 passed / 7 skipped; five-model smoke passed; local Streamlit health returned `ok` |

The detailed extracted-copy record is [`p26_a3_archive_run.md`](p26_a3_archive_run.md). The hash record is kept outside the ZIP to avoid changing the bytes it identifies. This is a software-only course package for simulated NASA FD001, not real aircraft maintenance evidence.

## Human submitter and submission

**Designated submitter: not yet named by the team.** No course-platform submission has been made. The submitter must be an authorized team member chosen by the group; no identity or submission receipt is inferred here.

Once the team names the submitter:

1. The submitter obtains the exact `dist/fleetpulse-submission.zip` candidate and verifies the SHA-256 above before upload.
2. The submitter signs into the course's official submission platform and confirms the correct course, assignment, group, and deadline with the instructor's instructions.
3. The submitter uploads this ZIP without rebuilding or editing it.
4. The submitter verifies that the platform shows the uploaded filename and a successful receipt/status, then shares that receipt with the group.

If any file inside the ZIP changes, regenerate the package and rerun the A2 audit plus the A3 extracted setup/test/inference/app checks. Update this record to the new byte size and SHA before the team signs off.

## Outstanding human decisions

- Name the authorized course-platform submitter.
- Shaheer, Abdullah, and Taha inspect and sign off on the exact SHA above; this record does not claim that sign-off.
- Confirm the official course portal/assignment destination and submission receipt. No course platform credentials or assignment link were supplied, so no upload was attempted.
- The Streamlit Cloud deployment commit SHA remains unverified in this workspace.
