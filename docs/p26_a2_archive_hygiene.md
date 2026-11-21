# P26-A2 archive hygiene contract

Issue: [P26-A2 #182](https://github.com/Artfever/AI_PROJECT/issues/182)

## What is excluded

The submission ZIP must not contain virtual environments, VCS/editor state,
Python/test/lint caches, build outputs, secrets, temporary/backup files, local
FD001 raw files, development model artifacts, or experiment scratch outputs.
`.env.example` is allowed because it contains documented placeholders; actual
`.env*` files are not. Owner handoffs and private local planning notes are
collaboration records, not submission materials, and stay out of the ZIP.

`.gitignore` now covers common Python/Jupyter/IDE caches, scratch files, key or
credential file patterns, and existing local datasets/artifacts. The tracked
source tree was inspected for forbidden filenames: none were tracked. The
working clone does contain ignored `.venv/`, `.pytest_cache/`, and `.ruff_cache/`
directories; Git-based packaging and the archive audit must omit them.

## Absolute path cleanup

The tracked docs contained two environment-specific examples: a hosted
container mount path in `docs/p23_a3_remote_parity.md` and a developer Windows
temporary checkout path in `docs/p24_a2_experiment_reproduction.md`.
They now use `<deployment checkout>/...` and a generic temporary clone
description. The local path samples in the new audit tests are assembled from
string fragments so the test source itself stays portable.

## Repeatable archive check

Run this against the exact ZIP proposed for submission:

```powershell
.venv/Scripts/python.exe scripts/audit_submission_archive.py dist/fleetpulse-submission.zip
```

The command prints the ZIP SHA-256 and findings as JSON; it exits nonzero for
absolute or traversal member names, environment/cache and scratch entries,
secret-like filenames/content, raw local datasets, development artifacts, or
machine-specific paths embedded in text files. Synthetic unit tests exercise
both the allowed `.env.example` case and representative violations. This tool
does not replace the later exact-archive extraction and runtime test in P26-A3.

At this point the final ZIP has not yet been assembled, so no clean result is
claimed for the final archive. The audit contract and detector are ready for
P26-A3/P26-T3 to use after the package file list is frozen.

A temporary source snapshot containing 482 entries (with owner handoffs
excluded and the A2 audit files included) passed with zero findings. This
demonstrates the detector against a realistic repository snapshot, but it is not
the final submission ZIP and has not been run from an extracted directory.
