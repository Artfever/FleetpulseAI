# P26-A3 exact archive extraction and run record

Issue: [P26-A3 #236](https://github.com/Artfever/FleetpulseAI/issues/236)

## Archive under test

The candidate group package is `dist/fleetpulse-submission.zip`. It contains the tracked project files and excludes the three owner handoffs and all ignored local files. Raw NASA files are not included; their documented source and checksums remain in `docs/fd001_provenance.md`. The file is a software project package for the simulated FD001 course scope, not a standalone installer.

The exact archive hygiene audit and its SHA-256 are captured in the P26-T3 package manifest, which is a repository review record kept outside the ZIP to avoid a self-referential archive hash. If any archive member changes, regenerate this record and rerun the audit and extracted-copy checks.

## Fresh extraction and setup

The ZIP was extracted to a new temporary directory containing 487 files. Python `3.12.16` was used. From the extracted project root, the documented full runtime/development setup completed successfully:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,xgboost,mlp]" --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m streamlit run app/Home.py
```

For the final candidate check, the new extraction was installed editable into its isolated Python 3.12.16 environment. `ruff check .` passed. `pytest -q` passed with **740 passed, 7 skipped**. Streamlit started from the extracted directory and `curl.exe --fail --silent --show-error http://localhost:8767/_stcore/health` returned `ok`.

The final saved-bundle smoke command also passed from the extracted project root:

```powershell
.venv/Scripts/python.exe scripts/smoke_p23_a1_release.py
```

It loaded all five frozen final bundles and predicted from the packaged 164-row FD001/train engine 4 history through cycle 164, without fitting and without opening raw train files or test labels. Raw RUL outputs were: Decision Tree `29.7941176471`, Random Forest `28.0240895095`, XGBoost `26.1461582184`, constant baseline `107.807862`, and Dense MLP `17.2417659760` cycles. The demo is simulated training data and the values are smoke outputs, not accuracy, official-test results, or maintenance evidence.

## Archive audit and limits

`python scripts/audit_submission_archive.py dist/fleetpulse-submission.zip` passed with zero findings. The audit rejects local caches/environments, credentials, raw datasets, development artifacts, path traversal, and machine-specific paths. No raw NASA dataset is needed for the extracted app/test/smoke checks above.

This run establishes that the candidate package's documented setup, tests, saved-bundle smoke, and local Streamlit health check work from an extracted copy in this environment. It does not establish the current hosted Streamlit commit, a course-platform upload, or a human group sign-off. Keep the exact tested ZIP unchanged for P26-T3; any reassembly requires repeating these checks.
