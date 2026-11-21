# P23-T1 full release package

## Package contents

The full release requires the five frozen P12 models used by the Model Laboratory:

- Decision Tree, Random Forest, XGBoost, and the constant mean baseline under `artifacts/final/tree/bundles/`.
- Dense MLP manifest and CPU weights under `artifacts/final/mlp/bundle/`.
- The approved 229-row FD001/train parity subset under `data/demo/fd001_training_prefixes.txt`.

The bundle manifest and model bytes are checksum-verified by the existing loaders. The final-model registry uses paths relative to the repository root, so inference does not require the original source training file or a developer-specific absolute path. The only raw history shipped for the public parity demo is the owner-approved subset; full training histories, official test histories/labels, development candidates, P13 evaluation exports, and refit intermediates remain excluded.

`requirements.txt` installs the pinned project dependencies and the `xgboost` and `mlp` extras. The app uses XGBoost and PyTorch to load those two final bundles; without the extras the full comparison screen cannot load all five models. Python is constrained to 3.12 by `pyproject.toml`. Streamlit Community Cloud still requires the app owner to select the repository, branch, `app/Home.py`, and Python version in the hosting settings; deployment on a real account is not established by this repository package change.

## Verification status

Run `.venv/Scripts/python.exe scripts/audit_p13_bundles.py` to compare the present local bundles with the frozen P12 inventory and P13 artifact provenance. Run `.venv/Scripts/python.exe scripts/smoke_p12_final_bundles_clean_process.py` to load and predict with all five models in separate clean processes using the packaged demo history. These checks require the local P13 report outputs for the full P13 audit; they do not read official test labels during inference. A successful local check does not establish that the public Streamlit deployment is reachable, uses this commit, or can load its cloud packages.

## Hosting evidence needed to finish P23-T1

Record the deployed branch and commit, app URL, runtime/dependency build results, and a cold-start Model Laboratory result from the account owner. The current public app's anonymous-access status must be checked independently. Do not mark the task complete from a local smoke test alone.
