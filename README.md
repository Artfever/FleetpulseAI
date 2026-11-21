# FleetPulse AI

FleetPulse is a Streamlit application for estimating the remaining useful life (RUL) of simulated NASA C-MAPSS FD001 engine histories and comparing illustrative maintenance schedules. It includes a small tracked training-history demo subset and verified model/evaluation artifacts; full raw datasets and development outputs are provisioned locally and are not all tracked in Git. These simulated-data estimates are not aircraft fault diagnoses or maintenance recommendations. See the [data, feature, and service architecture report](docs/report/p25_a1_data_features_architecture.md) for provenance, leakage boundaries, and implementation details.

## Get started

The project requires Python 3.12. From the repository root, create the
environment with the Python 3.12 interpreter explicitly. On Windows
PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,xgboost,mlp]" --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m streamlit run app/Home.py
```

If the Windows Python launcher does not recognize an installed 3.12 runtime,
create the same environment with uv instead:

```powershell
uv venv --python 3.12 --seed .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,xgboost,mlp]" --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m streamlit run app/Home.py
```

On Linux or macOS, use `python3.12` and the environment's `bin` directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,xgboost,mlp]' --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m streamlit run app/Home.py
```

See the [setup guide](docs/SETUP.md) for Windows, Linux, and macOS commands and the editable-install path contract.

## Reproduce training and evaluation

With the original, checksum-verified `data/raw/train_FD001.txt` available, rebuild development features and train/reload the P06 Decision Tree with:

```powershell
.venv/Scripts/python.exe scripts/run_p06_validation.py
```

This rebuild replaces generated P06 files under the ignored `artifacts/development/` directory; save any local outputs you want to keep before rerunning it.

For final reporting only, the P13 workflow predicts from the five frozen bundles on `FD001/test`, separately pairs the held-out RUL answers, and then calculates the final metrics. Use a new output directory on each run:

```powershell
$run = "artifacts/experiments/p13-$(Get-Date -Format yyyyMMdd-HHmmssfff)"
New-Item -ItemType Directory -Path $run | Out-Null
.venv/Scripts/python.exe -m fleetpulse.evaluation.test --test-data data/raw/test_FD001.txt --output "$run/official_test_predictions.json"
.venv/Scripts/python.exe -m fleetpulse.evaluation.official_test_pairing --predictions "$run/official_test_predictions.json" --rul-answers data/raw/RUL_FD001.txt --output "$run/official_test_pairings.json"
.venv/Scripts/python.exe -m fleetpulse.evaluation.final_test_results --pairings "$run/official_test_pairings.json" --output "$run/final_test_results.json" --residual-figure "$run/test_residual_histogram.html" --prediction-figure "$run/test_prediction_errors.html"
.venv/Scripts/python.exe scripts/audit_p13_metrics.py --predictions "$run/official_test_predictions.json" --rul data/raw/RUL_FD001.txt --reported "$run/final_test_results.json" --output "$run/metric_audit.json"
```

Full data requirements, Linux/macOS variants, exact observed outputs, limitations, and code-matched pipeline diagrams are in the [P25 reproduction and architecture report](docs/report/p25_a2_pipeline_and_reproduction.md). Final test scores are for held-out reporting; do not use them to tune or select models.

For the architecture/data walkthrough, see the [editable P25 slides](docs/slides/p25_abdullah_architecture_data.md) and [presenter guide](docs/slides/p25_abdullah_presenter_guide.md).

## Repository structure

```text
app/                  Streamlit entrypoint and six current application pages
src/fleetpulse/       Data, feature, model, evaluation, service, and optimization packages
tests/                Automated tests
data/                 Tracked demo subset; full raw datasets are local/ignored
artifacts/            Tracked final bundles; development outputs are local/ignored
docs/                 Setup, workflow, and project documentation
planning/             Backlog manifest and issue mapping
scripts/              Backlog and issue-workflow utilities
.github/              CI, issue and pull request templates, and code owners
```

See the [repository layout](docs/REPOSITORY_LAYOUT.md) for the tracked and locally provisioned contents.

## Working on the project

Start with the [contributing guide](docs/CONTRIBUTING.md), [repository instructions](AGENTS.md), and the relevant handoff ([Shaheer](handoff_Shaheer.md), [Abdullah](handoff_Abdullah.md), or [Taha](handoff_Taha.md)). The [issue backlog](docs/ISSUE_BACKLOG.md) and [dependency guide](docs/DEPENDENCIES_AND_MERGE_ORDER.md) describe the planned work. Changes follow the [branch and review workflow](docs/BRANCHING_AND_REVIEWS.md) and [CI checks](docs/CI_CD.md).

The repository currently has no license file. Contact the repository owner before assuming permission to reuse or redistribute its contents.

P25-S1's [model, evaluation, explanation, and scheduling chapter](docs/report/p25_s1_model_evaluation_scheduling.md) brings together the frozen model rationale, checked results, and simulation limits.
