# FleetPulse AI

FleetPulse is a planned Streamlit application for predicting the remaining useful life of simulated turbofan engines and exploring maintenance schedules. It is currently a bootstrap: no dataset, trained model, predictions, or evaluation results are included.

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

## Repository structure

```text
app/                  Streamlit entrypoint and future pages
src/fleetpulse/       Python package and planned domain modules
tests/                Automated tests
data/                 Local datasets (contents ignored by Git)
artifacts/            Local model and experiment outputs (contents ignored by Git)
docs/                 Setup, workflow, and project documentation
planning/             Backlog manifest and issue mapping
scripts/              Backlog and issue-workflow utilities
.github/              CI, issue and pull request templates, and code owners
```

Empty data and artifact folders are kept using `.gitkeep` files. See the [repository layout](docs/REPOSITORY_LAYOUT.md) for details and the distinction between planned and implemented components.

## Working on the project

Start with the [contributing guide](docs/CONTRIBUTING.md), [repository instructions](AGENTS.md), and the relevant handoff ([Shaheer](handoff_Shaheer.md), [Abdullah](handoff_Abdullah.md), or [Taha](handoff_Taha.md)). The [issue backlog](docs/ISSUE_BACKLOG.md) and [dependency guide](docs/DEPENDENCIES_AND_MERGE_ORDER.md) describe the planned work. Changes follow the [branch and review workflow](docs/BRANCHING_AND_REVIEWS.md) and [CI checks](docs/CI_CD.md).

The repository currently has no license file. Contact the repository owner before assuming permission to reuse or redistribute its contents.
