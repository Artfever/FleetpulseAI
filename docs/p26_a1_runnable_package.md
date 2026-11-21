# P26-A1 runnable package inventory

Issue: [P26-A1 #105](https://github.com/Artfever/AI_PROJECT/issues/105)

Phase gate: [P26 #31](https://github.com/Artfever/AI_PROJECT/issues/31)
Audited revision: `378a7c41cbbe870db357f783d346c72b487c0047` (`dev`, before this inventory document)

## What the repository supplies

The runnable project is the Git checkout, installed from its root. It is not a
standalone wheel: the Streamlit entrypoint and final model files are repository
paths resolved from `fleetpulse.PROJECT_ROOT`.

| Package part | Tracked contents | Purpose |
| --- | --- | --- |
| Application and package | 76 files under `app/` and `src/fleetpulse/` | Streamlit screens and the importable data, feature, model, evaluation, service, and planning code |
| Scripts | 75 files under `scripts/` | Training, artifact packaging, audits, reproduction, and workflow utilities |
| Tests | 127 files under `tests/` | Unit, contract, artifact, and application checks; synthetic fixtures are documented in `tests/fixtures/README.md` |
| Runtime and development dependencies | `pyproject.toml`, `requirements.txt` | Python 3.12 runtime pins, optional XGBoost/MLP packages, and the `dev` test/Ruff extra |
| Frozen serving bundles | 10 substantive tracked files under `artifacts/final/tree/bundles/` and `artifacts/final/mlp/bundle/` | Five saved predictors, their manifests, and the tree bundle index |
| Evaluation evidence | `artifacts/final/p13/final_test_results.json`, `artifacts/final/explanation/decision-tree-background.json` | Tracked model report and explanation background used by the app/report |
| Data access | `docs/fd001_provenance.md`, `docs/SETUP.md`, `README.md` | NASA source, expected raw file hashes, setup, and reproduction commands |

The final artifact inventory and bundle checksums are listed in
[`artifact_inventory.md`](artifact_inventory.md). Local development candidates,
raw datasets, virtual environments, and caches are not part of the tracked
package. The application demo subset is tracked in `data/demo/`.

## Setup and reproduction contract

From a clean Python 3.12 environment, the README installs the project and its
extras with:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,xgboost,mlp]" --extra-index-url https://download.pytorch.org/whl/cpu
```

The repository's supported commands are `python -m pytest -q`,
`python -m ruff check .`, and `python -m streamlit run app/Home.py`; run them
from the checkout root. Rebuilding model artifacts additionally requires the
original raw FD001 files described in [`fd001_provenance.md`](fd001_provenance.md).
The three files are intentionally not committed. That document records their
NASA source and expected SHA-256 values, plus a command to verify local copies.

For final reported metrics, the README documents the separate prediction,
answer-pairing, metric-generation, and audit commands. Final-test labels are
only used in that reporting workflow; do not use them for model selection or
tuning.

## Observed artifact reload check

On Python 3.12.16, the existing clean-process check was run against the local
checksum-verified `data/raw/train_FD001.txt`:

```powershell
.venv/Scripts/python.exe scripts/smoke_p12_final_bundles_clean_process.py
```

It sent one identical FD001/train engine 4 history through cycle 164 to each
saved final bundle in a separate fresh process. All five bundles loaded and
predicted without fitting or opening official-test data or labels:

| Model version | Raw RUL estimate (cycles) |
| --- | ---: |
| `decision-tree-rul-v1` | 29.79411765 |
| `random-forest-rul-v1` | 28.02408951 |
| `xgboost-rul-v1` | 26.14615822 |
| `constant-rul-v1` | 107.80786196 |
| `dense-rul-mlp-final-v1` | 17.24176598 |

Input history SHA-256: `b8febe5807d0095ae9828f6c8731622113923312df98271e46a361853cff1e97`. The
script wrote `artifacts/final/p12_clean_process_smoke.json` locally; that output
is ignored and is not included in this tracked inventory. These are smoke
outputs for one simulated training-history prefix, not model-quality metrics,
official-test results, or maintenance evidence.

The three local raw files also matched the SHA-256 values in the provenance
record. This confirms this checkout's copies, not their inclusion in a future
submission ZIP.

## Handoff to the final-package tasks

This inventory gives the package assembler a concrete source, test, dependency,
artifact, and dataset-instruction map. It does not create the submission ZIP or
claim that an extracted archive has been tested. P26-A2 must still audit the
archive contents for accidental local files; P26-A3 and Taha's final-package
task must test the exact archive after assembly.
