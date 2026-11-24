# C-MAPSS environment extension

## 2026-11-24 · Scope and starting point

- Started `feat/cmapss-environments` at `origin/dev` commit `fe40076`. Fetched dev and main had the same tree. This work adds FD002–FD004 alongside the original FD001 course scope.
- Preserved the existing Abdullah handoff edit and the local `project data set/` folder. Added a Git ignore rule for the raw dataset folder, which contains official answer files as well as train and test histories.
- Verified local file counts against the loader contract. FD002 has 260 training engines, 53,759 training rows, and 259 test engines. FD003 has 100, 24,720, and 100. FD004 has 249, 61,249, and 248.

## Implementation

- The updated RUL Prediction page uses an Environment radio control above the existing controls. FD001 retains its original source and model choices. Each new environment shows its own XGBoost model and offers a packaged sample, the matching local training file, or a headered 26-column CSV. For local data place files under `data/raw/` or keep them in `project data set/`.
- Selections support engine and observed cycle. The existing result card and 3D replay show actual model output at evaluated cutoffs. Raw output is retained for scoring and planner provenance. The card displays a nonnegative estimate. Changing environment or prediction input invalidates the corresponding saved planner snapshot.
- The three demo files are the first 100 rows of their respective training files with line endings and trailing spaces normalized. They represent engine 1 through cycle 100 and were used as included training examples in the final refit. Model Laboratory displays official metrics separately by subset.
- The new release directory contains native XGBoost JSON weights, a manifest, and aggregate official evaluation per subset. The manifest records the expected feature order, dataset identity, config, preprocessing, library versions, source digest, and model digest. Git preserves the manifest and sample bytes across platforms.
- FD002 and FD004 fit six operating-condition centroids in standardized setting space, then normalize sensor readings per condition. Development preprocessing uses fitting engines only. The refit uses the full corresponding training file. FD003 uses the untransformed feature contract. All three models use latest, mean, population standard deviation, and actual-cycle slope across 24 channels over a 30-cycle interval.

## Reproduce the preset run

Run from the project root in PowerShell with the installed Python 3.12 environment. Use a new `--bundle-root` and `--prediction-root` for later experiments so the released preset stays traceable.

```powershell
$env:OMP_NUM_THREADS = '2'
foreach ($subset in 'FD002','FD003','FD004') {
    foreach ($stage in 'train','predict','score') {
        .venv/Scripts/python.exe -m scripts.train_cmapss_environment $stage $subset --data-dir 'project data set'
        if ($LASTEXITCODE -ne 0) { throw "$subset $stage failed" }
    }
}
```

- The preset uses 100 trees, depth 4, learning rate 0.05, row and column subsampling of 0.8, L2 regularization of 1, minimum child weight 1, seed 42, and two CPU threads. The XGBoost histogram algorithm uses squared error.
- The train stage fits and validates on an approximately 80/20 split by whole engine. It records the selected engines and observation cutoffs in the manifest, then refits the same configuration on all training engines. Targets use final training cycle minus observed cycle.
- The predict stage generates one raw estimate at the final observed cycle for every official test engine and writes the unpaired outputs under `reports/environments/`. The score stage verifies saved test and artifact identities, then opens the matching answer file. It reports MAE, RMSE, signed error, short-RUL error, and a fitting-mean baseline. The official scores are final reporting evidence. Use the development split for future parameter comparisons.

## Completed training and official test results

The train stage finished in 118.24 s for FD002, 78.16 s for FD003, and 334.81 s for FD004. These are local wall clock times. Rewriting manifest line endings for Git portability was followed by new predict and score runs. The preset was fitted and refitted once per environment. Evaluation did not change weights.

| Subset | Validation engines | Validation MAE / RMSE | Test engines | Test MAE / RMSE | Test baseline MAE / RMSE |
|---|---:|---:|---:|---:|---:|
| FD002 | 52 | 26.58 / 37.75 | 259 | 19.62 / 27.15 | 50.86 / 60.16 |
| FD003 | 20 | 24.96 / 33.85 | 100 | 23.08 / 32.78 | 63.27 / 75.18 |
| FD004 | 49 | 27.12 / 37.58 | 248 | 26.02 / 34.96 | 60.04 / 71.83 |

All errors are measured in cycles. The independent local audit paired 607 predictions in engine order, verified the final observed cycle for each, checked file hashes, and independently reproduced MAE, RMSE, and signed error from the saved raw predictions. For official test engines with actual RUL at or below 30 cycles, MAE was 6.26 / 7.31 / 9.03 cycles for FD002 / FD003 / FD004 (61 / 20 / 53 engines). Overall signed errors were +5.52 / +16.82 / +15.63 cycles, so the estimates tended to overpredict on these test cohorts.

The packaged engine 1 through cycle 100 examples returned 51.96570206, 141.19947815, and 234.00703430 raw cycles respectively. These are training-prefix examples.

### Artifact locations

- `artifacts/environments/FD002/`, `FD003/`, and `FD004/` hold weights, manifests, and evaluation summaries.
- `reports/environments/FD00x-official-predictions.json` holds the unpaired per-engine official outputs locally.
- The independent audit ran with `.venv/Scripts/python.exe $audit`, where `$audit` was the local temporary `fleetpulse-environment-audit.py` script. The official reproducer is the committed three-stage CLI above.

## Verification and integration

- `.venv/Scripts/python.exe -m ruff check .` and `git diff --check` passed during development.
- `.venv/Scripts/python.exe -m pytest -q tests/test_prediction_page.py tests/test_upload_page.py tests/test_p21_t2_session_state.py` passed 20 in 113.04 s.
- `.venv/Scripts/python.exe -m pytest -q tests/test_cmapss_environments.py -k 'planner_schedulers or missing_bundle or corrupt_model'` passed 3 in 31.91 s against available bundles.
- `.venv/Scripts/python.exe -m streamlit run app/Home.py --server.headless true --server.address 127.0.0.1 --server.port 18647 --browser.gatherUsageStats false` started and the local health endpoint returned `ok`. Chrome on 1365×900 and 390×844 displayed FD001 and confirmed FD002 and FD003 estimate cards and rendered canvases without horizontal overflow.

- A verification run exposed a stale three-argument location in the new history cache call. Updated that call to use the source SHA-256 cache key and restarted the full suite. Stopped the superseded run after observing its failure. The repeat browser script also narrows its estimate selector to the result card to avoid matching the existing FD001 chart axis.
- The clean staged copy was exported with git checkout-index, using the existing installed interpreter and the exported src directory. It contains the tracked demo and release files without the ignored raw histories or development artifacts.

- PR [#558](https://github.com/Artfever/FleetpulseAI/pull/558) is open against dev. Implementation commit: a42983d. The clean staged copy on port 18649 returned health ok. Chrome at 1365x900 and 390x844 loaded all three packaged histories, returned 51.97 / 141.20 / 234.01 cycles, rendered all three canvases, and had no horizontal overflow.
- The clean copy exercised FD002–FD004 with shipped weights and histories alone. The existing FD001 predictor path still uses its existing provisioned Decision Tree artifact. A local session with that artifact verified the FD001 estimate at desktop and phone widths.

- Clean copy: `.venv/Scripts/python.exe -m pytest -q tests/test_cmapss_environments.py` with the staged source on the Python module path passed all 7 tests in 31.35 s. Chrome opened Model Laboratory's new official test section on the clean copy.
- Final local full command `.venv/Scripts/python.exe -m pytest -q` passed all 762 tests in 348.86 s. `.venv/Scripts/python.exe -m ruff check .` and `git diff --cached --check` passed.
- Implementation revision a42983d passed all five Development CI checks in run [45388587995](https://github.com/Artfever/FleetpulseAI/actions/runs/45388587995). The CI pytest job passed 754 tests and skipped 8 in 35.20 s. There were no review conversations at this check.
- Hosted verification requires the Streamlit deployment to load this revision. The existing FD001 deployment still requires its separately configured Decision Tree bundle. The new environments serve from the tracked release bundles and packaged demo samples.
