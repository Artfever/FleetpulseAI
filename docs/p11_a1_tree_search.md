# P11-A1: predeclared forest and boosting search

## Protocol

Ran six CPU configurations against the same P06 30-cycle `past-only-v1`
FD001/train tables: three Random Forest and three XGBoost. Within each family,
only `max_depth` changed; seed stayed 42 and other family-specific parameters
stayed fixed. All runs used the same 80 fitting engines and the same 20 saved
validation engines/cutoffs. The original FD001/test source and its RUL answers
were not opened.

Predeclared settings:

| Family | Configurations | Fixed settings |
| --- | --- | --- |
| Random Forest | 100 trees, depths 8/12/16 | seed 42; leaf size 5; `max_features=1.0`; bootstrap; one CPU worker |
| XGBoost | 100 trees, depths 3/4/6 | seed 42; learning rate 0.05; child weight 1; subsample/column sample 0.8; L2 1; histogram CPU; one worker |

Reproduce with:

```powershell
.venv/Scripts/python.exe scripts/run_p11_a1_tree_search.py
```

The script reads the saved feature tables, validates their source, split, and
30-cycle feature provenance, refuses to overwrite a nonempty experiment
folder, and writes a separate model and per-engine `results.json` for every
run under the Git-ignored `artifacts/experiments/P11_A1_tree_search/`.

## Results

Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1. Runtime is observed local
fit plus validation-prediction time and is not a cross-machine benchmark.

| Family | Depth | Validation MAE (cycles) | Validation RMSE (cycles) | Fit/predict (s) | Reload replay |
| --- | ---: | ---: | ---: | ---: | --- |
| Random Forest | 8 | 18.845563 | 24.877649 | 28.328 | 20/20 exact |
| Random Forest | 12 | 18.833479 | 25.243023 | 46.708 | 20/20 exact |
| Random Forest | 16 | **18.360242** | **24.563262** | 37.547 | 20/20 exact |
| XGBoost | 3 | 16.063677 | 21.616009 | 1.463 | 20/20 exact |
| XGBoost | 4 | **15.108697** | **21.092113** | 1.373 | 20/20 exact |
| XGBoost | 6 | 17.008195 | 23.231008 | 2.411 | 20/20 exact |

Each serialized model was reloaded in a fresh process and predicted the same 20
validation rows; its outputs matched the corresponding saved per-engine
predictions exactly. Depth 16 led this Random Forest subset; depth 4 led the
XGBoost subset and all six runs on both reported metrics. These observations
inform P11's selection discussion but do not finalize a production model.

## Limits and provenance

This is a small, predeclared search on one fixed development split of only 20
validation engines. It is not official-test performance, a new split, or a
guarantee of generalization. Other P11 experiments and the shared P11 gate must
be considered before the final configuration decision.

- Training source SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Semantic P04 split SHA-256: `6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
- Fitting table file SHA-256: `c87fb2f2039fe515cb7c5566cd0b7b87f43aa64bc56a7ff31e59e5d7a95be665`.
- Validation table file SHA-256: `b241a6ec5a72cade8925ed40802e2d118871f71eee1622beec18f6dc8b869487`.

All model files and per-engine experiment results are local ignored artifacts;
the tracked script and this report retain the protocol and summary for review.
