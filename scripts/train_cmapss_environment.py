"""Build, predict and score the three versioned C-MAPSS environment bundles."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
import xgboost

from fleetpulse.data.loader import load_engine_history, load_rul_values
from fleetpulse.services.environments import (
    BUNDLE_SCHEMA,
    FEATURES,
    NEW_ENVIRONMENTS,
    ConditionTransform,
    ObservedHistory,
    group_histories,
    load_environment_model,
    sha256,
)

PARAMETERS = {
    "device": "cpu",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "random_state": 42,
    "n_jobs": 2,
}
COUNTS = {"FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(predictions, targets):
    error = np.asarray(predictions) - np.asarray(targets)
    short = np.asarray(targets) <= 30
    return {
        "engines": len(error),
        "mae_cycles": float(np.abs(error).mean()),
        "rmse_cycles": float(np.sqrt(np.square(error).mean())),
        "mean_signed_error_cycles": float(error.mean()),
        "short_rul_engines": int(short.sum()),
        "short_rul_mae_cycles": float(np.abs(error[short]).mean()) if short.any() else None,
    }


def read_histories(path, count):
    histories = group_histories(load_engine_history(path))
    if set(histories) != set(range(1, count + 1)):
        raise ValueError(f"{path.name} has unexpected engine IDs.")
    return histories


def fitting_matrix(subset, histories, transform):
    features = []
    labels = []
    for rows in histories:
        history = ObservedHistory.build(subset, rows, transform, validated=True)
        for cycle in history.cycles:
            features.append(history.features(cycle).values)
            labels.append(history.cycles[-1] - cycle)
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def fit_transform(subset, histories):
    if subset == "FD003":
        return None
    return ConditionTransform.fit([row for rows in histories for row in rows])


def train(subset, data, bundle):
    path = data / f"train_{subset}.txt"
    histories = read_histories(path, COUNTS[subset][0])
    rng = np.random.default_rng(42)
    held = sorted(rng.permutation(sorted(histories))[: len(histories) // 5].tolist())
    held_set = set(held)
    fit_ids = sorted(set(histories) - held_set)
    fit_rows = [histories[unit] for unit in fit_ids]
    transform = fit_transform(subset, fit_rows)
    x, y = fitting_matrix(f"{subset}/train", fit_rows, transform)
    estimator = xgboost.XGBRegressor(**PARAMETERS)
    estimator.fit(x, y)
    validation_cases = []
    for unit in held:
        history = ObservedHistory.build(
            f"{subset}/train", histories[unit], transform, validated=True
        )
        minimum = min(len(history.cycles) - 2, FEATURES.window_cycles - 1)
        cutoff = history.cycles[int(rng.integers(minimum, len(history.cycles) - 1))]
        validation_cases.append(
            (unit, cutoff, history.cycles[-1] - cutoff, history.features(cutoff).values)
        )
    targets = [case[2] for case in validation_cases]
    validation_predictions = estimator.predict(np.asarray([c[3] for c in validation_cases]))
    baseline = float(y.mean())
    development = {
        "seed": 42,
        "fitting_engines": fit_ids,
        "fitting_rows": len(y),
        "validation_cutoffs": [{"unit_id": u, "cycle": c} for u, c, _, _ in validation_cases],
        "model": metrics(validation_predictions, targets),
        "constant_mean": metrics([baseline] * len(held), targets),
    }
    all_rows = list(histories.values())
    transform = fit_transform(subset, all_rows)
    x, y = fitting_matrix(f"{subset}/train", all_rows, transform)
    estimator = xgboost.XGBRegressor(**PARAMETERS)
    estimator.fit(x, y)
    bundle.mkdir(parents=True, exist_ok=True)
    estimator.save_model(bundle / "model.json")
    write_json(
        bundle / "manifest.json",
        {
            "schema": BUNDLE_SCHEMA,
            "subset": subset,
            "version": f"{subset.lower()}-xgboost-v1",
            "feature_version": FEATURES.version,
            "window_cycles": FEATURES.window_cycles,
            "feature_names": FEATURES.output_schema,
            "parameters": PARAMETERS,
            "condition_transform": None
            if transform is None
            else {k: v.tolist() for k, v in asdict(transform).items()},
            "model_sha256": sha256(bundle / "model.json"),
            "train_sha256": sha256(path),
            "training_engines": len(histories),
            "training_rows": len(y),
            "library_versions": {
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "development": development,
            "final_constant_mean_rul": float(y.mean()),
        },
    )
    print(f"{subset} validation: {development['model']}", flush=True)


def predict(subset, data, bundle, output):
    path = data / f"test_{subset}.txt"
    histories = read_histories(path, COUNTS[subset][1])
    model = load_environment_model(bundle, subset)
    rows = []
    for unit, history in sorted(histories.items()):
        cutoff = history[-1]["cycle"]
        ((_, value),) = model.predict(history, [cutoff])
        rows.append({"unit_id": unit, "observed_cycle": cutoff, "raw_rul_cycles": value})
    write_json(
        output,
        {
            "subset": subset,
            "model_version": model.version,
            "bundle_sha256": sha256(bundle / "manifest.json"),
            "test_sha256": sha256(path),
            "rows": rows,
        },
    )


def score(subset, data, bundle, output):
    model = load_environment_model(bundle, subset)
    manifest = read_json(bundle / "manifest.json")
    predictions = read_json(output)
    rows = predictions["rows"]
    count = COUNTS[subset][1]
    if (
        predictions["subset"] != subset
        or predictions["model_version"] != model.version
        or predictions["bundle_sha256"] != sha256(bundle / "manifest.json")
        or predictions["test_sha256"] != sha256(data / f"test_{subset}.txt")
        or [row["unit_id"] for row in rows] != list(range(1, count + 1))
    ):
        raise ValueError("Saved predictions do not match the official evaluation contract.")
    histories = read_histories(data / f"test_{subset}.txt", count)
    if any(row["observed_cycle"] != histories[row["unit_id"]][-1]["cycle"] for row in rows):
        raise ValueError("Saved predictions do not use the official observed cutoffs.")
    path = data / f"RUL_{subset}.txt"
    targets = load_rul_values(path, expected_engine_count=count)
    values = [row["raw_rul_cycles"] for row in rows]
    report = {
        "subset": subset,
        "partition": f"{subset}/test",
        "model_version": model.version,
        "model_sha256": manifest["model_sha256"],
        "bundle_sha256": predictions["bundle_sha256"],
        "test_sha256": predictions["test_sha256"],
        "answers_sha256": sha256(path),
        "predictions_sha256": sha256(output),
        "model": metrics(values, targets),
        "constant_mean": metrics([manifest["final_constant_mean_rul"]] * count, targets),
    }
    write_json(bundle / "evaluation.json", report)
    print(f"{subset} official test: {report['model']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "predict", "score"])
    parser.add_argument("subset", choices=NEW_ENVIRONMENTS)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path, default=Path("artifacts/environments"))
    parser.add_argument("--prediction-root", type=Path, default=Path("reports/environments"))
    args = parser.parse_args()
    start = perf_counter()
    bundle = args.bundle_root / args.subset
    output = args.prediction_root / f"{args.subset}-official-predictions.json"
    if args.stage == "train":
        train(args.subset, args.data_dir, bundle)
    elif args.stage == "predict":
        args.prediction_root.mkdir(parents=True, exist_ok=True)
        predict(args.subset, args.data_dir, bundle, output)
    else:
        score(args.subset, args.data_dir, bundle, output)
    print(f"{args.subset} {args.stage}: {perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
