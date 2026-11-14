"""Train the attrition model and log the run to MLflow.

    python -m src.train --config configs/config.yaml

Metrics are computed two ways:
  * val_*  - mean over stratified k-fold cross-validation on the training split.
             These are used to compare experiments and to check the thresholds.
  * test_* - the held-out test split, scored once with the final model.

Exits with code 1 if the validation metrics do not meet the thresholds in the
config, which is what fails the training job in CI.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import get_data_version, load_data
from src.evaluate import check_thresholds, evaluate_model
from src.preprocess import (
    build_preprocessor,
    clean_data,
    get_feature_columns,
    split_data,
    split_xy,
)

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(model_cfg, random_state=42):
    model_type = model_cfg["type"]
    if model_type not in MODELS:
        raise ValueError(f"unknown model type '{model_type}', choose from {list(MODELS)}")
    params = dict(model_cfg.get("params") or {})
    params.setdefault("random_state", random_state)
    return MODELS[model_type](**params)


def build_pipeline(cfg, numeric_cols, categorical_cols):
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", build_model(cfg["model"], cfg["data"]["random_state"])),
    ])


def cross_validate_pipeline(pipeline, X, y, n_folds=5, random_state=42):
    """Mean validation metrics over stratified k-fold CV (the pipeline is refit per fold)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    scores = cross_validate(clone(pipeline), X, y, cv=cv, scoring=scoring)
    return {name: scores[f"test_{name}"].mean() for name in scoring}


def train(cfg, run_name=None, save_outputs=True):
    data_cfg = cfg["data"]
    target = data_cfg["target"]

    df = clean_data(load_data(data_cfg["path"]), target, data_cfg["drop_columns"])
    train_df, test_df = split_data(df, target, data_cfg["test_size"], data_cfg["random_state"])
    numeric_cols, categorical_cols = get_feature_columns(
        train_df, target, data_cfg["categorical_columns"]
    )
    X_train, y_train = split_xy(train_df, target)
    X_test, y_test = split_xy(test_df, target)
    # integer columns can't hold NaN, so make everything numeric a float for the
    # model signature that mlflow infers
    X_train = X_train.astype({c: "float64" for c in numeric_cols})
    X_test = X_test.astype({c: "float64" for c in numeric_cols})

    pipeline = build_pipeline(cfg, numeric_cols, categorical_cols)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    data_version = get_data_version(data_cfg["dvc_file"])

    with mlflow.start_run(run_name=run_name or cfg["model"]["type"]) as run:
        mlflow.log_param("model_type", cfg["model"]["type"])
        mlflow.log_params(cfg["model"].get("params") or {})
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("test_size", data_cfg["test_size"])
        mlflow.log_param("n_features", len(numeric_cols) + len(categorical_cols))
        mlflow.log_param("dropped_columns", ",".join(data_cfg["drop_columns"]))

        mlflow.log_param("cv_folds", cfg["cv_folds"])
        # experiment runs vs the final model trained from config.yaml
        mlflow.set_tag("stage", "experiment" if not save_outputs else "final")

        # validation metrics on the training split only, this is what experiments
        # are compared on
        val_metrics = cross_validate_pipeline(
            pipeline, X_train, y_train, cfg["cv_folds"], data_cfg["random_state"]
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        failures = check_thresholds(val_metrics, cfg["thresholds"])
        mlflow.set_tag("passed_thresholds", str(not failures))

        # final fit on the whole training split, test split scored once
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # mlflow 3 saves sklearn models with skops and wants the non-sklearn
        # types inside the pipeline listed explicitly
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=X_test.head(5),
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    if save_outputs:
        out = cfg["output"]
        Path(out["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out["model_path"])
        Path(out["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
        with open(out["metrics_path"], "w") as f:
            json.dump({"run_id": run_id, "data_version": data_version,
                       "validation": val_metrics, "test": test_metrics}, f, indent=2)

    return run_id, val_metrics, test_metrics, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id, val_metrics, test_metrics, failures = train(cfg, run_name=args.run_name)

    print(f"run id: {run_id}")
    print(f"  {'metric':<10} {'validation':>11} {'test':>8}")
    for name in val_metrics:
        print(f"  {name:<10} {val_metrics[name]:>11.4f} {test_metrics[name]:>8.4f}")

    if failures:
        print("validation metrics did not meet the thresholds:")
        for msg in failures:
            print("  -", msg)
        sys.exit(1)
    print("all thresholds met")


if __name__ == "__main__":
    main()
