"""List all MLflow runs and pick the best one by the primary metric (ROC AUC)."""

import argparse

import mlflow
import pandas as pd

from src.config import load_config

COLUMNS = [
    "tags.mlflow.runName",
    "params.model_type",
    "params.data_version",
    "metrics.roc_auc",
    "metrics.f1",
    "metrics.recall",
    "metrics.precision",
    "tags.passed_thresholds",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--save", help="write the table to this markdown file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["experiment_name"]],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{args.metric} DESC"],
    )
    if runs.empty:
        print("no runs found, run `python -m src.run_experiments` first")
        return

    table = runs[[c for c in COLUMNS if c in runs.columns]]
    table.columns = [c.split(".")[-1] for c in table.columns]
    pd.set_option("display.width", 200)
    print(f"{len(runs)} runs, sorted by {args.metric}\n")
    print(table.to_string(index=False, float_format="{:.4f}".format))

    best = runs.iloc[0]
    print(f"\nbest run by {args.metric}: {best['tags.mlflow.runName']} "
          f"({best[f'metrics.{args.metric}']:.4f}), run id {best['run_id']}")

    # the top run can still fail another threshold (e.g. recall), so also show
    # the best run that passed all of them
    passed = runs[runs["tags.passed_thresholds"] == "True"]
    if not passed.empty:
        b = passed.iloc[0]
        print(f"best run that passed all thresholds: {b['tags.mlflow.runName']} "
              f"({b[f'metrics.{args.metric}']:.4f}), run id {b['run_id']}")

    if args.save:
        with open(args.save, "w") as f:
            f.write(f"# Experiment comparison (sorted by {args.metric})\n\n")
            f.write(table.to_markdown(index=False, floatfmt=".4f"))
            f.write(f"\n\nBest run by {args.metric}: **{best['tags.mlflow.runName']}**\n")
            if not passed.empty:
                f.write(f"\nBest run that passed all thresholds: **{b['tags.mlflow.runName']}**\n")
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
