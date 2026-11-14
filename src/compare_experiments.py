"""Compare MLflow runs on their validation metrics and pick the best one.

Runs are ranked by the cross-validated metric only. The test metrics are
printed just for the selected run, so the test split is not used to choose
between experiments.
"""

import argparse

import mlflow
import pandas as pd

from src.config import load_config

COLUMNS = [
    "tags.mlflow.runName",
    "params.model_type",
    "params.data_version",
    "metrics.val_roc_auc",
    "metrics.val_f1",
    "metrics.val_recall",
    "metrics.val_precision",
    "tags.passed_thresholds",
]
TEST_METRICS = ["roc_auc", "f1", "recall", "precision", "accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--save", help="write the table to this markdown file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    metric = f"metrics.val_{args.metric}"

    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["experiment_name"]],
        filter_string="attributes.status = 'FINISHED' and tags.stage = 'experiment'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty or metric not in runs.columns:
        print("no runs with validation metrics found, run `python -m src.run_experiments` first")
        return

    # keep only the latest run per name, otherwise re-running an experiment
    # shows up as a duplicate row
    runs = runs.drop_duplicates("tags.mlflow.runName", keep="first")
    runs = runs.sort_values(metric, ascending=False).reset_index(drop=True)

    table = runs[[c for c in COLUMNS if c in runs.columns]]
    table.columns = [c.split(".")[-1] for c in table.columns]
    pd.set_option("display.width", 200)
    print(f"{len(runs)} runs, sorted by val_{args.metric} ({cfg['cv_folds']}-fold CV on the training split)\n")
    print(table.to_string(index=False, float_format="{:.4f}".format))

    passed = runs[runs["tags.passed_thresholds"] == "True"]
    best = passed.iloc[0] if not passed.empty else runs.iloc[0]
    name = best["tags.mlflow.runName"]
    print(f"\nselected run (best val_{args.metric} among runs that passed the thresholds): "
          f"{name}, run id {best['run_id']}")
    print("test metrics of the selected run (held-out split, scored once):")
    for m in TEST_METRICS:
        print(f"  {m:<10} {best[f'metrics.test_{m}']:.4f}")

    if args.save:
        with open(args.save, "w") as f:
            f.write(f"# Experiment comparison\n\nRanked by val_{args.metric} "
                    f"({cfg['cv_folds']}-fold stratified CV on the training split).\n\n")
            f.write(table.to_markdown(index=False, floatfmt=".4f"))
            f.write(f"\n\nSelected run: **{name}**\n\nTest metrics of the selected run:\n\n")
            f.write("| metric | test |\n|---|---|\n")
            for m in TEST_METRICS:
                f.write(f"| {m} | {best[f'metrics.test_{m}']:.4f} |\n")
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
