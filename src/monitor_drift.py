"""Check for data drift between the training data and new "production" data.

    python -m src.monitor_drift              # simulated production data with drift
    python -m src.monitor_drift --no-drift   # untouched holdout, should pass

Reference = the training split, production = the test split (optionally with
simulated drift applied). Saves an HTML report to reports/ and exits with
code 1 if the share of drifted features is above the threshold in the config.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_data
from src.evaluate import evaluate_model
from src.preprocess import clean_data, get_feature_columns, split_data

warnings.filterwarnings("ignore")

from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402


def simulate_drift(df, seed=7):
    """Pretend a year has passed: pay review, office move, busy period with more
    overtime and travel. Returns a modified copy."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)

    out["MonthlyIncome"] = (out["MonthlyIncome"] * rng.normal(1.18, 0.05, n)).round()
    out["Age"] = (out["Age"] + rng.integers(2, 6, n)).clip(upper=60)
    out["TotalWorkingYears"] = out["TotalWorkingYears"] + rng.integers(1, 4, n)
    out["YearsAtCompany"] = out["YearsAtCompany"] + rng.integers(1, 3, n)

    moved = rng.random(n) < 0.6
    out.loc[moved, "DistanceFromHome"] += rng.integers(4, 11, moved.sum())

    more_overtime = (out["OverTime"] == "No") & (rng.random(n) < 0.3)
    out.loc[more_overtime, "OverTime"] = "Yes"

    more_travel = (out["BusinessTravel"] == "Travel_Rarely") & (rng.random(n) < 0.35)
    out.loc[more_travel, "BusinessTravel"] = "Travel_Frequently"
    return out


def run_drift_report(reference, production, numeric_cols, categorical_cols):
    definition = DataDefinition(numerical_columns=numeric_cols, categorical_columns=categorical_cols)
    ref = Dataset.from_pandas(reference, data_definition=definition)
    prod = Dataset.from_pandas(production, data_definition=definition)
    # with >1000 reference rows evidently uses Wasserstein distance for numeric
    # columns and Jensen-Shannon distance for categorical ones, threshold 0.1
    return Report([DataDriftPreset()]).run(current_data=prod, reference_data=ref)


def drift_table(snapshot):
    """One row per feature from the evidently snapshot."""
    rows = []
    for m in snapshot.dict()["metrics"]:
        c = m["config"]
        if c["type"].endswith("ValueDrift"):
            rows.append({
                "feature": c["column"],
                "method": c["method"],
                "score": round(m["value"], 4),
                "drifted": m["value"] >= c["threshold"],
            })
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-drift", action="store_true", help="use the clean holdout as production data")
    parser.add_argument("--threshold", type=float, help="max share of drifted features (default from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    threshold = args.threshold if args.threshold is not None else cfg["drift"]["share_threshold"]
    scenario = "no_drift" if args.no_drift else "drifted"

    df = clean_data(load_data(d["path"]), d["target"], d["drop_columns"])
    train_df, test_df = split_data(df, d["target"], d["test_size"], d["random_state"])
    numeric_cols, categorical_cols = get_feature_columns(train_df, d["target"], d["categorical_columns"])

    reference = train_df.drop(columns=[d["target"]])
    holdout = test_df.drop(columns=[d["target"]])
    production = holdout if args.no_drift else simulate_drift(holdout)

    snapshot = run_drift_report(reference, production, numeric_cols, categorical_cols)
    table = drift_table(snapshot)
    n_drifted = int(table["drifted"].sum())
    share = n_drifted / len(table)

    reports_dir = Path(cfg["drift"]["reports_dir"])
    reports_dir.mkdir(exist_ok=True)
    html_path = reports_dir / f"drift_report_{scenario}.html"
    snapshot.save_html(str(html_path))
    with open(reports_dir / f"drift_summary_{scenario}.json", "w") as f:
        json.dump({"drift_share": share, "n_drifted": n_drifted,
                   "features": table.to_dict(orient="records")}, f, indent=2)

    print(f"scenario: {scenario}  (reference {len(reference)} rows, production {len(production)} rows)\n")
    print(table.to_string(index=False))
    print(f"\ndrifted features: {n_drifted} of {len(table)}  (share {share:.1%}, threshold {threshold:.0%})")
    print(f"report saved to {html_path}")

    # labels are only known here because the production data is simulated from
    # the holdout, but it shows what the drift does to the model
    model_path = cfg["output"]["model_path"]
    if Path(model_path).exists():
        model = joblib.load(model_path)
        y = test_df[d["target"]]
        cast = {c: "float64" for c in numeric_cols}
        before = evaluate_model(model, holdout.astype(cast), y)
        after = evaluate_model(model, production.astype(cast), y)
        print("\nmodel metrics on holdout vs production data:")
        for k in before:
            print(f"  {k:<10} {before[k]:.3f} -> {after[k]:.3f}")

    if share > threshold:
        print("\nDRIFT DETECTED: share of drifted features is above the threshold")
        sys.exit(1)
    print("\nno significant drift")


if __name__ == "__main__":
    main()
