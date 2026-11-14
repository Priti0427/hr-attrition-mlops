"""Create data/raw/hr_attrition.csv from the original IBM attrition file.

The original dataset has no missing values, so this blanks out a few percent
of the cells in some columns (fixed seed). Only needs to be run once; the
result is tracked with DVC.

    python scripts/make_dataset.py emp_attrition.csv
"""

import sys

import numpy as np
import pandas as pd

MISSING = {
    "MonthlyIncome": 0.05,
    "TotalWorkingYears": 0.04,
    "DistanceFromHome": 0.03,
    "EnvironmentSatisfaction": 0.03,
    "BusinessTravel": 0.04,
    "MaritalStatus": 0.03,
}


def add_missing_values(df, plan, seed=42):
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col, share in plan.items():
        idx = rng.choice(out.index, size=int(len(out) * share), replace=False)
        out.loc[idx, col] = np.nan
    return out


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/raw/hr_attrition.csv"
    df = add_missing_values(pd.read_csv(src, encoding="utf-8-sig"), MISSING)
    df.to_csv(dst, index=False)
    print(f"wrote {len(df)} rows to {dst}")
    print(df.isna().sum()[lambda s: s > 0])
