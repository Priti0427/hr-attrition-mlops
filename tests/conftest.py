from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    cfg["data"]["path"] = str(ROOT / cfg["data"]["path"])
    return cfg


@pytest.fixture(scope="session")
def raw_data(config):
    path = Path(config["data"]["path"])
    if not path.exists():
        pytest.skip("dataset not found, run `dvc pull` first")
    return pd.read_csv(path, encoding="utf-8-sig")


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "Age": [25, 40, np.nan, 33, 51, 29],
        "MonthlyIncome": [3000.0, 8000.0, 5000.0, np.nan, 12000.0, 4200.0],
        "OverTime": ["Yes", "No", "No", "Yes", np.nan, "No"],
        "Department": ["Sales", "R&D", "R&D", "Sales", "HR", " Sales "],
        "EmployeeCount": [1, 1, 1, 1, 1, 1],
        "Attrition": ["Yes", "No", "No", "Yes", "No", "No"],
    })
