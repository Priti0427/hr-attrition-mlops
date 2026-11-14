"""Checks on the actual dataset (needs `dvc pull`)."""

import pandas as pd

EXPECTED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "JobLevel",
    "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "OverTime",
    "TotalWorkingYears", "YearsAtCompany",
]

# from the IBM data description
NUMERIC_RANGES = {
    "Age": (18, 60),
    "DistanceFromHome": (1, 29),
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "MonthlyIncome": (1000, 20000),
    "PercentSalaryHike": (11, 25),
    "TotalWorkingYears": (0, 40),
    "YearsAtCompany": (0, 40),
}


def test_expected_columns_present(raw_data):
    missing = set(EXPECTED_COLUMNS) - set(raw_data.columns)
    assert not missing, f"missing columns: {missing}"


def test_dataset_is_big_enough(raw_data):
    assert len(raw_data) >= 1000
    assert raw_data.shape[1] >= 9


def test_target_only_has_yes_and_no(raw_data, config):
    target = config["data"]["target"]
    assert raw_data[target].isna().sum() == 0
    assert set(raw_data[target].unique()) == {"Yes", "No"}


def test_numeric_features_in_expected_range(raw_data):
    for col, (lo, hi) in NUMERIC_RANGES.items():
        values = raw_data[col].dropna()
        assert values.min() >= lo, f"{col} min {values.min()} < {lo}"
        assert values.max() <= hi, f"{col} max {values.max()} > {hi}"


def test_categorical_columns_are_text(raw_data, config):
    for col in config["data"]["categorical_columns"]:
        assert not pd.api.types.is_numeric_dtype(raw_data[col]), col


def test_has_some_missing_values_but_not_too_many(raw_data):
    missing_share = raw_data.isna().mean()
    assert missing_share.sum() > 0
    assert missing_share.max() < 0.1
