import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_preprocessor, clean_data, get_feature_columns, split_data

NUM = ["Age", "MonthlyIncome"]
CAT = ["OverTime", "Department"]


def test_numeric_missing_values_are_filled_with_median(small_df):
    out = build_preprocessor(NUM, []).fit_transform(small_df[NUM])
    assert not np.isnan(out).any()
    # row 2 had a missing Age, median of the rest is 33 -> after scaling it
    # should equal the scaled value of 33, i.e. the same as if we had put 33 in
    filled = small_df[NUM].fillna({"Age": 33.0, "MonthlyIncome": 5000.0})
    expected = build_preprocessor(NUM, []).fit_transform(filled)
    np.testing.assert_allclose(out, expected)


def test_categorical_missing_values_are_filled_with_mode(small_df):
    pre = build_preprocessor([], CAT).fit(small_df[CAT])
    out = pd.DataFrame(pre.transform(small_df[CAT]), columns=pre.get_feature_names_out())
    # row 4 had OverTime missing, most common value is "No"
    assert out.loc[4, "OverTime_No"] == 1
    assert out.loc[4, "OverTime_Yes"] == 0


def test_categoricals_are_one_hot_encoded(small_df):
    df = clean_data(small_df, "Attrition")
    pre = build_preprocessor([], CAT).fit(df[CAT])
    out = pd.DataFrame(pre.transform(df[CAT]), columns=pre.get_feature_names_out())
    assert set(out.columns) == {"OverTime_Yes", "OverTime_No",
                                "Department_Sales", "Department_R&D", "Department_HR"}
    assert set(np.unique(out.values)) == {0.0, 1.0}
    # exactly one department and one overtime value per row
    assert (out.sum(axis=1) == 2).all()


def test_unseen_category_is_encoded_as_all_zeros(small_df):
    df = clean_data(small_df, "Attrition")
    pre = build_preprocessor([], CAT).fit(df[CAT])
    new = pd.DataFrame({"OverTime": ["No"], "Department": ["Legal"]})
    out = pd.DataFrame(pre.transform(new), columns=pre.get_feature_names_out())
    dept_cols = [c for c in out.columns if c.startswith("Department_")]
    assert out[dept_cols].sum(axis=1).iloc[0] == 0


def test_clean_data_does_not_change_the_input(small_df):
    original = small_df.copy()
    clean_data(small_df, "Attrition", drop_columns=["EmployeeCount"])
    pd.testing.assert_frame_equal(small_df, original)


def test_clean_data_encodes_target_and_drops_columns(small_df):
    out = clean_data(small_df, "Attrition", drop_columns=["EmployeeCount"])
    assert "EmployeeCount" not in out.columns
    assert out["Attrition"].tolist() == [1, 0, 0, 1, 0, 0]
    assert " Sales " not in out["Department"].values  # whitespace stripped


def test_clean_data_rejects_non_dataframe():
    with pytest.raises(TypeError):
        clean_data([1, 2, 3], "Attrition")
    with pytest.raises(TypeError):
        clean_data(None, "Attrition")


def test_clean_data_rejects_empty_or_missing_target(small_df):
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame(), "Attrition")
    with pytest.raises(ValueError):
        clean_data(small_df.drop(columns=["Attrition"]), "Attrition")


def test_clean_data_rejects_unexpected_labels(small_df):
    df = small_df.copy()
    df.loc[0, "Attrition"] = "Maybe"
    with pytest.raises(ValueError, match="unexpected target"):
        clean_data(df, "Attrition")


def test_get_feature_columns_catches_unlisted_text_column(small_df):
    df = clean_data(small_df, "Attrition")
    with pytest.raises(ValueError):
        get_feature_columns(df, "Attrition", categorical_columns=["OverTime"])
    num, cat = get_feature_columns(df, "Attrition", CAT)
    assert num == ["Age", "MonthlyIncome", "EmployeeCount"]
    assert cat == CAT


def test_split_data_is_stratified(small_df):
    df = pd.concat([clean_data(small_df, "Attrition")] * 10, ignore_index=True)
    train, test = split_data(df, "Attrition", test_size=0.2)
    assert len(train) + len(test) == len(df)
    assert train["Attrition"].mean() == pytest.approx(test["Attrition"].mean(), abs=0.05)
