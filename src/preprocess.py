import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(df, target, drop_columns=()):
    """Drop useless columns and turn the target into 0/1. Returns a new dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("dataframe is empty")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not in dataframe")

    out = df.copy()
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])

    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip()

    out = out.dropna(subset=[target])
    labels = set(out[target].unique())
    if not labels <= {"Yes", "No"}:
        raise ValueError(f"unexpected target values: {labels - {'Yes', 'No'}}")
    out[target] = (out[target] == "Yes").astype(int)
    return out.reset_index(drop=True)


def get_feature_columns(df, target, categorical_columns):
    """Return (numeric, categorical) feature names."""
    categorical = [c for c in categorical_columns if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical and c != target]
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"columns {bad} are text but not listed as categorical")
    return numeric, categorical


def build_preprocessor(numeric_columns, categorical_columns):
    """Impute missing values, scale numeric columns and one-hot encode categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, numeric_columns), ("cat", categorical, categorical_columns)],
        verbose_feature_names_out=False,
    )


def split_data(df, target, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_xy(df, target):
    return df.drop(columns=[target]), df[target]
