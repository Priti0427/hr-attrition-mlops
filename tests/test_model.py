import numpy as np
import pytest

from src.evaluate import evaluate_model
from src.preprocess import clean_data, get_feature_columns, split_data, split_xy
from src.train import build_model, build_pipeline


@pytest.fixture(scope="module")
def split(raw_data, config):
    d = config["data"]
    df = clean_data(raw_data, d["target"], d["drop_columns"])
    train_df, test_df = split_data(df, d["target"], d["test_size"], d["random_state"])
    num, cat = get_feature_columns(train_df, d["target"], d["categorical_columns"])
    # small sample so the test stays fast
    X_train, y_train = split_xy(train_df.sample(500, random_state=0), d["target"])
    X_test, y_test = split_xy(test_df, d["target"])
    return X_train, y_train, X_test, y_test, num, cat


def test_predictions_have_right_type_and_shape(config, split):
    X_train, y_train, X_test, y_test, num, cat = split
    pipe = build_pipeline(config, num, cat).fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_model_beats_minimum_performance(config, split):
    X_train, y_train, X_test, y_test, num, cat = split
    pipe = build_pipeline(config, num, cat).fit(X_train, y_train)
    metrics = evaluate_model(pipe, X_test, y_test)
    # a bit lower than the thresholds in the config because of the small sample
    assert metrics["roc_auc"] >= 0.70, metrics
    assert metrics["recall"] >= 0.40, metrics


@pytest.mark.parametrize("model_type", ["random_forest", "gradient_boosting"])
def test_other_model_types_train(config, split, model_type):
    X_train, y_train, X_test, _, num, cat = split
    cfg = {**config, "model": {"type": model_type, "params": {"n_estimators": 20}}}
    pipe = build_pipeline(cfg, num, cat).fit(X_train, y_train)
    assert pipe.predict(X_test).shape == (len(X_test),)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_model({"type": "svm"})
