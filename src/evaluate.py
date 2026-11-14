from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_model(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return compute_metrics(y, pred, proba)


def check_thresholds(metrics, thresholds):
    """Return a list of messages for every metric below its minimum."""
    failures = []
    for name, minimum in thresholds.items():
        if metrics[name] < minimum:
            failures.append(f"{name} = {metrics[name]:.3f}, required {minimum}")
    return failures
