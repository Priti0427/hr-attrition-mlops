from pathlib import Path

import pandas as pd
import yaml


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dvc pull` to get the dataset.")
    # the original IBM file has a BOM in the first column name
    return pd.read_csv(path, encoding="utf-8-sig")


def get_data_version(dvc_file):
    """md5 of the dataset from the .dvc pointer file, logged to MLflow."""
    dvc_file = Path(dvc_file)
    if not dvc_file.exists():
        return "unknown"
    with open(dvc_file) as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]
